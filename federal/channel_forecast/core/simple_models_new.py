import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

from OMA_tools.federal.channel_forecast.grid_preprocessing import *
from OMA_tools.federal.channel_forecast.core.pipelines import *
from OMA_tools.federal.channel_forecast.core.content_matching import *
from OMA_tools.federal.channel_forecast.calculator import *
from OMA_tools.io_data.time_series import TimeSeriesTransformer
from OMA_tools.io_data.colors import *

import locale
locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')

import warnings
warnings.filterwarnings('ignore')



class DataPreparator:
    """
        Класс для подготовки и отбора данных перед прогнозированием.
        Отвечает за фильтрацию, разделение по размерам, добавление признаков.
    """
    def __init__(
            self, 
            channel: str,
            palomars_df: pd.DataFrame,
            vimb_df: pd.DataFrame,
            vocabulary: pd.DataFrame
        ):
        self.channel = channel
        self.palomars_df = palomars_df
        self.vimb_df = vimb_df
        self.vocabulary = vocabulary

    

    def merge_programs_by_name(self, data_dict):
        """
            Объединяет датафреймы по названиям программ из всех дат
            
            Args:
                data_dict: словарь с структурой {дата: {программа: датафрейм}}
            
            Returns:
                словарь {название_программы: объединенный_датафрейм}
        """
        # Создаем словарь для накопления датафреймов по программам
        programs_collection = defaultdict(list)
        
        # Проходим по всем датам
        for date, programs_dict in data_dict.items():
            # Проходим по всем программам в этой дате
            for program_name, df in programs_dict.items():
                # Добавляем датафрейм в список для этой программы
                programs_collection[program_name].append(df)
        
        # Объединяем датафреймы для каждой программы
        result_dict = {}
        for program_name, df_list in programs_collection.items():
            if df_list:  # Если есть датафреймы для этой программы
                # Объединяем все датафреймы
                combined_df = pd.concat(df_list, ignore_index=True)
                # Сортируем по дате (если нужно)
                combined_df = combined_df.sort_values('Дата')
                # Удаляем дубликаты (если данные могут повторяться)
                combined_df = combined_df.drop_duplicates()
                
                result_dict[program_name] = combined_df
        
        return result_dict
    

    def aggregate_vimb_daily(self):
        """
            Метод для схлопывания программ VIMB для каждого дня. 
            Один из методов подготовки таблицы для прогнозирования.
            Это процедуру нудно делать, тк при прогнозировании мы опираемся на историческую смэтченную сетку Palomars и VIMB.
        """
        vimb_init_copy = self.vimb_df.copy()
        
        # Список из схлопнутых программ для каждого дня
        new_vimb = []
        
        dates_unique = vimb_init_copy['Дата'].unique()
        
        for date in dates_unique:
        
            table_vimb = vimb_init_copy[vimb_init_copy['Дата'] == date].reset_index(drop = True)
            
            #Программы в VIMB
            vimb_prepr = GeneralTextCleaner(self.channel)
            vimb_prgms, VIMB = vimb_prepr.clean_dataframe(table_vimb)
            
            VIMB = VIMB[['Дата', 'Название программы', 'program_name', 'Время выхода', 'Время окончания']]
            
            # Находим базовые названия программ. Производим замену
            base_names = ProgramMatcher.find_common_base_names(VIMB['program_name'].tolist())
            
            VIMB['Базовое_название'] = VIMB['program_name'].map(base_names)
            
            # Отсавляем только нужные столбцы для анализа
            VIMB = VIMB[['Дата', 'Базовое_название', 'Время выхода', 'Время окончания', 'Название программы']]
            VIMB.rename(columns = {
                'Название программы': 'Название программы vimb',
                'Базовое_название': 'Название программы'
            }, inplace = True)
            
            # Схлопывание программ по дню
            tv_processor = TVScheduleProcessor(self.channel, VIMB, self.palomars_df)
            vimb_joined = tv_processor.join_broadcasts(VIMB, 'vimb', include_share = False)
        
            new_vimb.append(vimb_joined)
        
        # Соединяем все в единый датафрейм
        data = pd.concat(new_vimb)
        
        # Делаем верную сортировку по дате и времени выхода программы
        data['Дата'] = pd.to_datetime(data['Дата'])
        sorted_webs = data.sort_values('Дата').reset_index(drop = True)
        
        res = []
        for date in data['Дата'].unique():
            
            date_dt = pd.to_datetime(date)
            t = sorted_webs[sorted_webs['Дата'] == date_dt]
            t['sort_key'] = t['Время выхода'].apply(BaseParser.get_sort_key)
            final = t.sort_values('sort_key').reset_index(drop = True)
            final = final.drop('sort_key', axis = 1)
            
            res.append(final)
        
        # Итог со схлопывавнием программ в оригинальной сетке
        self.vimb_analysis = pd.concat(res).reset_index(drop = True)
        self.vimb_analysis['Дата'] = self.vimb_analysis['Дата'].dt.strftime('%Y-%m-%d')
        return self.vimb_analysis
    

    def find_and_categorize_programs(self, year: int, month_num: int):
        """
            Поиск схожих программ в соответствии со схлопнутой исторической сеткой Mediascope, 
            а также составление больших, маленьких датафреймов. 
            Осуществляем поиск новых программ, которые раннее не встречались.

            Параметры:
            ----------
                year : int
                    Год, который будем отбирать. Отбирается весь текущий год, если прогнозируемый месяц не Январь. 
                    В противном случае весь прошлый год.
                month_num: int
                    Номер месяца, на который будем строить прогноз.
        """
        vimb_full = self.vimb_analysis.copy()

        grid_hist = self.palomars_df.copy()

        # Если прогнозируемый месяц январь, то отбираем весь прошлый год. В противном случае весь текущий
        if month_num == 1:
            grid_hist = grid_hist[grid_hist['Дата'] >= f'{year - 1}-01-01'].reset_index(drop = True)
        else:
            grid_hist = grid_hist[grid_hist['Дата'] >= f'{year}-01-01'].reset_index(drop = True)

        palomars_prgms = list(set(grid_hist['program_name']))

        need_forecast = {}
        self.new_programs = {}
        results = {}
        vimb_converted = {}

        # Отбираем уникальные даты в сетке VIMB
        dates_unique = self.vimb_analysis['Дата'].unique()

        for date in dates_unique:
            
            # Отбор конкретной даты в ВИМБ
            vimb = vimb_full[vimb_full['Дата'] == date].reset_index(drop = True)
            vimb_prgms = list(set(vimb['Название программы']))
            vimb.rename(columns = {'Название программы': 'program_name'}, inplace = True)
            
            # Делаем поиск по схожим программам
            similar = CosineSimilarity(palomars_prgms, vimb_prgms, grid_hist, vimb)
            # Составление таблицей со схожестью (similarity)
            result, not_found, comparison = similar.comparison(self.vocabulary, min_similarity = 0.5, use_vocabulary = True)

            results[date] = result

            df = result[result['similarity'].round(5) != 0.00000]
            features_dict = similar.generate_similar_features(df, print_df = False)
            need_forecast[date] = features_dict

            new_ones = result[result['similarity'].round(5) == 0.00000]
            if len(new_ones) != 0:
                self.new_programs[date] = new_ones
            
        # Программы, для которых нашлась история
        self.merged_dict = self.merge_programs_by_name(need_forecast)
        
        return self.merged_dict, self.new_programs
    

    def separate_programs_by_volume(self, window_size: int = 7):
        """
            Разделение программ на большие/маленькие датафреймы в зависимости от размера истории
        """
        small = {}
        big = {}
        not_found = pd.DataFrame()

        for program, data in self.merged_dict.items():
            
            data['Дата'] = pd.to_datetime(data['Дата'])
            data['День недели'] = data['Дата'].dt.strftime('%A')
            
            #Отбираем только ненулевые элементы
            if program != 0:
                if len(data) < 3 * window_size:
                    small[program] = data.reset_index(drop = True)
                else:
                    big[program] = data.reset_index(drop = True)

        # Для новых программ
        result = []
        for date, df in self.new_programs.items():
            
            vimb_copy = self.vimb_analysis.copy()
            
            # Отбираем новые программы для конкретного дня
            programs = list(df['Программа VIMB'])
            vimb_copy['Flag'] = self.vimb_analysis['Название программы'].str.contains('|'.join(programs))
            found = vimb_copy[vimb_copy['Flag'] == True].reset_index(drop = True)

            data = found[found['Дата'] == date]
            data_ = data.drop('Flag', axis = 1)
            result.append(data_)

        if len(result) != 0:
            not_found = pd.concat(result).reset_index(drop = True)

            # Добавляем столбец с днем недели
            not_found['Дата'] = pd.to_datetime(not_found['Дата'])
            not_found['День недели'] = not_found['Дата'].dt.strftime('%A')

            not_found = not_found[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'День недели']]
            not_found['Share'] = ''

            new = {}
            programs = not_found['Название программы'].unique()
            for program in programs:
                df = not_found[not_found['Название программы'] == program].reset_index(drop = True)
                new[program] = df
        
        return small, big, new
    

    def prepare(self, year: int, month_num: int):
        """
            Пайплайн для подготовки данных.

            Параметры:
            ----------
                year : int
                    Год, который будем отбирать. Отбирается весь текущий год, если прогнозируемый месяц не Январь. 
                    В противном случае весь прошлый год.
                month_num: int
                    Номер месяца, на который будем строить прогноз.
            
            Returns:
            ----------
                small: dict
                    Словарь из маленьких программ
                big: dict
                    Словарь из крупных программ
                not_found: pd.DataFrame
                    Таблица с ненайденными программами

        """
        # Шаг 1. Схлопываем программы для каждого дня в таблице VIMB
        self.vimb_analysis = self.aggregate_vimb_daily()

        # Шаг 2. Поиск схожих программ
        self.merged_dict, self.new_programs = self.find_and_categorize_programs(year, month_num)

        # Шаг 3. Разделение программ на большие, маленькие датафреймы, а также поиск новых программ
        small, big, not_found = self.separate_programs_by_volume()

        result = {
            'small': small,
            'big': big,
            'new': not_found
        }
        return result



class PrimitiveModel:
    """
        Класс с реализацией простейшей модели для прогнозирования будущих программ.
    """
    def __init__(self, 
                 channel: str,
                 current_year: int,
                 start_date_forecast: str, 
                 palomars_history: pd.DataFrame
                ):
        self.channel = channel
        self.current_year = current_year
        self.start_date_forecast = start_date_forecast
        self.palomars_history = palomars_history

        # Константы
        self.SMALL_SAMPLE_SIZE = 5
        self.EXACT_MATCH_SIZE = 3
        self.DURATION_TOLERANCE = 0.7

    @staticmethod
    def add_duration(df):
        """
            Метод для добавления продолжительности программ. В результате в таблице появляется новый столбец "Продолжительность"
        """
        # Считаем длительности программ
        df['Время выхода_dt'] = pd.to_datetime(df['Время выхода'])
        df['Время окончания_dt'] = pd.to_datetime(df['Время окончания'])
    
        # Автоматически корректируем переход через полночь
        df['Время окончания_dt'] = np.where(
            df['Время окончания_dt'] < df['Время выхода_dt'],
            df['Время окончания_dt'] + pd.Timedelta(days = 1),
            df['Время окончания_dt']
        )
    
        # Расчет продолжительности в секундах
        df['duration_in_sec'] = (
            df['Время окончания_dt'] - df['Время выхода_dt']
        ).dt.total_seconds()
        
        # Добавляем столбец с продолжительностью в минутах (целое число)
        df['dur_min'] = (df['duration_in_sec'] / 60).round().astype(int)
        
        # Форматирование в ЧЧ:ММ:СС
        df['Продолжительность'] = df['duration_in_sec'].apply(
            lambda x: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
        )
        
        # Удаляем вспомогательные столбцы
        df_new = df.drop(['Время выхода_dt', 'Время окончания_dt', 'duration_in_sec'], axis=1)
        
        return df_new

    
    @staticmethod
    def build_russian_holidays(holidays_path: str):
        """
            Генератор праздников РФ
            Args:
                holidays_path: путь к файлу json, в котором перечислены все празднику согласно производственному календарю
            Returns: 
                holidays['working_saturdays']: список из рабочих суббот
                all_holidays: список праздников за исключением рабочих суббот
        """
        with open(holidays_path, 'r', encoding = 'utf-8') as file:
            holidays = json.load(file)

        all_holidays = []
        for year in list(holidays.keys())[1:]:
            all_holidays.extend(holidays[year])
        return holidays['working_saturdays'], all_holidays
    

    @staticmethod
    def get_day_type(date, holidays: list, working_saturdays: list):
        date_str = datetime.strftime(date, '%Y-%m-%d')
        
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        weekday = date_obj.weekday()
        
        if date_str in working_saturdays:
            return 'Будни'
            
        elif date_str in holidays:
            return 'Выходной'
            
        elif weekday < 5:
            return 'Будни'
            
        else:
            return 'Выходной'
    

    @staticmethod
    def get_clean_mean(series):
        """
            Обработать выбросы и вернуть среднее значение
        """
        if len(series) == 0:
            return 0.0
        clean_shares = TimeSeriesTransformer(series).replace_outliers_with_median()
        return np.mean(clean_shares)


    def calculate_share(
            self, 
            data: pd.DataFrame, 
            source_mask: pd.Series = None, 
            parent_df: pd.DataFrame = None
        ):
        """
            Расчет средней доли с возвратом использованной маски
            Параметры:
            ----------
                data: pd.DataFrame
                    Отфильтрованные данные
                source_mask: pd.Series
                    Исходная маска (опционально)
                parent_df: pd.DataFrame
                    Родительский DataFrame для создания маски
            Returns:
            ----------
                share: float
                    Прогнозная доля
                source_mask:
                    Маска, которая использовалась для прогнозирования
        """
        if len(data) == 0:
            return 0.0, None
        
        if len(data) >= self.SMALL_SAMPLE_SIZE:
            # Берем последние 5 записей
            data_tail = data.tail(self.SMALL_SAMPLE_SIZE)
            
            if source_mask is not None:
                # Создаем маску на основе source_mask, но оставляем только tail индексы
                used_mask = pd.Series(False, index = source_mask.index)
                tail_indices = data_tail.index
                used_mask[tail_indices] = True
            elif parent_df is not None:
                # Создаем маску на основе parent_df
                used_mask = pd.Series(False, index = parent_df.index)
                used_mask[data_tail.index] = True
            else:
                used_mask = None
                
            return float(np.mean(data_tail['Share'])), used_mask
        
        # Для маленькой выборки
        share = PrimitiveModel.get_clean_mean(data['Share'])
        return share, source_mask
    

    def find_by_duration(
        self,
        data: pd.DataFrame, 
        dur_min: int, 
        use_tolerance: bool = False,
        base_mask: pd.Series = None
    ):
        """
            Поиск по длительности с возвратом маски

            Параметры:
            ----------
        """
        if use_tolerance:
            mask = (data['dur_min'] >= dur_min * self.DURATION_TOLERANCE) & \
                (data['dur_min'] <= dur_min / self.DURATION_TOLERANCE)
        else:
            mask = data['dur_min'] == dur_min
        
        # Сохраняем индексы до reset_index
        original_indices = data[mask].index
        
        filtered_data = data[mask].reset_index(drop=True)
        
        # Комбинируем маски
        if base_mask is not None:
            combined_mask = base_mask.copy()
            # Оставляем только те индексы, которые прошли фильтр длительности
            combined_mask[~combined_mask.index.isin(original_indices)] = False
        else:
            combined_mask = pd.Series(False, index = data.index)
            combined_mask[original_indices] = True
        
        share, _ = self.calculate_share(filtered_data, combined_mask, parent_df = data)
        return share, combined_mask if combined_mask.any() else None


    @staticmethod
    def generate_field_combinations(
                                fields, 
                                min_fields = 2, 
                                must_include = None, 
                                exclude = None, 
                                debug = False
                            ):
        """
            Универсальная функция для генерации комбинаций полей
            
            Параметры:
            ----------
            fields : list
                Список всех доступных полей
            min_fields : int, default=2
                Минимальное количество полей в комбинации
            must_include : list or str, optional
                Поле(я), которые ДОЛЖНЫ быть в каждой комбинации
            exclude : list or str, optional
                Поле(я), которые НЕ ДОЛЖНЫ участвовать в комбинациях
            debug : bool, default=False
                Печатать отладочную информацию
            
            Returns:
            --------
            list
                Список комбинаций полей в порядке убывания длины
        """
        # Нормализуем входные параметры
        if must_include is None:
            must_include = []
        elif isinstance(must_include, str):
            must_include = [must_include]
        
        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
        
        # Исключаем ненужные поля
        working_fields = [f for f in fields if f not in exclude]
        
        # Отделяем обязательные поля от опциональных
        mandatory = [f for f in must_include if f in working_fields]
        optional = [f for f in working_fields if f not in mandatory]
        
        combinations_list = []
        
        # Минимальное количество опциональных полей
        min_optional = max(0, min_fields - len(mandatory))
        
        # Генерируем комбинации опциональных полей
        # Идем от максимального количества к минимальному
        for i in range(len(optional), min_optional - 1, -1):
            for combo in combinations(optional, i):
                # Добавляем обязательные поля
                full_combo = mandatory + list(combo)
                combinations_list.append(full_combo)
        
        if debug:
            print(f"\n{'='*50}")
            print("ПАРАМЕТРЫ ГЕНЕРАЦИИ:")
            print(f"{'=' * 50}")
            print(f"Всего полей: {fields}")
            print(f"Исключены: {exclude if exclude else 'нет'}")
            print(f"Обязательные: {mandatory if mandatory else 'нет'}")
            print(f"Опциональные: {optional}")
            print(f"Минимальное количество полей: {min_fields}")
            print(f"{'='*50}")
            print(f"Сгенерировано комбинаций: {len(combinations_list)}")
            print(f"{'='*50}")
            
            if combinations_list:
                print("\nКОМБИНАЦИИ (в порядке приоритета):")
                for i, combo in enumerate(combinations_list, 1):
                    print(f"{i:2d}. {combo}")
            print(f"{'=' * 50}\n")
        
        return combinations_list
    

    def prepare_forecast_inputs(
            self, 
            df: pd.DataFrame,
            program_name: str,
            all_holidays, 
            work_saturdays, 
            n_weeks_ago: int = 3,
            debug = False
            ):
        """
            Строит датасет для программы, добавляет признаки.
            
            Параметры:
            ----------
                df: pd.DataFrame
                    Таблица, с которой будем работать
                program_name : str
                    Название программы
                all_holidays : 
                    Российские праздники
                work_saturdays :
                    Рабочие субботы РФ
                n_weeks_ago : int, default = 3
                   Количество последних недель, которые идут в расчет
            
            Returns:
            --------
            pd.DataFrame
                Таблица из последних N недель для какой-то программы

        """
        palomars_last_n_weeks = pd.DataFrame()
        last_n_weeks = pd.DataFrame()

        palomars_history = self.palomars_history.copy()
        palomars_history['Дата'] = pd.to_datetime(palomars_history['Дата'])
        palomars_history['День недели'] = palomars_history['Дата'].dt.strftime('%A')
        if 'program_name' in palomars_history.columns:
            palomars_history.rename(columns = {'program_name': 'Название программы'}, inplace = True)

        
        df_copy = df.copy()

        # Шаг 2. Добавление длительности программ.
        data = PrimitiveModel.add_duration(df_copy)
        palomars_history = PrimitiveModel.add_duration(palomars_history)

        # Шаг 3. Определение типа дня
        data['Тип дня'] = data['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))
        # 0 - Будни, 1 - Выходные
        data['Тип дня'] = LabelEncoder().fit_transform(data['Тип дня'])

        palomars_history['Тип дня'] = palomars_history['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))
        # 0 - Будни, 1 - Выходные
        palomars_history['Тип дня'] = LabelEncoder().fit_transform(palomars_history['Тип дня'])

        if debug:
            print('=============== ЗАПУСКАЮ ДЕБАГГЕР ===============\n')
            print(f'Анализ программы: {program_name}')
        
        columns_order = []
        if 'Жанр' in df.columns:
            columns_order = [
                'Дата', 'Название программы', 'Время выхода', 
                'Время окончания', 'Продолжительность', 'Жанр',
                'dur_min', 'День недели', 'Тип дня', 'Share'
                        ]
        else:
            columns_order = [
                'Дата', 'Название программы', 'Время выхода', 
                'Время окончания', 'Продолжительность',
                'dur_min', 'День недели', 'Тип дня', 'Share'
                        ]

        
        # Шаг 4. Отбираем ТОЛЬКО те даты, которые нужно спрогнозировать
        per_forecast = data[data['Share'] == ''].reset_index(drop = True)
        per_forecast = per_forecast[columns_order]
        if debug:
            print(f'Всего требуется спрогнозировать {len(per_forecast)} различных дней-слотов.')
        
        dates_per_forecast = per_forecast['Дата'].unique()
        if debug:
            print(f'Всего требуется спрогнозировать: {len(dates_per_forecast)} уникальных дат.')

        
        # Шаг 5. Отбираем ТОЛЬКО исторические значения из исходного датафрейма
        history = data[data['Share'] != ''].reset_index(drop = True)
        history = history[columns_order]
        palomars_history = palomars_history[[
                    'Дата', 'Название программы', 'Время выхода', 
                    'Время окончания', 'Продолжительность', 'Жанр',
                    'dur_min', 'День недели', 'Тип дня', 'Share'
                        ]]
        
        # Шаг 6. Отбор ТОЛЬКО текущего года
        current_year_mask = (history['Дата'] >= f'{self.current_year}-01-01') & \
                            (history['Дата'] < self.start_date_forecast)
        current_year = history[current_year_mask].reset_index(drop = True)
        
        if len(current_year) != 0:
            
            # Шаг 7. Последняя фактическая дата из истории.
            last_fact_date = history['Дата'].max()
            if debug:
                print(f"Последняя фактическая дата: {last_fact_date.strftime('%Y-%m-%d')}.")

            # Шаг 7. Отбор ПОСЛЕДНИХ N НЕДЕЛЬ, исходя из максимальной фактической даты 
            date_n_weeks_ago = last_fact_date - pd.Timedelta(weeks = n_weeks_ago)
            if debug:
                print(f"Последние {n_weeks_ago} недели: {date_n_weeks_ago.strftime('%Y-%m-%d')} - {last_fact_date.strftime('%Y-%m-%d')}.")
            last_n_weeks = history[history['Дата'] > date_n_weeks_ago].reset_index(drop = True).copy()

            palomars_last_n_weeks =  palomars_history[palomars_history['Дата'] > date_n_weeks_ago].reset_index(drop = True).copy()
        
        else:
            print('Программы ' + Color.BLUE + f"'{program_name}'" + Color.END + ' ещё не было в текущем году.')

            # Отбираем после N недель из истории Palomars
            last_fact_date = palomars_history['Дата'].max()
            if debug:
                print(f"Последняя фактическая дата в истории Palomars: {last_fact_date.strftime('%Y-%m-%d')}.")
            date_n_weeks_ago = last_fact_date - pd.Timedelta(weeks = n_weeks_ago)

            if debug:
                print(f"Последние {n_weeks_ago} недели из Palomars: {date_n_weeks_ago.strftime('%Y-%m-%d')} - {last_fact_date.strftime('%Y-%m-%d')}.")

            last_n_weeks = pd.DataFrame()
            palomars_last_n_weeks =  palomars_history[palomars_history['Дата'] > date_n_weeks_ago].reset_index(drop = True).copy()

        program_forecast_package = {
            'target_dates': dates_per_forecast,         # даты, на которые нужен прогноз
            'target_data': per_forecast,                # данные для прогнозирования
            'history_last_n_weeks': last_n_weeks,       # история программы за N недель
            'palomars_history': palomars_last_n_weeks   # история Palomars за N недель
        }

        return program_forecast_package
    


    def forecast_big(
            self,
            date: str,
            program_name: str,
            search_values: dict,
            last_n_weeks: pd.DataFrame,
            palomars_last_n_weeks: pd.DataFrame,
            debug = False
        ):
        """
            Метод для прогнозирования программ с богатой историей.
            Args:
                - date: str: Дата, для которой будем строить прогноз
                - program_name: str: Название программы
                - search_values: dict: Значения для поиска
                - last_n_weeks: pd.DataFrame: датафрейм с последними N неделями для конкретной программы
                - palomars_last_n_weeks: pd.DataFrame: pd.DataFrame: датафрейм с последними N неделями Palomars (внезависимости от программы)
        """
        dur_min = search_values['dur_min']

        mean_duration = np.mean(list(last_n_weeks['dur_min']))

        if dur_min > mean_duration * 1.3:
            if debug:
                date_str = ''
                if type(date) != str:
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = date

                print(f"❗️ ВНИМАНИЕ! Для программы {program_name} {date_str} будет рассмотрен особый случай." + \
                      f'Длительность программы {dur_min} выше среднего {int(mean_duration)} за последние N недель. В список с комбинациями добавляем обязательный параметр "Продолжительность"!')
                
            combinations_list = PrimitiveModel.generate_field_combinations(
                            ['Время выхода', 'dur_min', 'День недели', 'Тип дня'], 
                            min_fields = 2,
                            must_include = 'dur_min',
                            exclude = None,
                            debug = False
                        )

        # Основной поиск по комбинациям
        share_mean = 0.0
        used_mask = None

        combinations_list = PrimitiveModel.generate_field_combinations(
                            fields = ['Время выхода', 'dur_min', 'День недели', 'Тип дня'], 
                            min_fields = 1,
                            must_include = None,
                            exclude = None,
                            debug = False
                        )

        # Проверка на длительность. Если длительность программы выше среднего, то будем включать "Продолжительность" в список комбинаций

        # ====================== ПОИСК ДАННЫХ В ПОСЛЕДНИХ N НЕДЕЛЯХ ======================
        for fields in combinations_list:
            # Шаг 1. Создаем маску для текущей комбинации
            condition = pd.Series(True, index = last_n_weeks.index)
            for field in fields:
                condition &= (last_n_weeks[field] == search_values[field])
            
            if not condition.any():
                if debug:
                    print(f"  ❌ Нет совпадений по полям: {fields}")
                continue

            mask = condition
            filtered_data = last_n_weeks[mask]

            # Шаг 2. Обработка в зависимости от размера выборки
            ########################## Выборка небольшая ##########################
            if len(filtered_data) <= self.SMALL_SAMPLE_SIZE:
                if len(filtered_data) == self.EXACT_MATCH_SIZE:
                    share_mean, used_mask = self.calculate_share(filtered_data, condition)
                    if debug:
                        print(f'Количество записей в выборке {len(filtered_data)}. Данные были найдены по полям {", ".join(fields)}.')
                    break

                if debug:
                    print('Выборка слишком мала -> поиск по длительности...')
                
                share_mean, used_mask = self.find_by_duration(last_n_weeks, dur_min, base_mask = condition)

                if np.isclose(share_mean, 0.0):
                    if debug:
                        print('🔄 Точное совпадение не найдено -> ищу с люфтом ±30%')
                    share_mean, used_mask = self.find_by_duration(last_n_weeks, dur_min, use_tolerance = True, base_mask = condition)
            
            ########################## Выборка большая ##########################
            else:
                # Фильтруем по минимальной длительности
                duration_mask = filtered_data['dur_min'] >= dur_min
                duration_filtered = filtered_data[duration_mask].reset_index(drop = True)
                
                # Комбинируем маски: исходная + фильтр по длительности
                if duration_mask.any():
                    combined_mask = condition.copy()
                    combined_mask[condition] = duration_mask.values
                    share_mean, used_mask = self.calculate_share(duration_filtered, combined_mask)
                else:
                    share_mean, used_mask = 0.0, None
                    
                if debug and share_mean > 0:
                    print(f'📈 Найдено {len(duration_filtered)} записей с длительностью ≥{dur_min}мин')
            
            break


        # ========== ЕСЛИ НЕ НАШЛИ В last_n_weeks, ОСУЩЕСТВЛЯЕМ ПОИСК ДАННЫХ В palomars_history ==========
        if np.isclose(share_mean, 0.0):
            if debug:
                print('🔄 Переключаюсь на историю Palomars...')

            share_mean, used_mask = self.find_by_duration(palomars_last_n_weeks, dur_min)
        
            if np.isclose(share_mean, 0.0):
                if debug:
                    print('⚠️ Точное совпадение в Palomars не найдено -> ищу с люфтом ±30%')
                share_mean, used_mask = self.find_by_duration(palomars_last_n_weeks, dur_min, use_tolerance = True)
            
            if np.isclose(share_mean, 0.0):
                print(Color.RED + \
                    f"❌ Программа {self.program_name} для даты {date} не найдена в истории." + \
                    Color.END)
        
        return share_mean, used_mask 
    


    def forecast_small(
            self,
            date: str,
            program_name: str,
            search_values: dict,
            last_n_weeks: pd.DataFrame,
            palomars_last_n_weeks: pd.DataFrame,
            debug = False
        ):
        """
            Метод для прогнозирования программ с маленькой историей.
            Параметры:
            ----------
                date: str: 
                    Прогнозируемая дата.
                combinations_list: list: 
                    Всевозможные комбинации признаков
                search_values: dict: 
                    Значения для поиска
                last_n_weeks: pd.DataFrame: 
                    Датафрейм с последними N неделями для конкретной программы
                palomars_last_n_weeks: pd.DataFrame: 
                    Датафрейм с последними N неделями Palomars (внезависимости от программы)
                debug: bool
                    Дебаггер
            Returns:
            ----------
                share_mean: float
                    Прогнозная доля
                used_mask:
                    Маска, которая использовалась для построения прогноза
        """
        dur_min = search_values['dur_min']

        share_mean = 0.0
        used_mask = None

        # Рассматриваем случай, когда истории по текущему году нет
        if len(last_n_weeks) == 0:
            if debug:
                print(f'Истории по текущему году нет. Использую историю Palomars.')
                print(f'Генерируем всевозможные комбинации при условии, что параметр "Продолжительность" встречается в каждой.')

            combinations_list = PrimitiveModel.generate_field_combinations(
                            fields = ['Время выхода', 'dur_min', 'День недели', 'Тип дня'], 
                            min_fields = 2,
                            must_include = 'dur_min',
                            exclude = None,
                            debug = debug
                        )
        
            for fields in combinations_list:
                # Создаем маску для комбинации полей
                condition = pd.Series(True, index = palomars_last_n_weeks.index)
                for field in fields:
                    if field in palomars_last_n_weeks.columns:
                        condition &= (palomars_last_n_weeks[field] == search_values.get(field))
                
                if not condition.any():
                    if debug:
                        print(f"  ❌ Нет совпадений по полям: {fields}")
                    continue
                
                filtered_data = palomars_last_n_weeks[condition]

                # Шаг 2. Обработка в зависимости от размера выборки
                ########################## Выборка небольшая ##########################
                if len(filtered_data) <= self.SMALL_SAMPLE_SIZE:
                    if len(filtered_data) == self.EXACT_MATCH_SIZE:
                        share_mean, used_mask = self.calculate_share(
                            filtered_data, condition, parent_df = palomars_last_n_weeks
                        )
                        if debug:
                            print(f'Количество записей в выборке {len(filtered_data)}. Данные были найдены по полям {", ".join(fields)}.')
                        break
                
                    if debug:
                        print('Выборка слишком мала -> поиск по длительности...')
                    
                    share_mean, used_mask = self.find_by_duration(palomars_last_n_weeks, dur_min, base_mask = condition)
                
                    if np.isclose(share_mean, 0.0):
                        if debug:
                            print('🔄 Точное совпадение не найдено -> ищу с люфтом ±30%')
                        share_mean, used_mask = share_mean, used_mask = self.find_by_duration(
                            palomars_last_n_weeks, dur_min, use_tolerance = True, base_mask = condition
                        )
                
                ########################## Выборка большая ##########################
                else:
                    # Фильтруем по минимальной длительности
                    duration_mask = filtered_data['dur_min'] >= dur_min
                    duration_filtered = filtered_data[duration_mask].reset_index(drop = True)

                    if debug:
                        print(f'Фильтрация по длительности ≥ {dur_min} мин: найдено {len(duration_filtered)} записей')
                    
                    if len(duration_filtered) > 0:

                        # Комбинируем маски правильно
                        combined_mask = condition.copy()
                        # Оставляем только те индексы, которые прошли фильтр длительности
                        valid_indices = filtered_data[duration_mask].index
                        combined_mask[~combined_mask.index.isin(valid_indices)] = False
                        
                        share_mean, used_mask = self.calculate_share(
                            duration_filtered, combined_mask, parent_df = palomars_last_n_weeks
                        )
                        
                        if debug and share_mean > 0:
                            print(f'📈 Найдено {len(duration_filtered)} записей с длительностью ≥{dur_min}мин')
                    else:
                        share_mean, used_mask = 0.0, None
                

                # Если нашли ненулевое значение, выходим из цикла
                if not np.isclose(share_mean, 0.0):
                    break
                
            if np.isclose(share_mean, 0.0):
                print(Color.RED + \
                    f"❌ Не удалось найти релевантные данные для прогноза программы {program_name} на дату {date}. " + \
                    f"Программа, возможно, новая, не встречалась ранее в истории Palomars за последние N недель." + \
                    Color.END)
        
        else:
            if len(last_n_weeks) >= 3:
                table = last_n_weeks.tail(3)
                share_mean = np.median(list(table['Share']))
                if debug:
                    print('Нахожу медиану последних 3х значений истории')
            
            elif len(last_n_weeks) == 1:
                share_mean = list(last_n_weeks['Share'])[0]
                if debug:
                    print('Всего 1 значение в истории. В качестве прогноза беру именно его.')
            
            elif len(last_n_weeks) == 2:
                share_mean = np.mean(list(last_n_weeks['Share']))
                if debug:
                    print('Два значения в истории. В качестве прогноза беру Среднее между ними.')
            
        return share_mean, used_mask
    


    def forecast_new(
            self,
            date: str,
            program_name: str,
            search_values: dict,
            last_n_weeks: pd.DataFrame,
            palomars_last_n_weeks: pd.DataFrame,
            debug = False
        ):
        """
            Метод для прогнозирования НОВЫХ программ.
            Параметры:
            ----------
                date: str: 
                    Прогнозируемая дата.
                combinations_list: list: 
                    Всевозможные комбинации признаков
                search_values: dict: 
                    Значения для поиска
                last_n_weeks: pd.DataFrame: 
                    Датафрейм с последними N неделями для конкретной программы
                palomars_last_n_weeks: pd.DataFrame: 
                    Датафрейм с последними N неделями Palomars (внезависимости от программы)
                debug: bool
                    Дебаггер
            Returns:
            ----------
                share_mean: float
                    Прогнозная доля
                used_mask:
                    Маска, которая использовалась для построения прогноза
        """
        dur_min = search_values['dur_min']

        share_mean = 0.0
        used_mask = None

        # Генерируем всевозможные комбинации. Требуем, чтобы "Продолжительность" фигурировала в каждом варианте.
        combinations_list = PrimitiveModel.generate_field_combinations(
                            fields = ['Время выхода', 'dur_min', 'День недели', 'Тип дня'], 
                            min_fields = 2,
                            must_include = 'dur_min',
                            exclude = None,
                            debug = debug
                        )
        
        found = False
        for fields in combinations_list:
            # Создаем маску для комбинации полей
            condition = pd.Series(True, index = palomars_last_n_weeks.index)
            for field in fields:
                if field in palomars_last_n_weeks.columns:
                    condition &= (palomars_last_n_weeks[field] == search_values.get(field))
            
            if not condition.any():
                if debug:
                    print(f"  ❌ Нет совпадений по полям: {fields}")
                continue
            
            filtered_data = palomars_last_n_weeks[condition]

            # ========== Обработка в зависимости от размера выборки ==========
            # Случай 2: Маленькая выборка (≤ SMALL_SAMPLE_SIZE)
            if len(filtered_data) <= self.SMALL_SAMPLE_SIZE:
                if len(filtered_data) == self.EXACT_MATCH_SIZE:
                    # Ровно 3 записи - используем точное среднее
                    share_mean, used_mask = self.calculate_share(
                        filtered_data, condition, parent_df = palomars_last_n_weeks
                    )
                    if debug:
                        print(f"  📊 Использую точное среднее по {len(filtered_data)} записям: {share_mean:.4f}")
                else:
                    # Меньше 3 записей - тоже используем среднее, но без маски
                    share_mean = PrimitiveModel.get_clean_mean(filtered_data['Share'])
                    used_mask = condition
                    if debug:
                        print(f"  📊 Выборка мала ({len(filtered_data)} записей), среднее: {share_mean:.4f}")
            
            # Случай 3: Большая выборка (> SMALL_SAMPLE_SIZE)
            else:
                # Фильтруем по минимальной длительности
                duration_mask = filtered_data['dur_min'] >= dur_min
                duration_filtered = filtered_data[duration_mask].reset_index(drop=True)
                
                if debug:
                    print(f"  📊 Фильтрация по длительности ≥{dur_min}мин: найдено {len(duration_filtered)} записей")
                
                if len(duration_filtered) > 0:
                    # Комбинируем маски
                    combined_mask = condition.copy()
                    valid_indices = filtered_data[duration_mask].index
                    combined_mask[~combined_mask.index.isin(valid_indices)] = False
                    
                    share_mean, used_mask = self.calculate_share(
                        duration_filtered, combined_mask, parent_df = palomars_last_n_weeks
                    )
                    
                    if debug and not np.isclose(share_mean, 0.0):
                        print(f"  📈 Рассчитано среднее: {share_mean:.4f}")
                else:
                    if debug:
                        print("  ⚠️ Нет записей с достаточной длительностью, пробую поиск с люфтом")
                    
                    # Пробуем найти с люфтом
                    share_mean, used_mask = self.find_by_duration(
                        palomars_last_n_weeks, dur_min, use_tolerance = True, base_mask = condition
                    )
            
            found = True
            break
        
        # Если ничего не нашли
        if not found:

            if debug:
                print("Пробую поиск по длительности")
            
            share_mean, used_mask = self.find_by_duration(
            palomars_last_n_weeks, dur_min, use_tolerance = False
        )
            
            if np.isclose(share_mean, 0.0):

                share_mean, used_mask = self.find_by_duration(
                palomars_last_n_weeks, dur_min, use_tolerance = True,
            )
                if debug:
                    print(f"  🔄 Ищу с люфтом ±30%")
                    print(f"  📈 Рассчитано среднее: {share_mean:.4f}")
        

        # Если ничего не нашли
        if np.isclose(share_mean, 0.0):
            date_str = date.strftime('%Y-%m-%d') if not isinstance(date, str) else date

            print(
                f"❌ Для новой программы" + Color.RED + f" '{program_name}'" + Color.END + \
                f"на дату {date_str} не найдено подходящих аналогов в истории Palomars." + \
                Color.END)
        
        return share_mean, used_mask
    
    

    def build_forecast(
            self,
            programs_dict: dict,
            all_holidays,
            work_saturdays,
            volume_flag: str,
            debug = False
        ):
        """
            Метод для построения прогноза.
            Параметры:
            ----------
                programs_dict: dict 
                    Словарь с историей программ. 
                    Ключ - название программы, значение - датафрейм с историей, а также датами, которые нужно спрогнозировать.
                all_holidays
                    Праздники РФ
                work_saturdays
                    Рабочие субботы в РФ
                combinations_list: list
                    Список из всевозможных комбинаций
                volume_flag: str
                    Флаг, который будет указывать, какой метод для прогнозирования необходимо применять.
        """
        if volume_flag not in ['small', 'big', 'new']:
            raise ValueError(f"Флаг '{volume_flag}' не существует. Выберите из списка: {['small', 'big', 'new']}")
        
        if volume_flag == 'big':
            print(f'==== Прогнозирую крупные программы ====')
        
        elif volume_flag == 'small':
            print(f'==== Прогнозирую мелкие программы ====')
        
        elif volume_flag == 'new':
            print(f'==== Прогнозирую новые программы ====')

        results_per_program = {}
        all_masks = {}

        last_n_weeks = pd.DataFrame()
        palomars_last_n_weeks = pd.DataFrame()

        for program_name in programs_dict.keys():
        
            df = programs_dict[program_name].reset_index(drop = True)

            # Подготовка данных для прогнозирования
            dict_analysis = self.prepare_forecast_inputs(df, program_name, all_holidays, work_saturdays, debug = debug)

            last_n_weeks = dict_analysis['history_last_n_weeks']
            palomars_last_n_weeks = dict_analysis['palomars_history']

            if debug:
                print(f'==== ПОСТРОЕНИЕ ПРОГНОЗА ДЛЯ ПРОГРАММЫ {program_name} ====')
            
            forecast_results = []
            date_masks = {}
            for date in dict_analysis['target_dates']:
                if debug:
                    print(f'------------- Прогнозирую дату {date} -------------')

                # Отбор даты, которую собираемся спрогнозировать
                per_forecast = dict_analysis['target_data']
                future_df = per_forecast[per_forecast['Дата'] == date].reset_index(drop = True)
                
                share_forecast = 0.0
                
                # Итерируемся по выделенным данным
                for i in range(len(future_df)):
                    
                    time_start = future_df.iloc[i]['Время выхода']
                    dur_min = future_df.iloc[i]['dur_min']
                    day_of_week = future_df.iloc[i]['День недели']
                    day_type = future_df.iloc[i]['Тип дня']
            
                    if debug:
                        print(f'Для {date} буду искать следующие кейсы в истории:')
                        print(f' - Время выхода:        {time_start}')
                        print(f' - Продолжительность:   {dur_min} мин')
                        print(f' - День недели:         {day_of_week}')
                        print(f' - Тип дня:             {day_type}')
                        print('\n')
                
                    # Шаг 2. Задаем значения для поиска
                    # Значения для поиска
                    search_values = {
                        'Время выхода': time_start,
                        'dur_min': dur_min,
                        'День недели': day_of_week,
                        'Тип дня': day_type
                    }

                    # Построение прогноза для крупной программы
                    if volume_flag == 'big':
                        share_forecast, mask = self.forecast_big(
                            date, program_name, search_values, 
                            last_n_weeks, palomars_last_n_weeks,
                            debug = debug
                        )

                    # Построение прогноза для мелкой программы
                    elif volume_flag == 'small':
                        share_forecast, mask = self.forecast_small(
                            date, program_name, search_values,
                            last_n_weeks, palomars_last_n_weeks,
                            debug = debug
                        )
                    
                    # Построение прогноза для НОВОЙ программы
                    elif volume_flag == 'new':
                        share_forecast, mask = self.forecast_new(
                            date, program_name, search_values,
                            last_n_weeks, palomars_last_n_weeks,
                            debug = debug)

                    date_masks[date.strftime('%Y-%m-%d')] = mask
            
                    # Запись прогнозного значения в ячейку
                    future_df.at[i, 'Share'] = share_forecast
            
                forecast_results.append(future_df)
                
                if debug:
                    print('-' * 50)
                    print('\n')
                    
                all_masks[program_name] = date_masks
                
            results_per_program[program_name] = pd.concat(forecast_results).reset_index(drop = True)
        return results_per_program