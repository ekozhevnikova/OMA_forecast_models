import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Callable
from datetime import timedelta, datetime, time
from dateutil.relativedelta import relativedelta
from difflib import SequenceMatcher


class TVShareCalculator:
    """
        Класс для расчета долей в конкретных слотах через вес слота и процент длительности программы в часе
    """
    def __init__(self, df):
        """
            Атрибуты класса
            Args:
                df: pd.DataFrame: Датафрейм с исходной долей
        """
        self.df = df
        self._validate_data()
    

    def _validate_data(self) -> None:
        """
            Проверка обязательных колонок в данных.
        """
        required_columns = ['Дата', 'Время выхода', 'Время окончания']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f'Отсутствуют обязательные колонки: {missing}')
    

    @staticmethod
    def calculate_slot_weights(
                    total_tv_audience: pd.DataFrame,
                    rating_col: str = 'TTVRtg000',
                    timeslot_col: str = 'TimeSlot',
                    date_col: str = 'Date'
                ) -> pd.DataFrame:
        """
            Рассчитывает веса слотов на основе данных о TotalTVAudience.
        
            Args:
                total_tv_audience: Датафрейм с аудиторными данными
                rating_col: Название колонки с рейтингом
                timeslot_col: Название колонки с временным слотом
                date_col: Название колонки с датой
                
            Returns:
                Датафрейм с весами слотов
        """
        df = total_tv_audience.copy()
        
        df['TimeSlot_dt'] = pd.to_datetime(df[timeslot_col])
        df['hour_start'] = df['TimeSlot_dt'].dt.hour
        
        # Группировка и расчет весов
        df['daily_total'] = df.groupby(date_col)[rating_col].transform('sum')
        df['Slot_weight'] = df[rating_col] / df['daily_total']
        
        # Замена бесконечно малых значений
        df['Slot_weight'] = df['Slot_weight'].fillna(0)
        
        return df[[date_col, timeslot_col, rating_col, 'Slot_weight', 'hour_start']].reset_index(drop = True)
    

    @staticmethod
    def get_hour(dt, param: str):
        """
            Метод для генерации часа. 
            Возможные опции: начало текущего часа, конец текущего часа, начало следующего часа, начало предыдущего часа
        """
        # Вариант 1: Начало текущего часа
        if param == 'start_of_current_hour':
            return dt.replace(minute = 0, second = 0, microsecond = 0)

        # Вариант 2: Конец текущего часа
        elif param == 'end_of_current_hour':
            return dt.replace(minute = 59, second = 59, microsecond = 0)
        
        # Вариант 3: Старт следующего часа
        elif param == 'start_next_hour':
            return dt.replace(minute = 0, second = 0, microsecond = 0) + timedelta(hours = 1)
        
        # Вариант 4: Старт прошлого часа
        elif param == 'start_previous_hour':
            return dt.replace(minute = 0, second = 0, microsecond = 0) + timedelta(hours = -1)
    

    def round_time(self, time_column: str, minutes = 1, method = 'round'):
        """
            Точное округление времени до минут без использования float.
            
            Args:

                time_series : pd.Series: Серия со временем в формате 'HH:MM:SS'
                minutes : int: Шаг округления в минутах (1, 5, 10, 15, 30, 60)
                method : str: Метод округления: 'round', 'floor', 'ceil'
            
            Returns:
                pd.Series: Округленное время
        """
        df = self.df.copy()
        time_series = df[time_column]

        def round_single_time(time_str, minutes_step, method_type):
            # Разбираем время
            if isinstance(time_str, str):
                h, m, s = map(int, time_str.split(':'))
            elif hasattr(time_str, 'hour'):  # Если это datetime.time
                h, m, s = time_str.hour, time_str.minute, time_str.second
            else:
                return time_str

            # Если имеем начало часа, например, 05:00:00, то возвращаем в исходном виде
            if m == 0.0 and s == 0.0:
                return f'{h:02d}:00:00'

            # Если время кривое
            else:
            
                # Общее количество секунд
                total_seconds = h * 3600 + m * 60 + s
                step_seconds = minutes_step * 60
                
                if method_type == 'floor':
                    # Округление вниз
                    rounded_seconds = (total_seconds // step_seconds) * step_seconds
                elif method_type == 'ceil':
                    # Округление вверх
                    if total_seconds % step_seconds == 0:
                        rounded_seconds = total_seconds
                    else:
                        rounded_seconds = ((total_seconds // step_seconds) + 1) * step_seconds
                else:  # 'round' - стандартное округление
                    # Количество секунд от начала интервала
                    remainder = total_seconds % step_seconds
                    
                    # Если остаток >= половины интервала, округляем вверх
                    if remainder >= step_seconds / 2:
                        rounded_seconds = ((total_seconds // step_seconds) + 1) * step_seconds
                    else:
                        rounded_seconds = (total_seconds // step_seconds) * step_seconds
                
                # Преобразуем обратно
                new_h = (rounded_seconds // 3600) % 24
                new_m = (rounded_seconds % 3600) // 60
                
                return f'{new_h:02d}:{new_m:02d}:00'
        
        # Применяем к каждой строке
        return time_series.apply(lambda x: round_single_time(x, minutes, method))
    

    def adjust_hour_start(self, end_col = 'Время окончания', start_col = 'Время выхода'):
        """
            Заменяет XX:00:00 на XX:59:59, только если это действительно 
            начало нового часа в расписании (т.е. следующее время выхода начинается с этого часа).
        """
        df_adj = self.df.copy()
        
        for i in range(len(df_adj) - 1):
            current_end = df_adj.loc[i, end_col]
            next_start = df_adj.loc[i + 1, start_col]
            
            # Если текущее окончание - начало часа И следующее время выхода начинается с этого же часа
            if current_end.endswith(':00:00') and next_start.startswith(current_end[:2]):
                hours = int(current_end.split(':')[0])
                
                if hours == 0:
                    new_time = '23:59:59'
                else:
                    new_hour = hours - 1
                    new_time = f"{new_hour:02d}:59:59"
                
                df_adj.loc[i, end_col] = new_time
        
        # Обрабатываем последнюю строку отдельно
        if df_adj.loc[df_adj.index[-1], end_col].endswith(':00:00'):
            hours = int(df_adj.loc[df_adj.index[-1], end_col].split(':')[0])
            if hours == 0:
                new_time = '23:59:59'
            else:
                new_hour = hours - 1
                new_time = f'{new_hour:02d}:59:59'
            df_adj.loc[df_adj.index[-1], end_col] = new_time
        
        return df_adj

    

    def calculate_hour_jump(self, start_col = 'Время выхода', end_col = 'Время окончания'):
        """
            Рассчитывает количество скачков через час для всех программ в DataFrame.
            
            Args:
                df: DataFrame с колонками времени
                start_col: название колонки с временем начала
                end_col: название колонки с временем окончания
            
            Returns:
                DataFrame с добавленной колонкой 'Скачки через час'
        """
        result_df = self.df.copy()

        jumps_list = []
        hours_list = []
        durations_list = []
        
        for idx, row in result_df.iterrows():
            try:
                # Парсим время
                start_time = datetime.strptime(str(row[start_col]), '%H:%M:%S')
                end_time = datetime.strptime(str(row[end_col]), '%H:%M:%S')
                
                # Корректируем время окончания при переходе через полночь
                if end_time <= start_time:
                    end_time += timedelta(days=1)
                
                # Рассчитываем количество скачков
                current_time = start_time
                hour_jumps = 0
                hours_covered = []
                
                while current_time < end_time:
                    current_hour = current_time.hour
                    hours_covered.append(current_hour)
                    
                    # Определяем начало следующего часа
                    next_hour_start = current_time.replace(minute = 0, second = 0, microsecond = 0) + timedelta(hours = 1)
                    
                    # Если следующий час не превышает время окончания, это скачок
                    if next_hour_start < end_time:
                        hour_jumps += 1
                    
                    # Переходим к следующему часу
                    current_time = next_hour_start
                
                # Длительность в минутах
                duration = (end_time - start_time).total_seconds() / 60.0
                
                jumps_list.append(hour_jumps)
                hours_list.append(hours_covered)
                durations_list.append(duration)
                
            except Exception as e:
                print(f'Ошибка в строке {idx}: {e}')
                jumps_list.append(0)
                hours_list.append([])
                durations_list.append(0)
        
        # Добавляем результаты в DataFrame
        result_df['Скачки через час'] = jumps_list
        result_df['Пройдено часов'] = hours_list
        result_df['Длительность (мин)'] = durations_list
        
        return result_df
    

    def calculate_weighted_share(self, auedience) -> pd.DataFrame:
        """
            Функция для расчета взвешенной доли для какого-то конкретного дня.
            Args:
                auedience: pd.DataFrame: ДатаФрейм с весами слотов, посчитанными через TotalTVAuedience для конкретного дня.
            Returns:
                res: pd.DataFrame: Датафрейм с новой рассчитанной долей для какого-то конкретного дня
                share_sum: суммарная доля для какого-то конкретного дня
        """
        # Определяем количество скачков через час в датафрейме
        result_df = self.calculate_hour_jump()

        df = result_df.copy()

        # Создаём столбец с новой долей
        df['Share_weighted'] = 0.0

        # Создаем словарь весов для быстрого доступа
        weight_dict = dict(zip(auedience['hour_start'], auedience['Slot_weight']))
        
        for i in range(len(df)):
            num_of_jumps = df.iloc[i]['Пройдено часов']
            program_start = df.iloc[i]['Время выхода']
            program_finish = df.iloc[i]['Время окончания']
            share = df.iloc[i]['Share']

            start_dt = datetime.strptime(program_start, '%H:%M:%S')
            end_dt = datetime.strptime(program_finish, '%H:%M:%S')

            # Обрабатываем переход через полночь
            if end_dt <= start_dt:
                end_dt += timedelta(days = 1)

            coeffs = []

            # Если скачка нет (программа в пределах одного часа)
            if len(num_of_jumps) == 1:
                
                # Определение часа старта для подбора веса слота
                hour = num_of_jumps[0]

                # Длительность в минутах
                duration_minutes = (end_dt - start_dt).total_seconds() / 60.0
                # % длительности программы в часе
                percent_duration = duration_minutes / 60.0

                # Получаем вес слота
                slot_weight = weight_dict.get(hour, 1.0)

                # Если скачка через час нет, считаем долю в слоте как Share * вес слота * % длительности программы в часе
                coeffs.append(percent_duration * slot_weight)
                
            # Если есть скачки (он необязательно должен быть 1)
            else:
                for k in range(len(num_of_jumps)):
                    # Первый скачок
                    if k == 0:
                        # Определяем конец первого часа
                        end_hour = datetime.strptime(f'{num_of_jumps[k]:02d}:59:59', '%H:%M:%S')
                        # Длительность в минутах
                        duration_minutes = (end_hour - start_dt).total_seconds() / 60.0
                        # % длительности программы в часе
                        percent_duration = duration_minutes / 60.0


                        # Получаем вес слота
                        slot_weight = weight_dict.get(num_of_jumps[0], 1.0)

                        coeffs.append(percent_duration * slot_weight)
                    
                    # Последний скачок
                    elif k == len(num_of_jumps) - 1:
                        # Определяем начало последнего часа
                        start_hour = datetime.strptime(f'{num_of_jumps[k]:02d}:00:00', '%H:%M:%S')

                        # Если start_hour меньше start_dt (переход через полночь), добавляем день
                        if start_hour < start_dt:
                            start_hour += timedelta(days = 1)

                        # Длительность в минутах
                        duration_minutes = (end_dt - start_hour).total_seconds() / 60.0
                        # % длительности программы в часе
                        percent_duration = duration_minutes / 60.0

                        # Получаем вес слота
                        slot_weight = weight_dict.get(num_of_jumps[-1], 1.0)

                        coeffs.append(percent_duration * slot_weight)

                    # Промежуточный скачок
                    else:
                        # Определяем начало часа
                        start_hour = datetime.strptime(f'{num_of_jumps[k]:02d}:00:00', '%H:%M:%S')

                        # Определяем конец часа
                        end_hour = datetime.strptime(f'{num_of_jumps[k]:02d}:59:59', '%H:%M:%S')
                        # Длительность в минутах
                        duration_minutes = (end_hour - start_hour).total_seconds() / 60.0
                        # % длительности программы в часе
                        percent_duration = duration_minutes / 60.0

                        # Получаем вес слота
                        slot_weight = weight_dict.get(num_of_jumps[k], 1.0)

                        coeffs.append(percent_duration * slot_weight)
                    
            if len(coeffs) != 0:
                coefficient = np.sum(coeffs)
                df.at[i, 'Share_weighted'] = share * coefficient

        res = df[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share', 'Share_weighted']]
        #res.rename(columns = {'Share_NEW': 'Share'}, inplace = True)
        # Расчёт суммарной доли по дню
        share_sum = np.sum(list(res['Share_weighted']))
        return res, share_sum



class TVScheduleProcessor:
    """
        Класс для подгона сетки Palomars под сетку VIMB из Сводного отчёта
    """

    def __init__(self):
        """
            Atributes:
                vimb_init: pd.DataFrame: исходная сетка ТВ-программ VIMB
                palomars_init: pd.DataFrame: исходная сетка ТВ-программ Mediascope
        """
        self.vimb_init = None
        self.palomars_init = None
        self.stats = {}
    

    def _adjust_end_time(self, df: pd.DataFrame, time_col: str = 'Время окончания') -> pd.DataFrame:
        """
            Корректировка времени окончания для обработки границ часов. Если время окончания, например, 05:00:00, то будет сделана замена на 04:59:59.
            Отдельно обрабатывается перескок через полночь.

            Args:
                df: датафрейм, в котором хотим произвести конвертацию времени.
                time_col: str: название колонки, в которой хотим сделать конвертацию. По умолчанию 'Время окончания'.

            Returns:
                Датафрейм df с конвертированными слотами Времени окончания программ.
        """
        def adjust_time(time_str: str) -> str:
            h, m, s = map(int, time_str.split(':'))
            
            if m == 0 and s == 0:

                # Если полночь
                if h == 0:
                    return '23:59:59'
                
                return f'{h-1:02d}:59:59'
            
            return time_str
            
        df = df.copy()
        df[time_col] = df[time_col].apply(adjust_time)
        return df
    
    
    def _add_original_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Добавление колонок с исходным временем, чтобы можно было сопоставить с исходным датафреймом VIMB.
            Args:
                df: датафрейм, в котором хотим добавить всопомгательные столбцы
            Returns:
                pd.DataFrame: датафрейм df с двумя дополнительными колонками с исходными временами слотов
        """
        result = df.copy()
        
        if 'Время выхода_исходное' not in result.columns:
            result['Время выхода_исходное'] = result['Время выхода']
            
        if 'Время окончания_исходное' not in result.columns:
            result['Время окончания_исходное'] = result['Время окончания']
            
        return result
    

    def _prepare_dataframe(self, df: pd.DataFrame, minutes: int, 
                          include_share: bool = True) -> pd.DataFrame:
        """
            Подготовка датафрейма. Времена слотов округляются до установленных минут. 
            Таким образом, датафрейм подготавливается для дальнейшего анализа.
        """
        data = df.copy()
        
        # Сохраняем исходные значения времени
        if 'Время выхода_исходное' not in data.columns:
            data['Время выхода_исходное'] = data['Время выхода']

        if 'Время окончания_исходное' not in data.columns:
            data['Время окончания_исходное'] = data['Время окончания']
        
        # Округление времени
        calculator = TVShareCalculator(data)
        data['Время выхода'] = calculator.round_time('Время выхода', minutes)
        data['Время окончания'] = calculator.round_time('Время окончания', minutes)
        
        # Выбор колонок
        columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания',
                  'Время выхода_исходное', 'Время окончания_исходное']
        
        if include_share and 'Share' in data.columns:
            columns.insert(4, 'Share')
            
        result = data[columns].copy()
        result['Дата'] = pd.to_datetime(result['Дата']).dt.strftime('%Y-%m-%d')
        
        return result


    def _remove_found_programs(self, source_df: pd.DataFrame, 
                              found_df: pd.DataFrame) -> pd.DataFrame:
        """
            Удаление найденных программ из исходных данных

            Args:
                source_df: датафрейм, из которого будем удалять найденные программы
                found_df: датафрейм, с найденными программами

            Returns:
                pd.DataFrame: очищенный датафрейм source_df от найденных программ в датафрейме found_df

        """
        if found_df.empty:
            return source_df
            
        merge_keys = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
        merged = source_df.merge(
            found_df[merge_keys],
            on = merge_keys,
            how = 'left',
            indicator = True
        )
        
        return merged.query('_merge == "left_only"').drop('_merge', axis = 1)
    

    @staticmethod
    def group_broadcasts_by_hours(data):
        """
            Объединяет последовательные трансляции одной программы по часам.
            Объединяет сегменты, которые находятся в одном часу и имеют последовательное время.
            
            Args:
                data (pd.DataFrame): Исходный DataFrame с временными сегментами
                
            Returns:
                pd.DataFrame: DataFrame с консолидированными сегментами по часам
        """
        df = data.copy()
    
        df['Время выхода'] = pd.to_datetime(df['Время выхода'])
        df['Время окончания'] = pd.to_datetime(df['Время окончания'])
    
        # Создаем колонку с часом начала для группировки
        df['Час_начала'] = df['Время выхода'].dt.floor('H')
        
        # Сортируем данные для корректной обработки последовательностей
        df = df.sort_values(['Название программы', 'Дата', 'Время выхода']).reset_index(drop=True)
        
        rows = []
        
        # Обрабатываем каждую программу отдельно
        for name in df['Название программы'].unique():
            program_df = df[df['Название программы'] == name].copy()
            
            # Сортируем по дате и времени
            program_df = program_df.sort_values(['Дата', 'Время выхода']).reset_index(drop=True)
            
            i = 0
            while i < len(program_df):
                current_row = program_df.iloc[i]
                current_date = current_row['Дата']
                current_hour = current_row['Час_начала']
                
                # Начинаем новую последовательность
                start_time = current_row['Время выхода']
                end_time = current_row['Время окончания']
                shares = [current_row['Share']]
                
                j = i + 1
                
                # Ищем последовательные трансляции в том же часу
                while j < len(program_df):
                    next_row = program_df.iloc[j]
                    
                    # Проверяем условия для объединения:
                    # 1. Та же дата
                    # 2. Тот же час начала
                    # 3. Время выхода следующей равно времени окончания предыдущей
                    if (next_row['Дата'] == current_date and 
                        next_row['Час_начала'] == current_hour and
                        next_row['Время выхода'] == end_time):
                        
                        end_time = next_row['Время окончания']
                        shares.append(next_row['Share'])
                        j += 1
                    else:
                        break
                
                # Создаем запись для этой последовательности
                rows.append({
                    'Дата': current_date,
                    'Название программы': name,
                    'Время выхода': start_time,
                    'Время окончания': end_time,
                    'Share': sum(shares),
                    'Количество_отрезков': len(shares),
                    'Час_группы': current_hour.time()  # Время часа (без даты)
                })
                
                i = j  # Переходим к следующей непроверенной строке
    
        # Создаем DataFrame
        result_df = pd.DataFrame(rows)
        
        # Если нужно, можно отсортировать результаты
        result_df = result_df.sort_values(['Дата', 'Название программы', 'Время выхода'])
    
        result_df['Время выхода'] = result_df['Время выхода'].dt.strftime('%H:%M:%S')
        result_df['Время окончания'] = result_df['Время окончания'].dt.strftime('%H:%M:%S')
        # Возвращаем только нужные колонки
        return result_df[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share']]


    @staticmethod
    def join_broadcasts(data):
        """
            Упрощенная версия для объединения трансляций в рамках одного дня.
            Учитывает эфирные сутки (05:00-04:59).
        """
        df = data.copy()
        
        # Проверяем количество дней
        if df['Дата'].nunique() > 1:
            print("Предупреждение: Рекомендуется обрабатывать по одному дню за раз")
        
        # Сортируем по времени с учетом эфирных суток
        def broadcast_time_key(time_str):
            """Ключ для сортировки по эфирным суткам"""
            h, m, s = map(int, time_str.split(':'))
            # Время с 05:00 считаем текущего дня, с 00:00-04:59 - следующего
            return (0 if h >= 5 else 1, h, m, s)
        
        # Добавляем ключ сортировки
        df['sort_key'] = df['Время выхода'].apply(broadcast_time_key)
        df = df.sort_values(['Название программы', 'sort_key']).reset_index(drop=True)
        df = df.drop('sort_key', axis=1)
        
        results = []
        i = 0
        
        while i < len(df):
            current = df.iloc[i]
            program = current['Название программы']
            start_time = current['Время выхода']
            end_time = current['Время окончания']
            shares = [current['Share']]
            
            j = i + 1
            
            # Пытаемся объединить с последующими трансляциями той же программы
            while j < len(df) and df.iloc[j]['Название программы'] == program:
                next_start = df.iloc[j]['Время выхода']
                next_end = df.iloc[j]['Время окончания']
                
                # Проверяем, идет ли следующая трансляция сразу после текущей
                # Учитываем переход через полночь
                if end_time == next_start:
                    # Прямая последовательность
                    end_time = next_end
                    shares.append(df.iloc[j]['Share'])
                    j += 1
                elif (end_time == "04:59:59" and next_start == "05:00:00"):
                    # Переход через границу эфирных суток
                    end_time = next_end
                    shares.append(df.iloc[j]['Share'])
                    j += 1
                else:
                    # Непоследовательная трансляция
                    break
            
            # Сохраняем результат
            results.append({
                'Дата': current['Дата'],
                'Название программы': program,
                'Время выхода': start_time,
                'Время окончания': end_time,
                'Share': sum(shares),
                'Количество_сегментов': len(shares)
            })
            
            i = j
        
        result_df = pd.DataFrame(results)
        
        # Сортируем итоговый результат
        result_df['sort_key'] = result_df['Время выхода'].apply(broadcast_time_key)
        result_df = result_df.sort_values('sort_key').drop('sort_key', axis = 1).reset_index(drop = True)
        
        return result_df[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share']]
    
    
    @staticmethod
    def find_time_overlaps(program1_start: str, program1_end: str, 
                        program2_start: str, program2_end: str, 
                        min_overlap_ratio: float = 0.8) -> Tuple[bool, float]:
        """
        Находит пересечения между временными интервалами.
        
        Args:
            program1_start: время начала первой программы
            program1_end: время окончания первой программы
            program2_start: время начала второй программы
            program2_end: время окончания второй программы
            min_overlap_ratio: минимальная доля пересечения относительно 
                            меньшего из интервалов (по умолчанию 50%)
        
        Returns:
            Tuple[bool, float]: (есть ли пересечение, доля пересечения)
        """
        # Конвертируем в минуты для удобства расчетов
        def time_to_minutes(t):
            h, m, s = map(int, t.split(':'))
            return h * 60 + m + s / 60
        
        start1 = time_to_minutes(program1_start)
        end1 = time_to_minutes(program1_end)
        start2 = time_to_minutes(program2_start)
        end2 = time_to_minutes(program2_end)
        
        # Проверяем корректность интервалов
        if end1 <= start1:
            # Корректируем если программа переходит через полночь
            if start1 >= 23 * 60:  # После 23:00
                end1 += 24 * 60
            else:
                return False, 0.0
        
        if end2 <= start2:
            if start2 >= 23 * 60:
                end2 += 24 * 60
            else:
                return False, 0.0
        
        # Проверяем пересечение
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            duration1 = end1 - start1
            duration2 = end2 - start2
            
            # Используем меньшую длительность для расчета доли
            min_duration = min(duration1, duration2)
            
            # Проверяем, достаточно ли большое пересечение
            if min_duration > 0 and overlap_duration / min_duration >= min_overlap_ratio:
                return True, overlap_duration / min_duration
        
        return False, 0.0
    

    def _process_schedule_step(self, palomars: pd.DataFrame, 
                              vimb: pd.DataFrame, minutes: int, 
                              step_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
            Обработка одного шага сопоставления с округлением времени
        """

        vimb = self._add_original_time_columns(vimb)

        # Подготовка данных
        pal_processed = self._prepare_dataframe(palomars, minutes, include_share = True)
        
        # Корректировка времени окончания
        pal_processed = self._adjust_end_time(pal_processed, 'Время окончания')
        vimb_processed = self._adjust_end_time(vimb, 'Время окончания')

        # Поиск совпадений
        merge_keys = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']

        matches = pd.merge(
            vimb_processed, pal_processed,
            on = merge_keys, how = 'inner'
        )[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share']]
        
        # Сохранение статистики
        self.stats[f'{step_name}_matches'] = len(matches)

        # Удаление найденных
        vimb_remaining = self._remove_found_programs(vimb_processed, matches).reset_index(drop = True)
        pal_remaining = self._remove_found_programs(pal_processed, matches).reset_index(drop = True)
        
        self.stats[f'{step_name}_vimb_remaining'] = len(vimb_remaining)
        self.stats[f'{step_name}_pal_remaining'] = len(pal_remaining)
        
        return matches, vimb_remaining, pal_remaining


    def _extract_core_info(self, df: pd.DataFrame, 
                          include_share: bool = True) -> pd.DataFrame:
        """
            Извлечение основной информации с переименованием колонок
        """
        if 'Время выхода_исходное' and 'Время окончания_исходное' in df.columns:
            if include_share and 'Share' in df.columns:
                columns = ['Дата', 'Название программы', 
                        'Время выхода_исходное', 'Время окончания_исходное', 'Share']
            else:
                columns = ['Дата', 'Название программы', 
                        'Время выхода_исходное', 'Время окончания_исходное']
            
            result = df[columns].copy()
            result.rename(columns = {
                'Время выхода_исходное': 'Время выхода',
                'Время окончания_исходное': 'Время окончания'
            }, inplace = True)
            
            return result

        else:
            return df
    

    def broadcasts_overlaping(self, df, vimb):
        """
            Метод для сопоставления программ методом перекрытия двух длительностей.
        """
        vimb_ = self._adjust_end_time(vimb, 'Время окончания')

        print(df)
        print(vimb_)

        df['overlap'] = ''
        df['Время выхода VIMB'] = ''
        df['Время окончания VIMB'] = ''

        # Через расчет процента перекрытия понимаем, нужная это программа или нет
        for i in range(len(df)):
            name = df.iloc[i]['Название программы']
            start_palomars = df.iloc[i]['Время выхода']
            h_start_P, m, s = map(int, start_palomars.split(':'))

            end_palomars = df.iloc[i]['Время окончания']
            h_end_P, m, s = map(int, end_palomars.split(':'))

            in_vimb = vimb_[vimb_['Название программы'] == name].reset_index(drop = True)

            #print(name, len(in_vimb))
            
            if len(in_vimb) != 0:
                
                for j in range(len(in_vimb)):
                    start_vimb = in_vimb.iloc[j]['Время выхода']
                    h_start_V, m, s = map(int, start_vimb.split(':'))

                    end_vimb = in_vimb.iloc[j]['Время окончания']
                    h_end_V, m, s = map(int, end_vimb.split(':'))
                    
                    print(name, h_start_P, h_end_P, h_start_V, h_end_V)
                    if h_start_P == h_start_V and h_end_P == h_end_V or \
                      (h_start_V - h_start_P) == 1 and (h_end_V - h_end_P) == 1 or \
                      h_start_P == h_start_V and (h_end_V - h_end_P) == 1 or \
                      (h_start_V - h_start_P) == 1 and h_end_P == h_end_V:
                        
                        #print(name, h_start_P, h_end_P, h_start_V, h_end_V)
                        overlap = TVScheduleProcessor.find_time_overlaps(start_palomars, end_palomars, start_vimb, end_vimb)
                        print(df)
                        df.at[i, 'overlap'] = overlap[0]
                        df.at[i, 'Время выхода VIMB'] = start_vimb
                        df.at[i, 'Время окончания VIMB'] = end_vimb

                    else:
                        continue
            else:
                continue
        
        with_overlap = df[df['overlap'] == True].reset_index(drop = True)
        res_overlap = with_overlap[['Дата', 'Название программы', 'Время выхода VIMB', 'Время окончания VIMB', 'Share']]
        res_overlap.rename(columns = {
                            'Время выхода VIMB': 'Время выхода', 
                            'Время окончания VIMB': 'Время окончания'
                                }, 
                            inplace = True)
        return res_overlap

    

    def process_schedules(self, vimb: pd.DataFrame, palomars: pd.DataFrame, 
                         verbose: bool = True) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
        """
            Основной метод обработки телепрограмм

            Args:
                vimb: pd.DataFrame: датафрейм с сеткой ВИМБ
                palomars: pd.DataFrame: датафрейм с сеткой Mediascope
                verbose: bool:
            Returns:
                Tuple: (результирующий DataFrame, статистика, найденные программы, оставшиеся программы)
        """
        share_sum_init = np.sum(list(palomars['Share']))
        # Инициализация
        self.vimb_init = vimb.copy()
        self.palomars_init = palomars.copy()
        self.stats = {}

        #print(palomars)
        
        if verbose:
            print(f'Исходные данные: VIMB = {len(self.vimb_init)}, Palomars = {len(self.palomars_init)}')
        
        
        # Список для хранения найденных программ
        all_matches = []

        ############################ ШАГ 1: Обработка мелких программ, которые легко сджойнить ############################

        vimb_10min = self._prepare_dataframe(vimb, 10, False)
        pal_10min = self._prepare_dataframe(palomars, 10, True)

        matches, vimb_remaining, pal_remaining = self._process_schedule_step(
                pal_10min, vimb_10min, 10, '10_minutes'
            )

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(matches, 'Время окончания'))

        # Подготовка данных для следующего шага
        current_pal = self._extract_core_info(pal_remaining, True)
        current_vimb = self._extract_core_info(vimb_remaining, False)

        found_programs_after_1 = pd.concat(all_matches).reset_index(drop = True)

        # Удаление найденных длинных программ
        vimb_after_1 = self._remove_found_programs(
                                self._adjust_end_time(vimb, 'Время окончания'), 
                                found_programs_after_1).reset_index(drop = True)
        vimb_clean_after_1 = self._extract_core_info(vimb_after_1, False)

        #print(vimb_clean_after_1)


        #print('#' * 120)
        #print('ПОСЛЕ ШАГА 1')
        #print('Найденные короткие программы')
        #print(matches)
        #print('Остаточный Palomars')
        #print(current_pal)
        #print('#' * 120)

        ############################ ШАГ 2: Обработка длинных программ (округление 10 минут) ############################

        vimb_10min = self._prepare_dataframe(vimb_clean_after_1, 10, False)
        pal_10min = self._prepare_dataframe(palomars, 10, True)

        # Поиск длинных программ, которые шли более, чем 1 час
        long_programs = TVScheduleProcessor.join_broadcasts(pal_10min)

        long_matches = pd.merge(
            self._adjust_end_time(vimb_10min, 'Время окончания'), 
            self._adjust_end_time(long_programs, 'Время окончания'), 
            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
            how = 'inner'
        )

        # Переименование колонок
        long_matches_clean = self._extract_core_info(long_matches, True)

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(long_matches_clean, 'Время окончания'))

        # Удаление найденных длинных программ
        pal_remaining = self._remove_found_programs(pal_10min, long_matches).reset_index(drop = True)
        vimb_remaining = self._remove_found_programs(vimb_10min, long_matches).reset_index(drop = True)

        # Переименование колонок
        pal_clean = self._extract_core_info(pal_remaining, True)
        vimb_clean = self._extract_core_info(vimb_remaining, False)


        found_programs_after_2 = pd.concat(all_matches).reset_index(drop = True)

        # Удаление найденных длинных программ
        vimb_after_2 = self._remove_found_programs(
                                self._adjust_end_time(vimb, 'Время окончания'), 
                                found_programs_after_2).reset_index(drop = True)
        vimb_clean_after_2 = self._extract_core_info(vimb_after_2, False)


        #print('#' * 120)
        #print('ПОСЛЕ ШАГА 2')
        #print('Найденные длинные программы')
        #print(long_matches_clean)
        #print('Остаточный Palomars')
        #print(pal_clean)
        #print('#' * 120)

        ############################ ШАГ 3: Обработка длинных программ с overlaping ############################
        #print('VIMB для анализа overlapping')
        #print(vimb_clean)
        #print('#' * 120)

        # Заменяем значения начиная со второго
        for i in range(1, len(palomars)):
            palomars.loc[i, 'Время выхода'] = palomars.loc[i - 1, 'Время окончания']

        # Отбираем программы, которые шли больше 1 часа и соединяем
        res = TVScheduleProcessor.join_broadcasts(palomars)

        res_overlap = self.broadcasts_overlaping(res, vimb_clean_after_2)

        res_overlap['Дата'] = pd.to_datetime(res_overlap['Дата'])
        res_overlap['Дата'] = res_overlap['Дата'].dt.strftime('%Y-%m-%d')

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(res_overlap, 'Время окончания'))

        pal_clean['Дата'] = pd.to_datetime(pal_clean['Дата'])
        pal_clean['Дата'] = pal_clean['Дата'].dt.strftime('%Y-%m-%d')

        # Удаление найденных длинных программ
        pal_residual = self._remove_found_programs(pal_clean, res_overlap).reset_index(drop = True)
        
        #print('#' * 120)
        #print('ПОСЛЕ ШАГА 3')
        #print('Найденные программы в результате overlaping')
        #print(res_overlap)
        #print('Остаточный Palomars')
        #print(pal_cleaned)
        #print('#' * 120)

        found_programs_after_3 = pd.concat(all_matches).reset_index(drop = True)
        #print('Найденные программы на текущий момент')
        #print(current_found_programs)

        # Удаление найденных длинных программ
        vimb_after_3 = self._remove_found_programs(
                                self._adjust_end_time(vimb, 'Время окончания'), 
                                found_programs_after_3).reset_index(drop = True)
        vimb_clean_after_3 = self._extract_core_info(vimb_after_3, False)

        ############################ ШАГ 4: Обработка длинных программ (округление 10 минут) ############################     

        # Работаем с исходным датафреймом palomars
        pal_10min = self._prepare_dataframe(palomars, 10, True)
        #print(pal_10min)

        # Поиск длинных программ, которые шли в течение 1 часа
        hour_programs = TVScheduleProcessor.group_broadcasts_by_hours(pal_10min)

        #print(hour_programs)
        #print(vimb_clean_after_3)

        hour_overlap = self.broadcasts_overlaping(self._adjust_end_time(hour_programs), vimb_clean_after_3)


        hour_matches = pd.merge(
            self._adjust_end_time(vimb_clean_after_3), hour_overlap,
            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
            how = 'inner'
        )
        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(hour_matches, 'Время окончания'))

        #print('Программы, которые идут час')
        #print(hour_overlap)

        #print('MATCHING')
        #print('*' * 120)
        #print(matches)
        #print('*' * 120)
        #print(long_matches_clean)
        #print('*' * 120)
        #print(res_overlap)
        #print('*' * 120)
        #print(hour_overlap)

        found_programs = pd.concat(all_matches).reset_index(drop = True)
        #print('Найденные программы')
        #print(found_programs)

        merged = pd.merge(
            self._adjust_end_time(vimb), found_programs,
            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
            how = 'inner'
        )

        if len(merged) != len(vimb):
            print('Нужен дополнительный поиск!')
        
        if share_sum_init != np.sum(list(found_programs['Share'])):
            share_sum_curr = np.sum(list(found_programs['Share']))
            print(f'Обнаружено несовпадение суммарной доли по дню! Целевой показатель {np.round(share_sum_init, 2)}, а по итогу вышло {np.round(share_sum_curr, 2)}.')
        
        return merged









#        # Работаем с исходным датафреймом palomars
#        pal_5min = self._prepare_dataframe(palomars, 5, True)
#
#        # Поиск длинных программ, которые шли в течение 1 часа
#        hour_programs = TVScheduleProcessor.group_broadcasts_by_hours(pal_5min)
#        hour_programs = self._adjust_end_time(hour_programs)
#
#        hour_matches = pd.merge(
#            self._adjust_end_time(vimb_cleaned), hour_programs,
#            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
#            how = 'inner'
#        )
#        # Добавляем в список найденных программ
#        all_matches.append(hour_matches)
#
#        # Удаление найденных длинных программ
#        pal_remaining_new = self._remove_found_programs(pal_5min, hour_matches).reset_index(drop = True)
#
#        # Переименование колонок
#        pal_clean_new = self._extract_core_info(pal_remaining_new, True)
#
#        vimb_remaining_new = self._remove_found_programs(vimb_cleaned, hour_matches).reset_index(drop = True)
#
#
#
#
#
#        ############################ Шаг 3: Поиск по недлительным программам. Поочереди округляем слоты Palomars ############################
#        # Для анализа берем исходный датафрейм Palomars и последний преобразованный датафрейм VIMB
#        # Последовательная обработка с разным округлением
#        rounding_steps = [
#            (1, "1_minute", palomars, vimb_remaining_new),
#            (5, "5_minutes", None, None),
#            (10, "10_minutes", None, None)
#        ]
#
#        current_pal = palomars
#        current_vimb = vimb_remaining_new
#        for minutes, step_name, pal_input, vimb_input in rounding_steps:
#            if verbose:
#                print(f'\nШаг {step_name}: Округление {minutes} минут')
#            
#            # Используем переданные данные или результаты предыдущего шага
#            pal_to_process = pal_input if pal_input is not None else current_pal
#            vimb_to_process = vimb_input if vimb_input is not None else current_vimb
#
#            # Ищем совпадения
#            matches, vimb_remaining, pal_remaining = self._process_schedule_step(
#                pal_to_process, vimb_to_process, minutes, step_name
#            )
#
#            # Добавляем в список найденных программ
#            all_matches.append(matches)
#
#            # Подготовка данных для следующего шага
#            current_pal = self._extract_core_info(pal_remaining, True)
#            current_vimb = self._extract_core_info(vimb_remaining, False)
#
#            if minutes < 10:  # Для следующих шагов добавляем исходные метки
#                current_vimb = self._add_original_time_columns(current_vimb)
#        
#        ############################ ДОПОЛНИТЕЛЬНЫЙ ШАГ: Повторное округление до 10 минут ############################
#        if verbose:
#            print('\nДополнительный шаг: Повторное округление до 10 минут')
#        
#        # Снова округляем текущие данные до 10 минут
#        vimb_round_to_10 = self._prepare_dataframe(current_vimb, 10, False)
#        pal_round_to_10 = self._prepare_dataframe(current_pal, 10, True)
#
#        # Ищем совпадения
#        additional_matches = pd.merge(
#            vimb_round_to_10,
#            pal_round_to_10,
#            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
#            how = 'inner'
#        )[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share']]
#        
#        self.stats['additional_10min_matches'] = len(additional_matches)
#
#        # Добавляем в список найденных программ
#        all_matches.append(additional_matches)
#
#        # Удаление найденных программ
#        vimb_deleted = self._remove_found_programs(vimb_round_to_10, additional_matches).reset_index(drop = True)
#        pal_deleted = self._remove_found_programs(pal_round_to_10, additional_matches).reset_index(drop = True)
#
#        pal_deleted['Дата'] = pd.to_datetime(pal_deleted['Дата'], errors = 'coerce')
#        pal_deleted['Дата'] = pal_deleted['Дата'].dt.strftime('%Y-%m-%d')
#
#        # Подготовка данных для следующего шага
#        final_pal_remaining = self._extract_core_info(pal_deleted, True)
#        final_vimb_remaining = self._extract_core_info(vimb_deleted, False)
#
#        ############################ ПОСЛЕДНИЙ ШАГ: Поиск через перекрытия ############################
#        found_programs = pd.concat(all_matches).reset_index(drop = True)
#
#        found_programs_unique = found_programs.drop_duplicates(
#            subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
#        ).reset_index(drop = True)
#
#        print(final_vimb_remaining)
#        print('#' * 120)
#        print(found_programs_unique)
#
#        vimb_last = self._remove_found_programs(self._adjust_end_time(final_vimb_remaining, 'Время окончания'), found_programs_unique).reset_index(drop = True)
#
#        print('#' * 120)
#        print(vimb_last)
#        print('#' * 120)
#
#        if len(found_programs_unique) != len(self.vimb_init):
#            #final_vimb_remaining = self._adjust_end_time(final_vimb_remaining, 'Время окончания')
#            final_pal_remaining = self._adjust_end_time(final_pal_remaining, 'Время окончания')
#            matches_df, unmatched_vimb_df, palomar_df = TVScheduleProcessor.match_programs_with_tolerance(vimb_last, final_pal_remaining, consolidate_palomars = True)
#
#            print(matches_df)
#
#            matched = matches_df[['Дата', 'Название_VIMB', 'Время_начала_VIMB', 'Время_окончания_VIMB', 'Share']]
#            matched.rename(columns = {
#                                'Название_VIMB': 'Название программы',
#                                'Время_начала_VIMB': 'Время выхода',
#                                'Время_окончания_VIMB': 'Время окончания'
#                            }, 
#                           inplace = True)
#            # Добавляем в список найденных программ
#            all_matches.append(matched)
#
#            found_programs = pd.concat(all_matches).reset_index(drop = True)
#
#            found_programs_unique = found_programs.drop_duplicates(
#                subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
#            ).reset_index(drop = True)
#
#        ############################ Объединение всех найденных программ ############################
#        #found_programs = pd.concat(all_matches).reset_index(drop = True)
#
#        # Финальное сопоставление с исходными данными
#        vimb_adjusted = self._adjust_end_time(self.vimb_init, 'Время окончания')
#        found_adjusted = self._adjust_end_time(found_programs_unique, 'Время окончания')
#        
#        result = pd.merge(
#            vimb_adjusted,
#            found_adjusted,
#            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
#            how = 'left'
#        )
#        
#        # Расчёт статистики
#        final_stats = self._calculate_final_stats(final_vimb_remaining, found_programs_unique)
#
#        # Вывод статистики
#        if verbose:
#            self.print_statistics(final_stats)
#            print(f'\nОсталось ненайденных программ VIMB: {len(final_vimb_remaining)}')
#            print(f'Осталось ненайденных программ Palomar: {len(final_pal_remaining)}')
#        
#        #found_programs_unique = found_programs.drop_duplicates(
#        #    subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
#        #).reset_index(drop = True)
#
#        if len(found_programs_unique) != len(self.vimb_init):
#            print(f'ВНИМАНИЕ: Размер результата ({len(found_programs_unique)}) не совпадает с исходным VIMB ({len(self.vimb_init)})')
#        
#        if share_sum_init != np.sum(list(found_programs_unique['Share'])):
#            share_sum_curr = np.sum(list(found_programs_unique['Share']))
#            print(f'Обнаружено несовпадение суммарной доли по дню! Целевой показатель {np.round(share_sum_init, 2)}, а по итогу вышло {np.round(share_sum_curr, 2)}.')
#        
#        return result, final_stats, found_programs_unique, final_vimb_remaining, final_pal_remaining


    def _calculate_final_stats(self, vimb_final: pd.DataFrame, 
                             found_programs: pd.DataFrame) -> Dict:
        """
            Расчёт финальной статистики по найденным и ненайденным программам
        """
        # Общее количество программ в исходных данных
        total_vimb_programs = len(self.vimb_init)
        
        # Количество найденных программ (уникальных записей)
        found_count = len(found_programs.drop_duplicates(
            subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
        ))
        
        # Количество ненайденных программ
        not_found_count = total_vimb_programs - found_count
        
        # Процент найденных
        found_percentage = (found_count / total_vimb_programs * 100) if total_vimb_programs > 0 else 0
        
        # Статистика по шагам поиска
        step_stats = {}
        for key, value in self.stats.items():
            if '_matches' in key:
                step_name = key.replace('_matches', '')
                step_stats[step_name] = value
        
        return {
            'total_vimb_programs': total_vimb_programs,
            'found_programs': found_count,
            'not_found_programs': not_found_count,
            'found_percentage': round(found_percentage, 2),
            'step_by_step_matches': step_stats,
            'remaining_vimb_after_all_steps': len(vimb_final)
        }
    
    def print_statistics(self, stats: Dict) -> None:
        """
            Вывод статистики в удобном формате
        """
        print("\n" + "="*60)
        print("СТАТИСТИКА ОБРАБОТКИ ТЕЛЕПРОГРАММ")
        print("="*60)
        print(f"Всего программ в VIMB: {stats['total_vimb_programs']}")
        print(f"Найдено программ: {stats['found_programs']}")
        print(f"Не найдено программ: {stats['not_found_programs']}")
        print(f"Процент найденных: {stats['found_percentage']}%")
        print("-"*60)
        print("Пошаговая статистика:")
        for step, count in stats['step_by_step_matches'].items():
            print(f"  {step}: {count} совпадений")
        print("-"*60)
        print(f"Осталось ненайденных после всех шагов: {stats['remaining_vimb_after_all_steps']}")
        print("="*60 + "\n")