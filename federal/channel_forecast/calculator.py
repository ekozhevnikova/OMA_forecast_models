import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Callable
from datetime import timedelta, datetime, time
from dateutil.relativedelta import relativedelta
from difflib import SequenceMatcher
from OMA_tools.federal.channel_forecast.core.simple_models import *
import traceback
traceback.print_exc()


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
                    rating_col: str = 'Auedience',
                    timeslot_col: str = 'TimeSlot',
                    date_col: str = 'Дата'
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
                    end_time += timedelta(days = 1)
                
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

        res = df[['Канал', 'Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share', 'Share_weighted', 'Жанр', 'День недели']]
        #res.rename(columns = {'Share_NEW': 'Share'}, inplace = True)
        # Расчёт суммарной доли по дню
        share_sum = np.sum(list(res['Share_weighted']))
        return res, share_sum



class TVScheduleProcessor:
    """
        Класс для подгона сетки Palomars под сетку VIMB из Сводного отчёта для одного дня
    """

    def __init__(self, vimb_init, palomars_init):
        """
            Атрибуты:
                vimb_init: pd.DataFrame: исходная сетка ТВ-программ VIMB
                palomars_init: pd.DataFrame: исходная сетка ТВ-программ Mediascope
        """
        self.vimb_init = vimb_init
        self.palomars_init = palomars_init
    

    def convert_data_column(self, df):
        """
            Всопомгательный метод для конвертации столбца с названием 'Дата'
        """
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Дата'] = df["Дата"].dt.strftime("%Y-%m-%d")
        return df
    

    def adjust_end_time(
                    self, 
                    df: pd.DataFrame, 
                    time_col: str = 'Время окончания'
                ) -> pd.DataFrame:
        """
            Корректировка времени окончания для обработки границ часов. Если время окончания, например, 05:00:00, то будет сделана замена на 04:59:59.
            Отдельно обрабатывается перескок через полночь.

            Args:
                df: датафрейм, в котором хотим произвести конвертацию времени.
                time_col: str: название колонки, в которой хотим сделать конвертацию. По умолчанию 'Время окончания'.

            Returns:
                Датафрейм df с конвертированными слотами Времени окончания программ.
        """

        def adjust_time__(time_str: str) -> str:
            h, m, s = map(int, time_str.split(':'))

            if m == 0 and s == 0:

                # Если полночь
                if h == 0:
                    return '23:59:59'

                return f'{(h - 1):02d}:59:59'

            return time_str

        df = df.copy()
        df[time_col] = df[time_col].apply(adjust_time__)
        return df
    


    def join_broadcasts(self, data, include_share: bool = True):
        """
            Метод реализует объединение трансляций в рамках одного дня.
            Учитывает эфирные сутки (05:00-04:59).
            Объединяет смежные сегменты одной программы.
        """
        if data.empty:
            columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
            if include_share:
                columns.append('Share')
            return pd.DataFrame(columns = columns)
        
        df = data.copy()
        
        df = self.adjust_end_time(df)
        
        # Функция для сортировки по эфирным суткам
        def broadcast_time_key(time_str):
            h, m, s = map(int, time_str.split(':'))
            return (0 if h >= 5 else 1, h, m, s)
        
        # Преобразование времени в минуты для удобного сравнения
        def time_to_minutes(time_str):
            """
                Преобразует время в формате HH:MM:SS в минуты с начала эфирных суток
            """
            h, m, s = map(int, time_str.split(':'))
            # Для времени до 05:00 добавляем 24 часа
            total_minutes = h * 60 + m + s / 60

            if h < 5:  # Время с 00:00 до 04:59
                total_minutes += 24 * 60  # Добавляем сутки
            return total_minutes
        
        # Подготовка данных
        df['sort_key'] = df['Время выхода'].apply(broadcast_time_key)
        df = df.sort_values(['Дата', 'sort_key'])
        df = df.drop('sort_key', axis = 1)
        
        # Добавляем колонку с временем в минутах для сравнения
        df['start_minutes'] = df['Время выхода'].apply(time_to_minutes)
        df['end_minutes'] = df['Время окончания'].apply(time_to_minutes)
        
        results = []
        
        # Обработка каждой программы
        for program_name in df['Название программы'].unique():
            program_mask = df['Название программы'] == program_name
            program_data = df[program_mask].copy()
            
            if program_data.empty:
                continue
            
            # Сортировка программы по времени (уже отсортирована)
            program_data = program_data.sort_values('start_minutes')
            
            # Объединение сегментов
            current_group = {
                'Дата': program_data.iloc[0]['Дата'],
                'Название программы': program_name,
                'Время выхода': program_data.iloc[0]['Время выхода'],
                'Время окончания': program_data.iloc[0]['Время окончания'],
                'start_minutes': program_data.iloc[0]['start_minutes'],
                'end_minutes': program_data.iloc[0]['end_minutes'],
            }
            
            if include_share:
                current_group['shares'] = [program_data.iloc[0]['Share']]
            
            # Обработка остальных записей программы
            for i in range(1, len(program_data)):
                current_row = program_data.iloc[i]
                next_start_minutes = current_row['start_minutes']
                next_end_minutes = current_row['end_minutes']
                
                # Проверка на смежность сегментов с учетом разницы в 1 минуту
                time_gap = next_start_minutes - current_group['end_minutes']
                
                # Ключевое изменение: Не объединяем через границу эфирных суток
                # (кроме специального случая 04:59:59 → 05:00:00)
                prev_end_time = current_group['Время окончания']
                next_start_time = current_row['Время выхода']
                
                # Проверяем, не пересекаем ли мы границу эфирных суток
                # (следующий сегмент начинается в новых эфирных сутках, а текущий заканчивается в старых)
                crosses_broadcast_day = (
                    prev_end_time >= '00:00:00' and prev_end_time <= '04:59:59' and
                    next_start_time >= '05:00:00'
                )
                
                # Условия объединения:
                # 1. Нет разрыва (время окончания = время начала следующей) И не пересекаем границу
                # 2. Разрыв в пределах 1 минуты И не пересекаем границу
                # 3. Специальный случай: 04:59:59 → 05:00:00 (это допускается)
                is_adjacent = (
                    (time_gap == 0 and not crosses_broadcast_day) or  # Нет разрыва и не пересекаем границу
                    (0 < time_gap <= 1 and not crosses_broadcast_day) or  # Разрыв не более 1 минуты и не пересекаем границу
                    (prev_end_time == '04:59:59' and next_start_time == '05:00:00')  # Допустимый переход через границу
                )
                
                if is_adjacent:
                    # Объединяем с текущей группой
                    current_group['Время окончания'] = current_row['Время окончания']
                    current_group['end_minutes'] = next_end_minutes
                    if include_share:
                        current_group['shares'].append(current_row['Share'])
                else:
                    # Сохраняем текущую группу и начинаем новую
                    result_entry = {
                        'Дата': current_group['Дата'],
                        'Название программы': current_group['Название программы'],
                        'Время выхода': current_group['Время выхода'],
                        'Время окончания': current_group['Время окончания'],
                    }
                    
                    if include_share:
                        result_entry['Share'] = sum(current_group['shares'])
                        result_entry['Количество_сегментов'] = len(current_group['shares'])
                    
                    results.append(result_entry)
                    
                    # Новая группа
                    current_group = {
                        'Дата': current_row['Дата'],
                        'Название программы': program_name,
                        'Время выхода': current_row['Время выхода'],
                        'Время окончания': current_row['Время окончания'],
                        'start_minutes': next_start_minutes,
                        'end_minutes': next_end_minutes,
                    }
                    
                    if include_share:
                        current_group['shares'] = [current_row['Share']]
            
            # Сохраняем последнюю группу
            result_entry = {
                'Дата': current_group['Дата'],
                'Название программы': current_group['Название программы'],
                'Время выхода': current_group['Время выхода'],
                'Время окончания': current_group['Время окончания']
            }
            
            if include_share:
                result_entry['Share'] = sum(current_group['shares'])
                result_entry['Количество_сегментов'] = len(current_group['shares'])
            
            results.append(result_entry)
        
        # Формирование итогового DataFrame
        if not results:
            columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
            if include_share:
                columns.append('Share')
            return pd.DataFrame(columns = columns)
        
        result_df = pd.DataFrame(results)
        
        # Сортировка результатов
        result_df['sort_key'] = result_df['Время выхода'].apply(broadcast_time_key)
        result_df = result_df.sort_values('sort_key').drop('sort_key', axis = 1)
        
        # Выбор нужных колонок
        columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
        if include_share:
            columns.append('Share')
        
        return result_df[columns].reset_index(drop = True)
    

    def find_matches(self, minutes: int):
        """
            Метод для поиска совпадающих программ. 
        """

        vimb_joined = self.join_broadcasts(self.vimb_init, include_share = False)
        plmrs_joined = self.join_broadcasts(self.palomars_init)

        # Исходная суммарная доля по дню
        share_init = plmrs_joined['Share'].sum()

        # Округление времени
        calculator = TVShareCalculator(plmrs_joined)
        plmrs_joined['Время выхода'] = calculator.round_time('Время выхода', minutes)
        plmrs_joined['Время окончания'] = calculator.round_time('Время окончания', minutes)

        plmrs_joined = self.convert_data_column(plmrs_joined)

        plmrs_joined.rename(columns = 
                            {
                                'Время выхода': 'Время выхода _plmrs', 
                                'Время окончания': 'Время окончания _plmrs'
                             }, 
                             inplace = True)
        
        vimb_joined.rename(columns = 
                           {
                               'Время выхода': 'Время выхода _vimb', 
                               'Время окончания': 'Время окончания _vimb'
                            }, 
                            inplace = True)

        plmrs_joined['Время выхода'] = plmrs_joined['Время выхода _plmrs']
        vimb_joined['Время выхода'] = vimb_joined['Время выхода _vimb']

        # Попытка совместить две сетки между собой
        merged = pd.merge(plmrs_joined, vimb_joined, on = ['Дата', 'Название программы', 'Время выхода'], how = 'left')

        # Создаем финальные столбцы
        merged['Время выхода'] = merged['Время выхода _vimb'].combine_first(merged['Время выхода _plmrs'])
        merged['Время окончания'] = merged['Время окончания _vimb'].combine_first(merged['Время окончания _plmrs'])

        # Определяем строки, где vimb-данные отсутствуют
        vimb_missing = merged['Время выхода _vimb'].isna()

        # Для каждой строки с отсутствующими vimb-данными, корректируем время окончания предыдущей строки
        for idx in merged[vimb_missing].index:
            if idx > 0:  # Если это не первая строка
                # Берем время выхода текущей строки (из plmrs)
                current_start = merged.loc[idx, 'Время выхода _plmrs']
                # Корректируем время окончания предыдущей строки
                merged.loc[idx - 1, 'Время окончания'] = current_start

        # Удаляем ненужные столбцы
        result = merged.drop(columns = ['Время выхода _plmrs', 'Время окончания _plmrs', 
                            'Время выхода _vimb', 'Время окончания _vimb'])

        result = result[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share']].reset_index(drop = True)

        share_end = result['Share'].sum()

        if share_init != share_end:
            print('Нужен дополнительный анализ!')
        
        return result


class MonthlyShareAnalyzer:
    """
    (!!!) ВАЖНО (!!!) Работает для какого-то конкретного месяца и года!
        Класс по расчёту месячной доли через TTV и количество дней в месяце. 

        Итоговая доля считается как TVR_summ (будни) + TVR_summ (выходные) / TVR_summ (за месяц), где

        TVR_summ (будни) = Средняя доля будние * TTV (будни) * Кол-во будних дней в месяце
        TVR_summ (выходные) = Средняя доля выходные * TTV (выходные) * Кол-во выходные дней в месяце
        TVR_summ (за месяц) = TTV (за месяц) * Кол-во дней в месяце
    """
    def __init__(self, year: int, month: str, bca: str, forecast_df: pd.DataFrame, holidays_file: str):
        self.year = year
        self.month = month
        self.bca = bca
        self.forecast_df = forecast_df
        self.holidays_file = holidays_file


    def calculate_ttv(
        self,
        ttv_filepath: str, 
        need_columns: list
        ):
        """
            Метод для чтения файла с TTV.
        """
        # 1. Чтение файла
        try:
            ttv = pd.read_excel(
                    ttv_filepath, 
                    sheet_name = 'Для шаблонов',
                    skiprows = 3,
                    nrows = 39
                )
            
        except FileNotFoundError:
            raise FileNotFoundError(f'Файл не найден: {ttv_filepath}')

        # 2. Отбор нужных колонок
        df_ttv = ttv[[self.year] + need_columns].copy()
        
        df_ttv.rename(columns = {'Unnamed: 2': 'Месяц'}, inplace = True)
        
        # 3. Отбор анализируемого месяца
        month_clean = str(self.month).strip()
        month_mask = df_ttv['Месяц'].str.lower() == month_clean.lower()
        
        ttv_filtered = df_ttv[month_mask].reset_index(drop = True)

        # 4. Поиск TTV за весь месяц, за будние дни, за выходные дни
        try:
            full_ttv = ttv_filtered.iloc[0]      # за весь месяц
            weekday_ttv = ttv_filtered.iloc[1]   # за будние дни
            weekend_ttv = ttv_filtered.iloc[2]   # за выходные дни
        except IndexError as e:
            raise ValueError(f"Недостаточно строк данных для месяца '{self.month}': {str(e)}")


        # 5. Запись в словарь интересующих данных
        result = {
            'итого': full_ttv[self.bca],
            'будни': weekday_ttv[self.bca],
            'выходные': weekend_ttv[self.bca]
        }
        return result
    

    def calculate_monthly_share(self, ttv_dict: dict):
        """
            Метод для расчета месячной доли 
        """
        # 1. Считаем праздники России
        work_saturdays, all_holidays = PrimitiveModel.build_russian_holidays(self.holidays_file)

        # 2. Определение типа дня
        self.df['Тип дня'] = self.df['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))

        # 3. Отбор будних и выходных дней
        weekdays = self.df[self.df['Тип дня'] == 'Будни'].reset_index(drop = True)
        weekends = self.df[self.df['Тип дня'] == 'Выходной'].reset_index(drop = True)

        # 4. Подсчет средней доли будних и выходных дней
        mean_share_weekdays = np.mean(list(weekdays['Share']))
        mean_share_weekend = np.mean(list(weekends['Share']))

        # 5. Подсчет количества будних, выходных и количества дней в месяце
        count_weekends = (self.df['Тип дня'] == 'Выходной').sum()
        count_weekdays = (self.df['Тип дня'] == 'Будни').sum()
        n_days = count_weekends + count_weekdays

        TVR_summ = {
            'итого': ttv_dict['итого'] * n_days,
            'будни': mean_share_weekdays * ttv_dict['будни'] * count_weekdays,
            'выходные': mean_share_weekend * ttv_dict['выходные'] * count_weekends
            }

        share_per_month = (TVR_summ['будни'] + TVR_summ['выходные']) / TVR_summ['итого']
        return share_per_month
    

    def fit_calculate(self, ttv_filepath: str, target_column: str, need_columns: list):
        """
            Пайплайн для расчета
        """
        unique_dates = self.forecast_df['Дата'].unique()

        shares = {}
        for date in unique_dates:
            df = self.forecast_df[self.forecast_df['Дата'] == date].reset_index(drop = True)
            
            # Расчёт суммарной доли по дню
            shares[date] = np.sum(list(df[target_column]))
        
        self.df = pd.DataFrame({
            'Дата': pd.to_datetime(list(shares.keys())),
            'Share': list(shares.values())
        })

        # Сортировка по дате (если нужно)
        self.df = self.df.sort_values('Дата').reset_index(drop = True)

        ttv = self.calculate_ttv(ttv_filepath, need_columns)
        share_per_month = self.calculate_monthly_share(ttv)
        return share_per_month

