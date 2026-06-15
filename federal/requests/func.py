import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from calendar import monthrange

from OMA_tools.io_data.colors import *


import warnings
warnings.filterwarnings('ignore')

T = 24 * 60 # эфирные сутки в секундах 1440 мин
NUMBER_OF_DAYS = 90

class GeneralClass:
    def __init__(self):
        pass

        self.MONTHS = {
            1: 'Январь 2026', 2: 'Февраль 2026', 3: 'Март 2026'
        }


    def sort_time(self, t):
        """
        Преобразует время в числовое значение для сортировки.
        Значения до 05:00 получают +24 часа, чтобы оказаться после 23:59
        """
        if isinstance(t, str):
            # Если строка
            hour, minute, second = map(int, t.split(':'))
        else:
            # Если datetime.time
            hour, minute, second = t.hour, t.minute, t.second
        
        if hour < 5:
            total_seconds = (hour + 24) * 3600 + minute * 60 + second
        else:
            total_seconds = hour * 3600 + minute * 60 + second
        
        return total_seconds
    
    
    
    def calculate_end_time(self, row, col_with_duration, start_col):
        """Расчет времени окончания"""
        time_val = row[start_col]
        duration_sec = float(row[col_with_duration])
        
        # Если это datetime.time, конвертируем в секунды напрямую
        if hasattr(time_val, 'hour'):
            start_seconds = time_val.hour * 3600 + time_val.minute * 60 + time_val.second
        else:
            # Если строка
            h, m, s = map(int, time_val.split(':'))
            start_seconds = h * 3600 + m * 60 + s
        
        # Считаем конечное время в секундах
        end_seconds = start_seconds + duration_sec
        
        # Конвертируем обратно в HH:MM:SS
        hours = int((end_seconds // 3600) % 24)
        minutes = int((end_seconds % 3600) // 60)
        seconds = int(end_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    
    def calculate_duration(self, df, time_start_column_name: str):
        # Создаем столбцы для округленного времени и корректировки
        rounded_times = []
        adjustments = []
    
        for time_val in df[time_start_column_name]:
            seconds = time_val.second
            minute = time_val.minute
            hour = time_val.hour
    
            if seconds < 30:
                # Округляем вниз
                new_time = time_val.replace(second=0)
                adjustment = -seconds
            else:
                # Округляем вверх
                new_minute = minute + 1
                new_hour = hour
                if new_minute == 60:
                    new_minute = 0
                    new_hour = hour + 1
                    # Если час стал 24, значит это полночь следующего дня
                    if new_hour == 24:
                        new_hour = 0
                new_time = time_val.replace(hour=new_hour, minute=new_minute, second=0)
                adjustment = 60 - seconds
    
            rounded_times.append(new_time)
            adjustments.append(adjustment)
        
        return rounded_times, adjustments
    
    
    
    def split_overnight_broadcasts(self, df):
        """
        Разбивает записи, пересекающие границу 05:00:00, пересчитывая длительность
        """
        result_rows = []
        boundary = pd.to_datetime('05:00:00').time()
        
        for idx, row in df.iterrows():
            start_time = pd.to_datetime(row['Время выхода'], format='%H:%M:%S').time()
            end_time = pd.to_datetime(row['Время окончания'], format='%H:%M:%S').time()
            
            # Проверяем, пересекает ли запись границу дня
            # Условие: начало до 05:00 И конец после 05:00 (либо на следующий день)
            if start_time < boundary and (end_time > boundary or end_time < start_time):
                # Создаем первую часть (до 05:00:00)
                part1 = row.copy()
                part1['Время окончания'] = '05:00:00'
                
                # Создаем вторую часть (после 05:00:00)
                part2 = row.copy()
                part2['Время выхода'] = '05:00:00'
                part2['ResearchDate'] = (pd.to_datetime(row['ResearchDate']) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                result_rows.append(part1)
                result_rows.append(part2)
            else:
                result_rows.append(row)
        
        return pd.DataFrame(result_rows)
    
    
    def calculate_program_duration(self, df: pd.DataFrame):
        """
            Метод для расчёта длительностей программ
    
            Параметры:
            ----------
            df: pd.DataFrame
                Таблица в которой будем производить преобразования
    
            Returns:
            ----------
            df: pd.DataFrame
                Та же таблица, но с дополнительными новыми столбцами
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
    
        df['Продолжительность'] = (
            pd.to_datetime(df['Время окончания_dt']) - pd.to_datetime(df['Время выхода_dt'])
        ).dt.total_seconds()
    
        # Форматирование
        df['Продолжительность'] = df['Продолжительность'].apply(
            lambda x: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
        )
        return df
    
    
    def format_sessions(self, data: pd.DataFrame, flag = False):
        data_analysis = pd.DataFrame()
        
        # Форматирование столбца с Датой
        data['ResearchDate'] = pd.to_datetime(data['ResearchDate'])
        data_sorted = data.sort_values(by = ['ResearchDate'], ascending = [True]).reset_index(drop = True)
        data_sorted['ResearchDate'] = data_sorted['ResearchDate'].dt.strftime('%Y-%m-%d')
    
        # Преобразуем Start в datetime и извлекаем только время
        data_sorted['Время выхода'] = pd.to_datetime(data_sorted['Start']).dt.strftime('%H:%M:%S')
        
        if flag:
            data_analysis = data_sorted[['SubjectID', 'ResearchDate', 'Время выхода', 'Duration', 'ChannelID', 'Media']]
            return data_analysis
        else:
            # Оставляем только нужные колонки
            data_analysis = data_sorted[['SubjectID', 'ResearchDate', 'Время выхода', 'Duration', 'ChannelID']]
            return data_analysis
    
    
    def preprocess_data(self, channels_dict, data_analysis, respondents, flag = False):
        """
            Функция, которая подготавливает данные для анализа
        """
        data_analysis_new = pd.DataFrame()
        
        if flag:
            data_analysis_new = pd.merge(data_analysis, channels_dict, on = ['ChannelID', 'Media'], how='inner')
            data_analysis_new = data_analysis_new.drop(['ContentProviderName'], axis = 1)
            data_analysis_new.rename(columns = {'MediaProductName': 'Channel'}, inplace = True)
            data_analysis_new = data_analysis_new[['ResearchDate', 'SubjectID', 'Channel', 'ChannelID', 'Время выхода', 'Duration', 'Media']]
        else:
            data_analysis_new = pd.merge(data_analysis, channels_dict, on = ['ChannelID'], how='inner')
            data_analysis_new = data_analysis_new.drop(['ContentProviderName'], axis = 1)
            data_analysis_new.rename(columns = {'MediaProductName': 'Channel'}, inplace = True)
            data_analysis_new = data_analysis_new[['ResearchDate', 'SubjectID', 'Channel', 'ChannelID', 'Время выхода', 'Duration']]
        
        merged_df = pd.merge(data_analysis_new, respondents, on = ['SubjectID', 'ResearchDate'], how = 'inner')
        
        if flag:
            merged_df = merged_df[['ResearchDate', 'SubjectID', 'Channel', 'Время выхода', 'Duration', 'weight_new', 'Media']]
        
        else:
            merged_df = merged_df[['ResearchDate', 'SubjectID', 'Channel', 'Время выхода', 'Duration', 'weight_new']]
        merged_df['Channel'] = merged_df['Channel'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))
        return merged_df
    
    
    def group_by_channels(self, data: pd.DataFrame, ordered_names: list):
        """
            Функция для разбивки по каналам или радио-станциям
        """
        channels_data = data[data['Channel'].isin(ordered_names)]
        groups = {name: group for name, group in channels_data.groupby('Channel')}
        groups_dict = {channel: df.reset_index(drop = True) 
                           for channel, df in channels_data.groupby('Channel')}
        return groups_dict
    
    
    def calculate_universe(self, respondents_df: pd.DataFrame):
        """
            Функция для расчета статистики Universe, основываясь на весах отобранных респондентов
        """
        Universe_dict = {}
    
        for date in respondents_df['ResearchDate'].unique():
            df = respondents_df[respondents_df['ResearchDate'] == date]
            universe = df['weight_new'].sum()
            Universe_dict[date] = universe
    
        UNIVERSE = pd.DataFrame(list(Universe_dict.items()), columns = ['ResearchDate', 'Universe'])
        return UNIVERSE
    

    def prepare_data_according_to_mediascope(self, df):
        """
            Подготавливает данные в соответствии с правилами Mediascope
        """
        # 1. Считаем время окончания просмотра
        df['Время окончания'] = df.apply(
            lambda row: GeneralClass().calculate_end_time(row, 'Duration', 'Время выхода'), 
            axis = 1
        )
    
        # 2. Считаем скачки через сутки
        table_converted = GeneralClass().split_overnight_broadcasts(df)
    
        # 3. Настраиваем верную сортировку от 05:00:00 до 05:00:00
        res = []
        for date in table_converted['ResearchDate'].unique():
            t = table_converted[table_converted['ResearchDate'] == date].copy()
            t['ResearchDate'] = pd.to_datetime(t['ResearchDate'])
            # Создаем колонку для сортировки
            t['sort_key'] = t['Время выхода'].apply(GeneralClass().sort_time)
            # Сортируем по sort_key
            final = t.sort_values('sort_key').reset_index(drop = True)
            # Удаляем вспомогательную колонку
            final = final.drop('sort_key', axis = 1)
            final['ResearchDate'] = final['ResearchDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
            res.append(final)
    
        table_sorted = pd.concat(res, ignore_index = True)
    
        # 4. Преобразуем время в datetime.time
        table_sorted['Время выхода'] = pd.to_datetime(table_sorted['Время выхода'], format = '%H:%M:%S').dt.time
        table_sorted['Время окончания'] = pd.to_datetime(table_sorted['Время окончания'], format = '%H:%M:%S').dt.time
    
        # 5. Округляем время выхода и время окончания программ
        rounded_times_start, adjustments_start = GeneralClass().calculate_duration(table_sorted, 'Время выхода')
        rounded_times_stop, adjustments_stop = GeneralClass().calculate_duration(table_sorted, 'Время окончания')
    
        # 6. Добавляем столбцы с ОКРУГЛЕННЫМ ВРЕМЕНЕМ ВЫХОДА И ОКОНЧАНИЯ
        table_sorted['Время выхода_округленное'] = rounded_times_start
        table_sorted['Время окончания_округленное'] = rounded_times_stop
    
        # 7. Удаляем ненужные колонки и переименовываем другие
        table_new = table_sorted.drop(['Время выхода', 'Время окончания', 'Duration'], axis = 1)
    
        table_new.rename(columns = {
            'Время выхода_округленное': 'Время выхода',
            'Время окончания_округленное': 'Время окончания',
        }, inplace = True)
    
        # 8. Переводим время в формат строки
        table_new['Время выхода'] = table_new['Время выхода'].apply(lambda x: x.strftime('%H:%M:%S'))
        table_new['Время окончания'] = table_new['Время окончания'].apply(lambda x: x.strftime('%H:%M:%S'))
    
        # 9. Считаем новую длительность
        table_df = GeneralClass().calculate_program_duration(table_new)
    
        table_df = table_df.drop(['Время выхода_dt', 'Время окончания_dt'], axis = 1)
        table_df['Duration'] = pd.to_timedelta(table_df['Продолжительность']).dt.total_seconds()
    
        # 10. Отфильтровавыем события > 1 минуты
        table_df = table_df[table_df['Duration'] >= 30].reset_index(drop = True)
    
        # 11. Переводим длительность в минуты
        table_df['dur_min'] = table_df['Duration'] / 60
    
        table_df = table_df.drop(['Продолжительность'], axis = 1)
    
        table_df = table_df[(table_df['ResearchDate'] >= '2026-01-01') & (table_df['ResearchDate'] <= '2026-03-31')]
        
        return table_df
    

    def form_dataframe_from_dict(self, channel_dict: dict):
        """
            Формирует датафрейм из словаря 
        """
        # Создаём список для хранения DataFrame с метками каналов
        dfs_with_channel = []

        for channel_name, df in channel_dict.items():
            df_copy = df.copy()
            df_copy['Channel'] = channel_name  # добавляем колонку с названием канала
            dfs_with_channel.append(df_copy)

        # Объединяем все
        df_all = pd.concat(dfs_with_channel, ignore_index=True)
        return df_all

    
    
    def calculate_monthly_values(self, data: pd.DataFrame, ordered_names):
        """
            Функция для расчета средних месячных показателей по КАНАЛАМ 
        """
        # Создаем ВСПОМОГАТЕЛЬНЫЙ столбец с номером месяца
        data['num_month'] = pd.to_datetime(data['ResearchDate']).dt.month
    
        result_data = []
        for num_month, name in self.MONTHS.items():
            df_month = data[data['num_month'] == num_month].copy()
    
            # Количество дней в месяце
            days_in_month = pd.to_datetime(df_month['ResearchDate']).dt.days_in_month.iloc[0]
            print(f'Количество дней в {name} равно {days_in_month}.')
    
            # Для каждого канала (столбца) считаем среднее
            for column in df_month.columns:
                if column not in ['ResearchDate', 'num_month']:
                    avg = df_month[column].mean()
                    # Добавляем словарь в список
                    result_data.append({
                        'Канал': column,
                        'Месяц': name,
                        'Значение': avg
                    })
    
        # Создаем DataFrame из списка словарей
        temp_df = pd.DataFrame(result_data)
    
        # Преобразуем в нужный формат (каналы в строки, месяцы в колонки)
        data_monthly = temp_df.pivot(index='Канал', columns='Месяц', values='Значение').reset_index()
    
        # Убираем имя у колонок
        data_monthly.columns.name = None
        data_monthly = data_monthly[['Канал', 'Январь 2026', 'Февраль 2026', 'Март 2026']]
        
        # ===== СОРТИРОВКА КАНАЛОВ В УКАЗАННОМ ПОРЯДКЕ =====
        # Создаем категориальный тип с нужным порядком
        data_monthly['Канал'] = pd.Categorical(
            data_monthly['Канал'], 
            categories=ordered_names, 
            ordered=True
        )
        
        # Сортируем DataFrame по категории
        data_monthly = data_monthly.sort_values('Канал')
        
        # Опционально: сбрасываем индекс
        data_monthly = data_monthly.reset_index(drop=True)
        return data_monthly
    
    
    def calculate_average_values(self, data: pd.DataFrame, statistic: str):
        """
            Функция для расчета средних месячных показателей Audience и AverageTime
        """
        # Создаем ВСПОМОГАТЕЛЬНЫЙ столбец с номером месяца
        total_days = len(data)
        print(f'Общее количество дней равно {total_days}.')
    
        result_data = []
        # Для каждого канала (столбца) считаем среднее
        for column in data.columns:
            if column not in ['ResearchDate', 'num_month']:
                avg = data[column].mean()
                # Добавляем словарь в список
                result_data.append({
                    'Канал': column,
                    'Статистика': statistic,
                    'Значение': avg
                })# $$\textbf{Расчёт показателей показателей за весь период}$$
    
        # Создаем DataFrame из списка словарей
        temp_df = pd.DataFrame(result_data)
    
        # Преобразуем в нужный формат (каналы в строки, месяцы в колонки)
        data_average = temp_df.pivot(index='Канал', columns='Статистика', values='Значение').reset_index()
    
        # Убираем имя у колонок
        data_average.columns.name = None
        return data_average
    
    
    def calculate_average_values_per_period(self, data_dict: dict, ordered_names: list):
        """
            Функция для расчета средних показателей Audience и AverageTime за весь период
        """
        results = []
        for statistic, data in data_dict.items():
            res_df = self.calculate_average_values(data, statistic)
            results.append(res_df)
    
        general_result = results[0].merge(results[1], on = 'Канал', how = 'outer')
        general_result = general_result.merge(results[2], on = 'Канал', how = 'outer')
    
        # Приводим порядок каналов в соответствии со списком FEDERAL_CHANNELS
        general_result['Канал'] = pd.Categorical(general_result['Канал'], categories = ordered_names, ordered = True)
        general_result = general_result.sort_values('Канал').reset_index(drop=True)
        return general_result



class Audience_TimeAverage:
    """
        Класс для расчёта статистики Audience
    """
    def __init__(self, list_names, universe_df):
        self.list_names = list_names
        self.universe_df = universe_df
    
    
    def generate_result(self, data_groups: dict):
        """
            Функция для расчета показателя Audience для группы (ТВ или Радио)
        """
        by_groups = {}
    
        for channel, df in data_groups.items():
    
            #print(f'Делаю расчет для канала {channel}')
            # Форматируем таблицу
            #table_df = self.calculate_audience_timeaverage(df)
    
            # Векторизованный расчет вместо цикла по дням
            grouped = df.groupby('ResearchDate').apply(
                lambda g: pd.Series({
                    'Audience': np.sum(g['weight_new'] * g['dur_min']) / 1440,  # T = 1440 минут
                    'Average Time': np.sum(g['weight_new'] * g['dur_min']) / 
                                   self.universe_df.loc[self.universe_df['ResearchDate'].isin([g.name]), 'Universe'].values[0]
                })
            ).reset_index()
            
            grouped.columns = ['ResearchDate', f'Audience {channel}', f'Average Time {channel}']
            by_groups[channel] = grouped
        
        # Получаем все уникальные даты из всех каналов
        all_dates = set()
        for df in by_groups.values():
            all_dates.update(df['ResearchDate'].values)
        all_dates = sorted(list(all_dates))
        
        # СОЗДАЕМ РЕЗУЛЬТАТЫ ДЛЯ ВСЕХ КАНАЛОВ ИЗ list_names
        audience_dict = {'ResearchDate': all_dates}
        avgtime_dict = {'ResearchDate': all_dates}
        
        for channel in self.list_names:  # Итерируемся по СПИСКУ, а не по by_groups
            if channel in by_groups:
                # Канал есть в данных - берем значения
                df_channel = by_groups[channel]
                # Создаем словарь для маппинга дат
                audience_map = dict(zip(df_channel['ResearchDate'], df_channel[f'Audience {channel}']))
                avgtime_map = dict(zip(df_channel['ResearchDate'], df_channel[f'Average Time {channel}']))
                
                # Заполняем значения для всех дат (отсутствующие = 0)
                audience_dict[channel] = [audience_map.get(date, 0) for date in all_dates]
                avgtime_dict[channel] = [avgtime_map.get(date, 0) for date in all_dates]
            else:
                # Канала нет в данных - весь столбец из нулей
                audience_dict[channel] = [0] * len(all_dates)
                avgtime_dict[channel] = [0] * len(all_dates)
                print(f'Внимание: Канал "{channel}" отсутствует в данных, заполнен нулями')
        
        # Создаем DataFrame
        audience_df = pd.DataFrame(audience_dict)
        avgtime_df = pd.DataFrame(avgtime_dict)
        
        # Убеждаемся, что порядок колонок правильный
        audience_df = audience_df[['ResearchDate'] + self.list_names]
        avgtime_df = avgtime_df[['ResearchDate'] + self.list_names]
        
        result = {
            'Audience': audience_df,
            'AvgTime': avgtime_df
        }
        return result


class Reach:
    """
        Класс для расчёта статистики Reach
    """
    def __init__(self, names_list: list):
        self.names_list = names_list


    def calculate_reach_per_channel(self, df: pd.DataFrame, channel_name: str, flag = False):
        """
            Расчет Reach для какого-то канала
            Если flag = True, то считаем совокупные показатели ТВ+Радио
        """
        # Оставляем события > 1 мин
        df = df[df['Duration'] >= 30].reset_index(drop = True)
    
        Reach_by_dates = {}
        for target_date in df['ResearchDate'].unique():
            d = df[df['ResearchDate'] == target_date]

            reach = None
            if flag:
                reach = d.drop_duplicates(subset = ['SubjectID', 'Media'])['weight_new'].sum()
            else: 
                reach = d.drop_duplicates(subset = ['SubjectID'])['weight_new'].sum()
    
            Reach_by_dates[target_date] = reach
    
        reach_df = pd.DataFrame(list(Reach_by_dates.items()), columns = ['ResearchDate', f'Reach {channel_name}'])
        return reach_df
    
    
    def generate_result_Reach(self, data_group: dict, flag = False):
        """
            Функция для расчета статистики Reach для каждого канала/радио станции
            Если flag = True, то считаем совокупные показатели ТВ+Радио
        """
        reach_by_channels = {}
    
        for channel, df in data_group.items():
    
            #print(f'Делаю расчет для канала {channel}')
            reach_by_channels[channel] = self.calculate_reach_per_channel(df, channel, flag)
    
        # Создаем DataFrame через concat с заполнением NaN нулями
        reach_dfs = []
        for channel in self.names_list:
            if channel in reach_by_channels:
                df_channel = reach_by_channels[channel]
                if 'ResearchDate' in df_channel.columns:
                    df_channel = df_channel.set_index('ResearchDate')
                col_name = f'Reach {channel}'
                if col_name in df_channel.columns:
                    # Заполняем NaN нулями
                    reach_dfs.append(df_channel[col_name].fillna(0).rename(channel))
    
        reach_df = pd.concat(reach_dfs, axis=1).reset_index()
        reach_df.columns.name = None
        return reach_df

    @staticmethod
    def func_reach_by_dates(data: pd.DataFrame, flag = False):
        """
            Reach по дням для расчета совокупных показателей по всем каналам/радио-станциям
            Если flag = True, то считаем совокупные показатели ТВ+Радио
        """
    
        results_by_dates = {}
        for target_date in data['ResearchDate'].unique():
            
            table = data[data['ResearchDate'] == target_date]
            # Отбираем только те события, которые больше минуты
            table = table[table['Duration'] >= 30].reset_index(drop = True)
            
            t = pd.DataFrame()
            if flag:
                t = table[['SubjectID', 'Media', 'weight_new']]
            else:
                
                t = table[['SubjectID', 'weight_new']]
                
            cleaned = t.drop_duplicates()
            results_by_dates[target_date] = cleaned['weight_new'].sum()
        
        reach_by_dates = pd.DataFrame.from_dict(results_by_dates, orient='index', columns = ['Reach']).reset_index()
        reach_by_dates.rename(columns = {'index': 'Date'}, inplace = True)
        return reach_by_dates
    
    @staticmethod
    def func_reach_by_months(data: pd.DataFrame):
        """
            Reach по месяцам для расчета совокупных показателей по всем каналам/радио-станциям
        """
    
        reach_by_dates = data.copy()
        reach_by_dates['month'] = pd.to_datetime(reach_by_dates['Date']).dt.month
    
        results_by_months = {}
        for month in reach_by_dates['month'].unique():
            month_data = reach_by_dates[reach_by_dates['month'] == month]
            if month == 1:
                results_by_months['Январь 2026'] = month_data['Reach'].sum() / 31
            elif month == 2:
                results_by_months['Февраль 2026'] = month_data['Reach'].sum() / 28
            elif month == 3:
                results_by_months['Март 2026'] = month_data['Reach'].sum() / 31
        
        reach_by_months = pd.DataFrame.from_dict(results_by_months, orient = 'index', columns = ['Reach']).reset_index()
        reach_by_months.rename(columns = {'index': 'Месяц'}, inplace = True)
        return reach_by_months
