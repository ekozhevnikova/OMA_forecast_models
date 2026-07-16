import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import xlsxwriter
import copy
import re
import os
from datetime import datetime, timedelta, date

import threading
from threading import Lock
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed


import locale
locale.setlocale(locale.LC_ALL,'ru_RU')

from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.io_data.dates import Dates_Operations

from OMA_tools.regions.data_extraction.task_builder import *
from OMA_tools.regions.data_extraction.data_coworker import EmployeeExportService
from OMA_tools.regions.data_extraction.leader_ship import LeaderShipDataExtractor


class KUS_Calculation:
    """
        Класс для выгрузки данных по Эфирному рейтингу для дальнейшего прогнозирования КУСа.
    """

    def __init__(self, date_filter):
        self.date_filter = date_filter
    

    @staticmethod
    def format_time(time_int):
        """
            Вспомогательный метод для преобразования временного слота
        """
        if time_int[:2] in ['24', '25', '26', '27', '28', '29']:
            time_int = str(int(time_int[:2]) - 24) + '0000'

        time_str = str(time_int).zfill(6)   

        return f'{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}'
    

    @staticmethod
    def extract_city(channel_name: str):
        """
            Вспомогательный метод для извлечения города из названия канала
        """
        match = re.search(r'\((.*?)\)', channel_name)
        if match:
            return match.group(1)
        return None
        
        
    @staticmethod
    def sum_2_channels(
                df_1: pd.DataFrame, 
                df_2: pd.DataFrame, 
                keys: list, 
                flag = False
                ) -> pd.DataFrame:
        """
            Функция для суммирования значений по двум каналам. Применяется для Р1, ГТРК, Пятница (ЕКБ), Четвертый канал (ЕКБ)
            flag = False: обновление в середине месяца
            flag = True: закрытие месяца
        """
        # Суммирование России 1 и ГТРК
        data_merged = pd.merge(df_1, df_2, on = keys, how = 'left')
        data_merged['TVR'] = data_merged['TVR_x'] + data_merged['TVR_y']
        data_merged.rename(columns = {'Month_x': 'Month', 'ChannelBCA_x': 'ChannelBCA'}, inplace = True)

        if flag == True:
            columns = ['Month', 'ChannelBCA', 'time', 'TVR']
        else:
            columns = ['Month', 'ChannelBCA', 'TVR'] 
            
        return data_merged[columns]
    

    @staticmethod
    def clean_table(table):
        """
            Метод для зачистки данных.
        """
        # Создаем копию, чтобы не модифицировать исходные данные
        table = table.copy()

        # 1. Обработка ChannelBCA
        table['ChannelBCA'] = table['ChannelBCA'].str.replace(') ', ',', regex = False)

        # Разделение на Channel и BCA
        table[['Channel', 'BCA']] = table['ChannelBCA'].str.split(',', expand = True, n = 1)

        # 2. Нормализация названий каналов
        channel_replacements = {
            'ТЕЛЕКАНАЛ': 'канал',
            'САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ': 'телеканал санкт-петербург',
            ' (': ' ',
            'ПЕРВЫЙ КАНАЛ': 'ПЕРВЫЙ',
            'ПЯТЫЙ КАНАЛ': '5 канал',
            'ТВ-3': 'тв3'
        }

        for old, new in channel_replacements.items():
            table['Channel'] = table['Channel'].str.replace(old, new, regex = False)

        # 3. Приведение к нижнему регистру
        table['Channel'] = table['Channel'].str.lower()
        table['BCA'] = table['BCA'].str.lower()

        # 4. Обработка даты
        if 'Month' in table.columns and not table.empty:
            try:
                first_month = pd.to_datetime(table['Month'].iloc[0])
                month_str = datetime.strftime(first_month, '%B %Y')
                table['Month'] = month_str
            except (ValueError, TypeError) as e:
                print(f"Ошибка при обработке даты: {e}")

        # 5. Удаление исходной колонки
        table = table.drop(['ChannelBCA'], axis = 1)

        return table
    

    def get_data_total_day(
                    self,
                    tasks_json: dict,
                    bca: str,
                    statistics = ['RtgPer'],
                    slices = ['researchMonth', 'tvCompanyName'],
                    ):
        # Убираем проверку на self.tasks_json_fact_month
        if tasks_json is None:
            raise ValueError("tasks_json не передан.")

        #with WrapperNoPrints():
        df = BaseDataService._execute_tasks(tasks_json)  # Используем переданный tasks_json

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[slices + statistics]
        df['BCA'] = bca

        df.rename(columns = {
                            'researchMonth': 'Month', 
                            'RtgPer': 'TVR', 
                            'tvCompanyName': 'Channel'
                            }, 
                  inplace = True)
        filtered_data = df[~df['Channel'].str.contains(' ДОП', na = False)]
        filtered_data_ = filtered_data[['Month', 'Channel', 'BCA', 'TVR']]
        filtered_data_.insert(loc = 2, column = 'Channel new', value = filtered_data_['Channel'] + ' ' + filtered_data_['BCA'])
        data = filtered_data_[['Month', 'Channel new', 'TVR']]
        data_final = data.rename(columns = {'Channel new': 'ChannelBCA'})
        return data_final
    

    def get_data_by_hours(
                    self,
                    tasks_json: dict,
                    bca: str,
                    statistics = ['RtgPer'],
                    slices = ['researchMonth', 'tvCompanyName','timeBand60']
                    ):
        if tasks_json is None:
            raise ValueError("tasks_json не передан.")
        
        #with WrapperNoPrints():
        df = BaseDataService._execute_tasks(tasks_json)

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[slices + statistics]
        df['BCA'] = bca

        df.rename(columns = {
                        'researchMonth': 'Month',
                        'timeBand60':'time', 
                        'RtgPer': 'TVR', 
                        'tvCompanyName': 'Channel'}, 
                        inplace = True)
        filtered_data = df[~df['Channel'].str.contains(' ДОП', na = False)]
        filtered_data_ = filtered_data[['Month','Channel','time', 'BCA', 'TVR']]
        filtered_data_.insert(loc = 2, column = 'Channel new', value = filtered_data_['Channel'] + ' ' + filtered_data_['BCA'])
        data = filtered_data_[['Month', 'Channel new', 'time', 'TVR']]
        data.rename(columns = {'Channel new': 'ChannelBCA'}, inplace = True)
        #Приведение слота к нормальному виду
        data['time'] = data['time'].apply(KUS_Calculation.format_time)
        return data
    

    @staticmethod
    def split_dict_to_dataframes(api_results: dict, keys_to_merge: list, flag = False):
        """
            Вспомогательный метод для генерации датафреймов по основной массе каналов-городов, каналов ГТРК и четвертый канал.
        """
        # 1. Объединение основной массы каналов-городов и локальных каналов Санкт-Петербург
        main = []
        for bca, df in api_results['main'].items():
            main.append(df)
        main_df = pd.concat(main)

        res_local_channels = []
        for bca, df in api_results['spb'].items():
            res_local_channels.append(df)
        local = pd.concat(res_local_channels)
        # Удаление ненужных подстрок из БЦА
        local['ChannelBCA'] = local['ChannelBCA'].str.replace(r'_СПБ_\d+', '', regex = True)

        general_df = pd.concat([main_df, local]).reset_index(drop = True)

        # 2. Объединение каналов ГТРК
        gtrk_res = []
        for bca, df in api_results['gtrk'].items():
            gtrk_res.append(df)
        gtrk = pd.concat(gtrk_res)
        # Удаление ненужных подстрок из БЦА
        gtrk['ChannelBCA'] = gtrk['ChannelBCA'].str.replace(r'_ГТРК_\d+', '', regex = True)

        # 3. Выделение 4 канала
        task_4_channel = list(api_results['channel_4'].values())[0]
        task_4_channel['ChannelBCA'] = task_4_channel['ChannelBCA'].str.replace(' Пятница ЕКБ', '')

        # 4. Применяем функцию к столбцу ChannelBCA
        gtrk['City'] = gtrk['ChannelBCA'].apply(KUS_Calculation.extract_city)

        # 5. Отбор России 1
        russia_1 = general_df[general_df['ChannelBCA'].str.contains('РОССИЯ 1')]
        russia_1['City'] = russia_1['ChannelBCA'].apply(KUS_Calculation.extract_city)

        # 6. Суммирование России 1 и ГТРК
        if flag == True:
            Russia_1 = KUS_Calculation.sum_2_channels(russia_1, gtrk, keys = keys_to_merge, flag = True)
        
        else:
            Russia_1 = KUS_Calculation.sum_2_channels(russia_1, gtrk, keys = keys_to_merge)


        # 7. Отбор ПЯТНИЦА ЕКБ
        pyatniza = general_df[general_df['ChannelBCA'] == 'ПЯТНИЦА (ЕКАТЕРИНБУРГ) все 14-44']
        pyatniza['City'] = pyatniza['ChannelBCA'].apply(KUS_Calculation.extract_city)
        task_4_channel['City'] = task_4_channel['ChannelBCA'].apply(KUS_Calculation.extract_city)

        # 8. Суммирование ПЯТНИЦА ЕКБ и ЧЕТВЕРТЫЙ КАНАЛ
        if flag == True:
            Pyatniza = KUS_Calculation.sum_2_channels(pyatniza, task_4_channel, keys = keys_to_merge, flag = True)
        else:
            Pyatniza = KUS_Calculation.sum_2_channels(pyatniza, task_4_channel, keys = keys_to_merge)

        main = general_df[
                            ~general_df['ChannelBCA'].str.contains('РОССИЯ 1', case = False, na = False)
                            &
                            ~general_df['ChannelBCA'].str.contains('ПЯТНИЦА.*ЕКАТЕРИНБУРГ', case = False, na = False)
                        ]
        
        full_result = pd.concat([main, Russia_1, Pyatniza]).reset_index(drop = True)

        return full_result
    
    

    def extract_data_fact_month(
                            self,
                            tasks_json,
                            statistics = ['RtgPer'],
                            slices = ['researchMonth', 'tvCompanyName'],
                            sortings = {'researchMonth': 'ASC', 'tvCompanyName': 'ASC'}
                            ):
        """
            Метод для извлечения данных за фактическую часть месяца.
        """
        # 1. Выгрузка данных из API
        results = {}
        for group, json_tasks_full in copy.deepcopy(tasks_json).items():  # Используем локальную переменную
            
            # Создаем временный словарь для хранения соответствий
            temp_dict = {}
            for bca, tasks_data in json_tasks_full.items():
                # Сохраняем ключ в данных
                modified_data = {'bca': bca, 'tasks': tasks_data}
                temp_dict[bca] = modified_data
            
            def process_with_bca(modified_data):
                # Создаем экземпляр и передаем tasks_json напрямую в метод
                calculator = KUS_Calculation(self.date_filter)
                return calculator.get_data_total_day(
                    tasks_json=modified_data['tasks'],  # Передаем tasks_json как параметр
                    bca=modified_data['bca']
                )
            
            results[group] = EmployeeExportService.process_tasks_parallel(temp_dict, process_with_bca)
        
        # 2. Формирование выходных датафреймов
        full_result = KUS_Calculation.split_dict_to_dataframes(results, ['City'])

        # Обработка
        results_df = full_result.reset_index(drop = True)
        results_df = KUS_Calculation.clean_table(results_df)
        return results_df
    

    def extract_data_by_slots(
                        self,
                        tasks_json, 
                        data_fact_month,
                        statistics = ['RtgPer'],
                        slices = ['researchMonth', 'tvCompanyName','timeBand60'],
                        sortings = {'researchMonth': 'ASC', 'tvCompanyName': 'ASC'}
                        ):
        """
            Метод для извлечения данных за фактическую часть месяца.
        """
        # 1. Выгрузка данных из API
        results = {}
        for group, json_tasks_full in copy.deepcopy(tasks_json).items():  # Используем локальную переменную
            
            # Создаем временный словарь для хранения соответствий
            temp_dict = {}
            for bca, tasks_data in json_tasks_full.items():
                # Сохраняем ключ в данных
                modified_data = {'bca': bca, 'tasks': tasks_data}
                temp_dict[bca] = modified_data
            
            def process_with_bca(modified_data):
                # Создаем экземпляр и передаем tasks_json напрямую в метод
                calculator = KUS_Calculation(self.date_filter)
                return calculator.get_data_by_hours(
                    tasks_json=modified_data['tasks'],  # Передаем tasks_json как параметр
                    bca=modified_data['bca']
                )
            
            results[group] = EmployeeExportService.process_tasks_parallel(temp_dict, process_with_bca)
        
        # 2. Формирование выходных датафреймов
        full_result = KUS_Calculation.split_dict_to_dataframes(results, ['City', 'time'], flag=True)

        # Обработка и объединение с выгрузкой без разбивки
        results_df_h = full_result.reset_index(drop=True)
        results_df_h = KUS_Calculation.clean_table(results_df_h)

        results_df_all = results_df_h.merge(data_fact_month, how='left', on = ['Month', 'Channel', 'BCA'], suffixes=('', ' Total'))
        return results_df_all.sort_values('time').reset_index(drop=True)
    

    def API_calculation(
                    self, 
                    tasks_json_fact_month: dict,
                    path: str, 
                    jasks_json_by_slots = None,
                    close_month: bool = False
                    ) -> list:
        """
            Метод для постобработки результатов
            Args:
                path: путь к папке, где хранятся исторические данные.
                close_month: флаг, отвечающий за закрытие месяца. Если True, то месяц закрываем. В противном случае нет.
            Return:
                res: список из результатов. Первый элемент - выгрузка по фактической частит месяца, второй элемент - месяц целиком
        """
        if close_month == True:
            print('Выгружаю предыдущий месяц. Пожалуйста, подождите ...')
            fact_month = self.extract_data_fact_month(tasks_json_fact_month)
            print('✅ Данные за фактическую часть месяца выгружены.')
            print('#' * 80)
            print('Выгружаю предыдущий месяц с разбивкой по слотам. Пожалуйста, подождите ...')
            full_month = self.extract_data_by_slots(jasks_json_by_slots, fact_month)
            print('✅ Данные за предыдущий месяц выгружены. Спасибо за ваше ожидание! 😊')
            
            # Форматируем название выгружаемого месяца
            first_date = self.date_filter[0][0] 
            date_object = datetime.strptime(first_date, '%Y-%m-%d')
            formatted_date = date_object.strftime('%B %Y')
            # Загружаем в эксель историрование по месяцам
            full_month.to_excel(path + 'TVR\\' + formatted_date + '.xlsx')
            
            res = [fact_month, full_month]
            return res
        
        else:
            print('Выгружаю фактическую часть месяца. Пожалуйста, подождите ...')
            res = [self.extract_data_fact_month(tasks_json_fact_month), None]
            print('✅ Данные за фактическую часть месяца выгружены. Спасибо за ваше ожидание! 😊')
            return res


class KUS_Forecast:
    def __init__(self, table):
        self.table = table
    
    
    @staticmethod
    def sdate(row):
        return datetime.strptime(row,'%B %Y')
    
    
    @staticmethod
    def div(numer, denom):
        return lambda row: 0 if row[denom] == 0 else row[numer] / row[denom]
    
    
    @staticmethod
    def read_excel_from_start_marker(file_path: str, start_marker: str = 'Начало периода') -> pd.DataFrame:
        """
            Метод для чтения файла. Файл начинает считываться только тогда, когда найдена определенная строка.
            Метод возвращает количество строк, которое нужно пропустить в файле для успешного его чтения.
        """
        # Читаем весь файл для поиска маркера
        df_temp = pd.read_excel(file_path, header = None)

        # Ищем строку с маркером
        start_row = None
        for idx, row in df_temp.iterrows():
            if any(str(cell).strip() == start_marker for cell in row if pd.notna(cell)):
                start_row = idx + 1  # +1 потому что header=None, данные начинаются со следующей строки
                break

        if start_row is None:
            raise ValueError(f"Маркер '{start_marker}' не найден в файле")
        
        return start_row - 1
    
    
    @staticmethod
    def from_file(path, skip = 0, col_list = [], is_vimb = True):
        """
            Метод для чтения данных из папки
        """
        files = []
        files += os.listdir(path)

        result = pd.DataFrame()
        for months in files:
            if is_vimb:
                num_skiprows = KUS_Forecast.read_excel_from_start_marker(path + months)
                table = pd.read_excel(path + months, skiprows = num_skiprows, usecols = col_list)
            else:
                table = pd.read_excel(path + months, skiprows = skip, usecols = col_list)
            result = pd.concat([result, table], ignore_index = True)

        return result
    
    
    @staticmethod
    def parse_VIMB(table, is_current_month = True):
        """
            Метод для парсинга файлов с выгрузкой из ВИМБа.
            flag = True: текущий месяц
            flag = False: закрытие месяца
        """
        table = table.rename(columns = {'Время эфира с' : 'time',
                                        'Начало периода':'Month',
                                        'Открытый объем без учёта квот' : 'Volume',
                                        'Открытый WGRP без учета квот' : 'GRP',
                                        'Базовая ЦА' : 'BCA',
                                        'Ср. рейтинг' : 'tvr_adv',
                                        'Канал': 'Channel'})  

        table['BCA'] = table['BCA'].str.split(' ', expand = True)[0].str.lower() + ' ' + table['BCA'].str.split(' ', expand = True)[1]
        table['Channel'] = table['Channel'].str.lower()
    
        if is_current_month is not True:
            table['time'] = table['time'].astype(str)

        if not pd.api.types.is_datetime64_any_dtype(table['Month']):
            table['Month'] = pd.to_datetime(table['Month'])
        
        # Теперь применяем strftime ко всему столбцу (векторизованно, без цикла)
        table['Month'] = table['Month'].dt.strftime('%B %Y')

        return table
    
    
    def predict(self, data_api: pd.DataFrame, first_date, last_date, filepath_vimb_curr_month: str):
        """
            Метод для непосредственного прогнозирования КУС
            Args:
                data_api: pd.DataFrame: Выгрузка с TVR из БД Mediascope
                first_date: Дата начала вывода данных
                last_date: Дата окончания вывода данных
                filepath_vimb_curr_month: str: путь к файлу с выгрузкой из VIMB по текущему месяцу
            Returns:
                df_result_all: pd.DataFrame: Датафрейм с прогнозом
        """
        
        today = date.today()
        # дата начала и окончания исторических данных для предсказания куса (база прогноза)
        first_fact_date = today - pd.offsets.MonthBegin()- DateOffset(months = 6) #1 число пол года назад
        last_fact_date = today - pd.offsets.MonthBegin()    
        table = self.table

        ################################### ОБЩАЯ ПОДГОТОВКА #############################################################
        table['Month'] = table['Month'].apply(KUS_Forecast.sdate)
        table['PredWeight'] = table['TVR'] / table['TVR Total']
        table['PredKM'] = table.apply(KUS_Forecast.div('tvr_adv', 'TVR'), axis = 1)
        table = table.astype({"PredKM": "float64"})


        ############################# ПОДСЧЕТ ЗАКРЫТЫХ МЕСЯЦЕВ ВСЕ КАНАЛЫ СРАЗУ #####################################################
        zakr_piv = table[(table['Month'] >= first_date) & (table['Month'] < last_fact_date)]
        zakr_piv = zakr_piv.pivot_table(index = 'Channel', columns = ['Month'], values = ['TVR Total', 'Volume', 'GRP'],
                         aggfunc = {'TVR Total': 'mean', 'Volume': 'sum','GRP': 'sum'})

        if zakr_piv.empty != True:
            df_zakrytye = zakr_piv['GRP'] * 20 / (zakr_piv['Volume'] * zakr_piv['TVR Total'])
            
        else:
            df_zakrytye=pd.DataFrame(index=pd.Index(table['Channel'].unique(),name='Channel'))
            print('Нет месяцев в факте!')


        ############################# ПРОГНОЗ МЕСЯЦЕВ ВСЕ КАНАЛЫ СРАЗУ #####################################################
        short_pred = table[(table['Month'] >= last_fact_date) & (table['Month'] <= last_date)]
        short_tmp = table[(table['Month'] >= first_fact_date) & (table['Month'] < last_fact_date)]

        tab_for_fill = short_tmp.pivot_table(index = ['Channel', 'time'],values = ['PredWeight', 'PredKM'],
                                             aggfunc = {'PredWeight': 'mean', 'PredKM': 'median'})

        ans = pd.merge(short_pred[['Channel', 'Month', 'time', 'BCA', 'Volume']], tab_for_fill, how = "left", on = ['Channel', "time"])
        
        ans['PredKM']=ans.groupby('Channel')['PredKM'].transform(lambda x: x.fillna(x.rolling(2, min_periods=1).mean().ffill().bfill()))
        
        df_prognoz = ans.pivot_table(index = ['Channel'], columns = 'Month', values = ['PredWeight'], aggfunc = lambda x: 
                        (ans.loc[x.index, 'PredWeight'] * ans.loc[x.index,'PredKM'] * ans.loc[x.index, 'Volume']).sum()/
                        ans.loc[x.index, 'Volume'].sum()).droplevel(0, axis = 1)

        ############################# ПОДСЧЕТ ТЕКУЩЕГО МЕСЯЦА #####################################################
        # Если текущий месяц январь
        if today.month == 1:
            if today.day > 14:
                print('Реализуем обновление в середине месяца')
                if today.day < 21:
                    fact_w = 0.5
                    prognoz_w = 0.5
                else:
                    fact_w = 0.7
                    prognoz_w = 0.3     

                # Загрузка данных вимб и апи
                adv_tec = pd.read_excel(filepath_vimb_curr_month,
                                    skiprows = 19,
                                    usecols = [0, 5, 6, 7, 12, 13])
                adv_tec = KUS_Forecast.parse_VIMB(adv_tec)
                tvr_prog_tec = data_api
                tec_main = pd.merge(tvr_prog_tec, adv_tec, how = "right", on = ['Channel', 'BCA', "Month"])

                # Подсчет факта
                tec_main['kus_f'] = tec_main['tvr_adv'] / tec_main['TVR']
                results_prognoz = pd.merge(df_prognoz, tec_main[['kus_f','Channel']], how = "left", on = ['Channel'])
                results_prognoz['tec_m_progn'] = results_prognoz.iloc[:,1]

                results_prognoz.iloc[:, 1] = results_prognoz['kus_f'] * fact_w + (results_prognoz.iloc[:, 1]) * prognoz_w

                df_result_all = df_zakrytye.merge(results_prognoz,on = 'Channel', how = 'left')

            else:
                print('Закрытие месяца!')
                df_result_all = df_zakrytye.merge(df_prognoz, on = 'Channel', how = 'left')

        # Если текущий месяц не январь
        else:
            if today.day > 10:
                print('Реализуем обновление в середине месяца')
                if today.day < 21:
                    fact_w = 0.5
                    prognoz_w = 0.5
                else:
                    fact_w = 0.7
                    prognoz_w = 0.3     

                    #Загрузка данных вимб и апи
                adv_tec = pd.read_excel(filepath_vimb_curr_month,
                                    skiprows = 19,
                                    usecols = [0, 5, 6, 7, 12, 13])
                adv_tec = KUS_Forecast.parse_VIMB(adv_tec)
                tvr_prog_tec = data_api
                print(adv_tec)
                print(adv_tec.dtypes)
                tec_main = pd.merge(tvr_prog_tec, adv_tec, how = "right", on = ['Channel', 'BCA', "Month"])

                        #Подсчет факта
                tec_main['kus_f'] = tec_main['tvr_adv'] / tec_main['TVR']
                results_prognoz = pd.merge(df_prognoz, tec_main[['kus_f','Channel']], how = "left", on = ['Channel'])
                results_prognoz['tec_m_progn'] = results_prognoz.iloc[:,1]

                results_prognoz.iloc[:, 1] = results_prognoz['kus_f'] * fact_w + (results_prognoz.iloc[:, 1]) * prognoz_w

                df_result_all = df_zakrytye.merge(results_prognoz,on = 'Channel', how = 'left')

            else:
                print('Закрытие месяца!')
                df_result_all = df_zakrytye.merge(df_prognoz, on = 'Channel', how = 'left')

        return df_result_all