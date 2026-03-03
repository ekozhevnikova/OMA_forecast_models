import pandas as pd
import numpy as np
import datetime as dt
import xlsxwriter
from datetime import datetime, timedelta
import pymorphy3 as pmrph
import time
import copy
import threading
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
import OMA_tools
from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.io_data.dates import Dates_Operations

from OMA_tools.regions.data_extraction.task_builder import *
from OMA_tools.regions.data_extraction.leader_ship import DataConfig

import warnings
warnings.filterwarnings('ignore')


class EmployeeExportService:
    def __init__(self, date_filter, tasks_json):
        self.date_filter = date_filter
        self.tasks_json = tasks_json
    

    @staticmethod
    def get_output(df):
        """
            Форматирование выходных данных
        """
        df = df[df.columns.drop(list(df.filter(regex = ' / ДОП ')))]
        df = df.rename(columns={'regionName': 'City'})
        result_df = pd.pivot_table(
                            df, 
                            values = 'Share', 
                            index = None, 
                            columns = ['tvCompanyName']
                            )
        result_df = result_df[result_df.columns.drop(list(result_df.filter(regex = ' / ДОП ')))]
        return result_df
    

    @staticmethod
    def sum_2_channels(df_channel_1, df_channel_2, regions: list):
        """
            Функция для суммирования значений по двум каналам.
        """
        if df_channel_1.empty or df_channel_2.empty:
            return df_channel_1 if not df_channel_1.empty else df_channel_2
            
        df_channel_1['Date'] = pd.to_datetime(df_channel_1['Date'])
        df_channel_2['Date'] = pd.to_datetime(df_channel_2['Date'])
        full = []
        
        for i in range(len(regions)):
            # Находим колонки, содержащие название региона
            channel1_cols = [col for col in df_channel_1.columns if regions[i] in col]
            channel2_cols = [col for col in df_channel_2.columns if regions[i] in col]
            
            if not channel1_cols or not channel2_cols:
                continue
                
            data_1 = df_channel_1[['Date', channel1_cols[0]]]
            data_2 = df_channel_2[['Date', channel2_cols[0]]]
            data = pd.merge(data_1, data_2, on='Date', how='outer')
            data['total'] = data[[data.columns[1], data.columns[2]]].astype(float).sum(1)
            data_ = data[['Date', 'total']]
            data_final = data_.rename(columns={'total': df_channel_1.columns[i+1]})
            full.append(data_final)
            
        if not full:
            return df_channel_1
            
        merged_channels = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), full)
        return merged_channels


    def get_data_by_days(
                self,
                statistics = DataConfig.STATISTICS,
                slices = ['regionName', #регион
                      'tvCompanyName',
                      'researchDate']
                     ):
        """
            Метод для генерации задания API для руководителей групп
        """
        #with WrapperNoPrints():
        df = BaseDataService._execute_tasks(self.tasks_json)    

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[slices + statistics]
        df_= pd.pivot_table(df, values = statistics,
                            index = ['researchDate'], 
                            columns = ['tvCompanyName'])
        
        return df_
    

    def get_data_fact_month(
                self,
                statistics = DataConfig.STATISTICS,
                slices = ['regionName', #регион
                            'tvCompanyName' #телесеть
                            ]
                     ):
        """
            Метод для генерации задания API для руководителей групп
        """
        #with WrapperNoPrints():
        df = BaseDataService._execute_tasks(self.tasks_json)    

        return df
    

    def get_data_by_months(
                self,
                statistics = DataConfig.STATISTICS,
                slices = ['regionName', #регион
                      'tvCompanyName',
                      'researchMonth'],
                flag = True
                     ):
        """
            Метод для генерации задания API для руководителей групп
        """
        #with WrapperNoPrints():
        df = BaseDataService._execute_tasks(self.tasks_json)    

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[slices + statistics]
        df_= pd.pivot_table(df, values = statistics,
                            index = ['researchMonth'], 
                            columns = ['tvCompanyName'])
        return df_
    

    @staticmethod
    def process_tasks_parallel(json_tasks_full, process_func, max_workers = 10):
        """
        Универсальная параллельная обработка задач
        
        Parameters:
            - json_tasks_full: словарь {bca: json_tasks}
            - process_func: функция для обработки каждой задачи
            - max_workers: количество потоков
        
        Returns:
            - словарь {bca: результат}
        """
        def process_single_bca(bca_task):
            bca, json_tasks = bca_task
            return bca, process_func(json_tasks)

        tsks = {}
        lock = threading.Lock()
        tasks_list = list(json_tasks_full.items())

        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            results = executor.map(process_single_bca, tasks_list)
            
            for bca, result in results:
                with lock:
                    tsks[bca] = result
        
        return tsks
    

    @staticmethod
    def extract_russia_1__and__pyatniza(
                        dict_data: dict, 
                        GTRK: pd.DataFrame, 
                        df_channel_4: pd.DataFrame, 
                        columns_order_dict: dict, 
                        local_channels_list: list, 
                        columns_order_russia_1 = DataConfig.RUSSIA_COLUMNS, 
                        cities_russia_1 = DataConfig.RUSSIA_1_CITIES) -> dict:
        """
            Метод для выделения Р1 и Пятница (Екатеринбург) из словаря с данными
            Args:
                dict_data: Словарь с выгрузкой API для основной массы каналов-городов
                GTRK: Датафрейм с каналми ГТРК Все 18+
                df_channel_4:Датафрейм с выгрузкой по каналу Четвертый канал (Екатеринбург) Все 14-44
                columns_order_dict: словарь с порядком колонок для каждой БЦА, ключ: БЦА, значение: список с порядком колонок
                local_channels_list: Результат выгрузки по локальным каналам Телеканал 78 Санкт-Петербург и Санкт-Петербург
                columns_order_russia_1: Порядок колонок для датафрейма с Россия 1
                cities_russia_1: список городов, которые относятся к каналу Россия 1
            Returns:
                df_dict: словарь из датафреймов с нужным порядком столбцов, а также с просуммированными значениями Р1+ГТРК, Пятница(ЕКБ) + Четвертый канал
        """
        # 1. Манипуляции с Россия 1
        Russia_1 = dict_data['All 18+'][columns_order_russia_1]

        #Удаление России 1 из Датафрейма
        data_all_18_senza_russia_1 = dict_data['All 18+'].drop([col for col in dict_data['All 18+'].columns if 'РОССИЯ 1' in col], axis = 1)


        # 2. Выделение Пятница (Екатеринбург)
        dict_data['All 14-44']['Date'] = pd.to_datetime(dict_data['All 14-44']['Date'])
        pyatniza = dict_data['All 14-44'][['Date', 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)']]

        #Удаление Пятница (Екатеринбург) из Датафрейма
        data_all_14_44_senza_pyatniza_ekb = dict_data['All 14-44'].drop([col for col in dict_data['All 14-44'].columns if 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)' in col], axis = 1)

        # 3. Суммирование значений по России 1 и ГТРК
        Russia_1_updated = EmployeeExportService.sum_2_channels(Russia_1, GTRK, cities_russia_1)

        #Обновление датафрейма для Все 18+
        new_data_all_18 = pd.merge(data_all_18_senza_russia_1, Russia_1_updated, how = 'left', on = 'Date')
        dict_data['All 18+'] = Table.make_left_join(new_data_all_18, local_channels_list[0], local_channels_list[1], 'Date')
        dict_data['All 18+'] = dict_data['All 18+'][columns_order_dict['All 18+']]
        
        
        # 4. Суммирование значений долей Пятница(Екатеринбург) с Четвертым каналом (Екатеринбург)
        pyatniza_updated = EmployeeExportService.sum_2_channels(pyatniza, df_channel_4, ['ЕКАТЕРИНБУРГ'])
        
        # 5. Обновление датафрейма для Все 14-44
        new_data_all_14_44 = pd.merge(data_all_14_44_senza_pyatniza_ekb, pyatniza_updated, how = 'left', on = 'Date')
        dict_data['All 14-44'] = new_data_all_14_44[columns_order_dict['All 14-44']]

        result_dict = {'All 18+': dict_data['All 18+'], 
                    'All 14-59': dict_data['All 14-59'], 
                    'All 10-45': dict_data['All 10-45'], 
                    'All 14-44': dict_data['All 14-44'], 
                    'All 14-54': dict_data['All 14-54'], 
                    'All 25-49': dict_data['All 25-49'], 
                    'All 25-54': dict_data['All 25-54'], 
                    'All 4-45': dict_data['All 4-45'], 
                    'All 6-54': dict_data['All 6-54'], 
                    'W 14-44': dict_data['W 14-44'], 
                    'W 25-59': dict_data['W 25-59']}
        
        # 6. Переупорядочивание столбцов в словаре в соответствии с порядком в словаре columns_order_dict
        df_dict = Dict_Operations(result_dict).rename_columns_in_dict_with_df(columns_order_dict)

        return df_dict
    

    @staticmethod
    def merge_gtrk_dataframes(
                        gtrk_dataframes_list: list, 
                        gtrk_regions: list, 
                        columns_order_gtrk = DataConfig.GTRK_COLUMNS
                        ):
        """
            Метод для создания единого датафрейма каналов ГТРК с определенным порядком столбцов
            Args:
                gtrk_dataframes_list: Список датафреймов с выгрузкой API для каналов ГТРК
                gtrk_regions: список используемых регионов для ГТРК
                columns_order_gtrk: порядок колонок каналов ГТРК
            Returns:
        """
        # Совмещение ГТРК в один единый датафрейм
        merged_gtrk = gtrk_dataframes_list[0]
        for df in gtrk_dataframes_list[1:]:
            merged_gtrk = pd.merge(merged_gtrk, df, on = 'Date', how = 'left')

        # Переименование столбцов для каналов ГТРК  
        current_columns = list(merged_gtrk.columns[1:])

        # Извлекаем все названия регионов из списка
        region_names = []
        for region_dict in gtrk_regions:
            region_names.extend(list(region_dict.values()))

        # Создаем словарь переименования автоматически
        auto_rename_dict = {}
        for i, old_col in enumerate(current_columns):
            if i < len(region_names):
                city = old_col.split('(')[-1].replace(')', '').strip()
                new_name = f'ГТРК {region_names[i]}'
                auto_rename_dict[old_col] = new_name

        # Применяем переименование
        GTRK = merged_gtrk.rename(columns = auto_rename_dict)
        return GTRK[columns_order_gtrk]
    

    def make_api_calculation_by_days(
                            self, 
                            gtrk_regions: list,
                            columns_order: str,
                            historical_filepath_by_days: str,
                            filepath_last_n_days: str,
                            LAST_N_DAYS: int = 28,
                            date_column: str = 'Date'
                            ):
        # Загрузка порядка столбцов
        loaded_dict_columns = Dict_Operations.load_pkl_file(columns_order)

        ############################# 1. Выгрузка данных для всевозможных групп ##################
        results = {}
        for group, json_tasks_full in copy.deepcopy(self.tasks_json).items():
            results[group] = EmployeeExportService.process_tasks_parallel(
                                                        json_tasks_full, 
                                                        lambda x: EmployeeExportService(self.date_filter, x).get_data_by_days()
                                                    )
        

        results_new = {}
        for bca, df in results['main'].items():
            df_new = Table(df).make_table(column_name = 'Date')
            df_new = df_new[df_new.columns.drop(list(df_new.filter(regex = ' ДОП ')))]
            df_new['Date'] = pd.to_datetime(df_new['Date'])
            results_new[bca] = df_new
        
        ############################# 2. Манипуляции с каналами ГТРК #############################
        res_gtrk = []
        for bca, df in results['gtrk'].items():
            t_new = Table(df).make_table(column_name = 'Date')
            t_new['Date'] = pd.to_datetime(t_new['Date'])
            res_gtrk.append(t_new)
        
        # Объединение каналов ГТРК с один единый датафрейм с определенным порядком колонок, чтобы потом было удобнее суммировать с Россия 1
        full_vgtrk = EmployeeExportService.merge_gtrk_dataframes(res_gtrk, gtrk_regions)

        ############################# 3. Манипуляции с каналами Санк-Петербург и Телеканал 78 #############################
        tasks_local_channels = []
        for key, df in results['spb'].items():
            t_new = Table(df).make_table(column_name = 'Date')
            t_new['Date'] = pd.to_datetime(t_new['Date'])
            tasks_local_channels.append(t_new)
        
        ############################# 4. Манипуляции с Четвертый канал (Екатеринбург) #############################
        t = Table(list(results['channel_4'].values())[0]).make_table(column_name = 'Date')
        t['Date'] = pd.to_datetime(t['Date'])
        column = t.columns[1]
        column_name = 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44'
        df = t.rename(columns = {f'{column}': f'{column_name}'})
        df_channel_4 = df[['Date', 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44']]

        ########################################################## П Р Е О Б Р А З О В А Н И Я ##########################################################
        # 5. Суммирование данных по России 1 + ГТРК, Пятница (Екатеринбург) + Четвертый канал (Екатеринбург)
        df_dict = EmployeeExportService.extract_russia_1__and__pyatniza(
                                                        results_new,
                                                        full_vgtrk, 
                                                        df_channel_4, 
                                                        loaded_dict_columns, 
                                                        tasks_local_channels)

        # 6. Обновление файла с историческими данными
        for key, df in df_dict.items():
            df_dict[key][date_column] = df_dict[key][date_column].apply(lambda x: pd.to_datetime(x))

        #data_old = File(historical_filepath_by_days).from_file(0)
        data_new = File(filename = historical_filepath_by_days).update_file(
                                                            df_dict,
                                                            date_column, 
                                                            DataConfig.BCA_LIST
                                                                )
        
        try:
            # Установка внешнего вида итоговой таблицы по дням за последние 28 дней
            writer = pd.ExcelWriter(historical_filepath_by_days, engine = 'xlsxwriter')
            for key, df in data_new.items():
                Table(df = df).make_style_of_table(writer = writer, sheet_name = key, width_col_1 = 4.5, width_col_2 = 17.57, width_col_3 = 15.86)
            writer.close()
            print("✅ Файл с историческими данными успешно сохранен")
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")
        


        df_dict_tail = {}
        for bca, df in df_dict.items():
            if len(df_dict[bca]) < LAST_N_DAYS:
                raise ValueError('Количество выгружаемых дней не соответствует количеству дней, записываемых в файл. Выберите другой временной период')
            df_dict_tail[bca] = df.tail(LAST_N_DAYS) #Записывает последние 30 значений из выгрузки
        
        # 6. Сохранение последних 28 дней в файл
        File(filename = filepath_last_n_days).to_file(df_dict_tail)

        try:
            # Установка внешнего вида итоговой таблицы по дням за последние 28 дней
            writer = pd.ExcelWriter(filepath_last_n_days, engine = 'xlsxwriter')
            for key, df in df_dict_tail.items():
                Table(df = df).make_style_of_table(writer = writer, sheet_name = key, width_col_1 = 4.5, width_col_2 = 17.57, width_col_3 = 15.86)
            writer.close()
            print("✅ Файл с выгрузкой по дням успешно сохранен")
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")
        
        return data_new, df_dict_tail


    
    def make_api_calculation_fact_month(
                            self,
                            gtrk_regions: list,
                            columns_order: str,
                            filepath_by_months: str,
                            date_column: str = 'Date',
                            statistics = DataConfig.STATISTICS,
                            slices = ['regionName', #регион
                                      'tvCompanyName' #телесеть
                            ]
                            ):
        # Выделение даты старта
        target_date = self.date_filter[0][0]

        # Загрузка порядка столбцов
        loaded_dict_columns = Dict_Operations.load_pkl_file(columns_order)

        ############################# 1. Выгрузка данных для всевозможных групп ##################
        results = {}
        for group, json_tasks_full in copy.deepcopy(self.tasks_json).items():
            results[group] = EmployeeExportService.process_tasks_parallel(
                                                        json_tasks_full, 
                                                        lambda x: EmployeeExportService(self.date_filter, x).get_data_fact_month()
                                                    )
        
        results_new = {}
        for bca, df in results['main'].items():
            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[['prj_name'] + slices + statistics]
            df['prj_name'] = df['prj_name'].str.upper()
            df_new = EmployeeExportService.get_output(df)
            df_new = df_new.reset_index()
            df_new = df_new.rename(columns = {df_new.columns[0]: 'Date'})
            df_new_ = df_new.replace(list(df_new['Date']), target_date)
            df_new_['Date'] = pd.to_datetime(df_new_['Date'])
            results_new[bca] = df_new_
        

        ############################# 2. Манипуляции с каналами ГТРК #############################
        res_gtrk = []
        for bca, df in results['gtrk'].items():
            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[['prj_name'] + slices + statistics]
            df['prj_name'] = df['prj_name'].str.upper()
            df_new = EmployeeExportService.get_output(df)
            df_new = df_new.reset_index()
            df_new = df_new.rename(columns = {df_new.columns[0]: 'Date'})
            df_new_ = df_new.replace(list(df_new['Date']), target_date)
            df_new_['Date'] = pd.to_datetime(df_new_['Date'])
            res_gtrk.append(df_new_)
        
        # Объединение каналов ГТРК с один единый датафрейм с определенным порядком колонок, чтобы потом было удобнее суммировать с Россия 1
        full_vgtrk = EmployeeExportService.merge_gtrk_dataframes(res_gtrk, gtrk_regions)

        ############################# 3. Манипуляции с каналами Санк-Петербург и Телеканал 78 #############################
        tasks_local_channels = []
        for key, df in results['spb'].items():
            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[['prj_name'] + slices + statistics]
            df['prj_name'] = df['prj_name'].str.upper()
            df_new = EmployeeExportService.get_output(df)
            df_new = df_new.reset_index()
            df_new = df_new.rename(columns = {df_new.columns[0]: 'Date'})
            df_new_ = df_new.replace(list(df_new['Date']), target_date)
            df_new_['Date'] = pd.to_datetime(df_new_['Date'])
            tasks_local_channels.append(df_new_)
        
        ############################# 4. Манипуляции с Четвертый канал (Екатеринбург) #############################
        task_4_channel = list(results['channel_4'].values())[0]

        task_4_channel = task_4_channel[slices + statistics]
        task_4_channel_ = pd.pivot_table(task_4_channel, values = statistics,
                            index = None, 
                            columns = ['tvCompanyName'])
        task_4_channel_['Date'] = target_date
        task_4_channel_['Date'] = pd.to_datetime(task_4_channel_['Date'])

        column = task_4_channel_.columns[0]
        column_name = 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44'
        channel_4 = task_4_channel_.rename(columns = {f'{column}': f'{column_name}'})
        df_channel_4 = channel_4[['Date', 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44']]

        ########################################################## П Р Е О Б Р А З О В А Н И Я ##########################################################
        # Суммирование значений по Россия 1 + ГТРК, Пятница (Екатеринбург) + Четвертый канал. Сбор в один единый словарь.
        df_dict = EmployeeExportService.extract_russia_1__and__pyatniza(
                                                        results_new,
                                                        full_vgtrk, 
                                                        df_channel_4, 
                                                        loaded_dict_columns, 
                                                        tasks_local_channels)

        replacements = {'Share': datetime.today().strftime('%B %Y')}
        for bca, df in df_dict.items():
            df_dict[bca].set_index('Date', inplace = True)
            df_dict[bca].index = [replacements.get(x, x) for x in df_dict[bca].index]
            df_dict[bca] = df_dict[bca].reset_index()
            df_dict[bca] = df_dict[bca].rename(columns = {df_dict[bca].columns[0]: 'Date'})
        dict_data_new = Dict_Operations(df_dict).convert_column_with_date('Date')

    
        data_old = File(filepath_by_months).from_file(0, 0)
        full_data = Dict_Operations(data_old).replace_keys_in_dict(list_of_replacements = DataConfig.BCA_LIST)

        for bca, df in full_data.items():
            if full_data['All 18+'].iloc[-1][0] == dict_data_new['All 18+'].iloc[-1][0]:
                full_data[bca] = full_data[bca].iloc[:-1]
                res = Dict_Operations.make_concat_of_dicts_dataframes(full_data, dict_data_new)
            else:
                res = Dict_Operations.make_concat_of_dicts_dataframes(full_data, dict_data_new)
        File(filepath_by_months).to_file(res)
        
        #Придание внешнего вида итоговой таблице
        df_dict = File(filepath_by_months).from_file(0, 0)
        data = Dict_Operations(df_dict).replace_keys_in_dict(list_of_replacements = DataConfig.BCA_LIST)
        
        try:
            writer = pd.ExcelWriter(filepath_by_months, engine = 'xlsxwriter')
            for key, df in data.items():
                Table(df = df).make_style_of_table(writer = writer, 
                                                sheet_name = key, 
                                                width_col_1 = 4.5, 
                                                width_col_2 = 13.43, 
                                                width_col_3 = 13.0)
            writer.close()
            print("✅ Файл успешно сохранен")

        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")

        return dict_data_new
    


    def make_api_calculation_by_months(
                            self,
                            gtrk_regions: list,
                            columns_order: str,
                            filepath_by_months: str,
                            date_column: str = 'Date'
                            ):
        # Загрузка порядка столбцов
        loaded_dict_columns = Dict_Operations.load_pkl_file(columns_order)

        ############################# 1. Выгрузка данных для всевозможных групп ##################
        results = {}
        for group, json_tasks_full in copy.deepcopy(self.tasks_json).items():
            results[group] = EmployeeExportService.process_tasks_parallel(
                                                        json_tasks_full, 
                                                        lambda x: EmployeeExportService(self.date_filter, x).get_data_by_months()
                                                    )
        
        results_new = {}
        for bca, df in results['main'].items():
            df_new = Table(df).make_table(column_name = 'Date')
            df_new['Date'] = df_new['Date'].apply(lambda x: pd.to_datetime(x))
            df_new = df_new[df_new.columns.drop(list(df_new.filter(regex = ' ДОП ')))]
            results_new[bca] = df_new
        

        ############################# 2. Манипуляции с каналами ГТРК #############################
        res_gtrk = []
        for bca, df in results['gtrk'].items():
            df_new = Table(df).make_table(column_name = 'Date')
            df_new['Date'] = df_new['Date'].apply(lambda x: pd.to_datetime(x))
            df_new = df_new[df_new.columns.drop(list(df_new.filter(regex = ' ДОП ')))]
            res_gtrk.append(df_new)
        

        # Объединение каналов ГТРК с один единый датафрейм с определенным порядком колонок, чтобы потом было удобнее суммировать с Россия 1
        full_vgtrk = EmployeeExportService.merge_gtrk_dataframes(res_gtrk, gtrk_regions)

        ############################# 3. Манипуляции с каналами Санк-Петербург и Телеканал 78 #############################
        tasks_local_channels = []
        for key, df in results['spb'].items():
            t_new = Table(df).make_table(column_name = 'Date')
            t_new['Date'] = t_new['Date'].apply(lambda x: pd.to_datetime(x))
            tasks_local_channels.append(t_new)
        
        ############################# 4. Манипуляции с Четвертый канал (Екатеринбург) #############################
        task_4_channel = list(results['channel_4'].values())[0]

        t = Table(task_4_channel).make_table(column_name = 'Date')
        column = t.columns[1]
        column_name = 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44'
        df = t.rename(columns = {f'{column}': f'{column_name}'})
        df_channel_4 = df[['Date', 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44']]

        ########################################################## П Р Е О Б Р А З О В А Н И Я ##########################################################
        df_dict = EmployeeExportService.extract_russia_1__and__pyatniza(
                                                        results_new,
                                                        full_vgtrk, 
                                                        df_channel_4, 
                                                        loaded_dict_columns, 
                                                        tasks_local_channels)
        
        dict_data_new = Dict_Operations(df_dict).convert_column_with_date('Date')

        # Сохранение данных в файл
        data = File(filepath_by_months).from_file(0, 0)
        data_old = Dict_Operations(data).replace_keys_in_dict(list_of_replacements = DataConfig.BCA_LIST)

        for bca, df in data_old.items():
            #Проверка на то, что последние две строчки в исходном DataFrame различны
            if data_old[bca].iloc[-1][0] == data_old[bca].iloc[-2][0]:
                data_old[bca] = data_old[bca].iloc[:-2]
            #Join выгрузки и исходного DataFrame
            if data_old[bca].iloc[-1][0] == dict_data_new[bca].iloc[-1][0]:
                data_old[bca] = data_old[bca].iloc[:-1]
                res = Dict_Operations.make_concat_of_dicts_dataframes(data_old, dict_data_new)
            else:
                res = Dict_Operations.make_concat_of_dicts_dataframes(data_old, dict_data_new)

        #Сохранение в файл
        File(filepath_by_months).to_file(res)

        #Придание внешнего вида итоговой таблице
        df_dict = File(filepath_by_months).from_file(0, 0)
        data = Dict_Operations(df_dict).replace_keys_in_dict(list_of_replacements = DataConfig.BCA_LIST)
        try:
            writer = pd.ExcelWriter(filepath_by_months, engine = 'xlsxwriter')
            for key, df in data.items():
                Table(df = df).make_style_of_table(writer = writer, 
                                                sheet_name = key, 
                                                width_col_1 = 4.5, 
                                                width_col_2 = 13.43, 
                                                width_col_3 = 13.0)
            writer.close()
            print("✅ Файл успешно сохранен")

        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")

        return dict_data_new


    @staticmethod
    def update_monthly_data(current_year: int, df: pd.DataFrame, columns_order: str, filepath_by_months: str):
        """
            Новый метод для получения данных за фактическую часть месяца
            Args:
                current_year: текущий год
                df: датафрейм с выгрузкой фактической части месяца, которая была получена в выгрузке для руководителей групп
        """
        # Загрузка порядка столбцов
        loaded_dict_columns = Dict_Operations.load_pkl_file(columns_order)

        data_anal = df.copy()

        # Получаем название столбца с периодом
        period_column = data_anal.columns[-1]

        # Разбиваем строку с периодом на подстроку с целью извлечения месяца
        text = period_column
        splitted = text.split(' ')
        month = splitted[-1]
        # Приводим название к первоначальному виду и добавляем текущий год
        morph = pmrph.MorphAnalyzer()
        month_normalized = morph.parse(month)[0].normal_form.capitalize()
        period_column_new = month_normalized + ' ' + str(current_year)

        # Переименовываем названия БЦА и оставляем только нужные колонки для дальнейшего анализа
        gender_map = {
            'ВСЕ 18+': 'All 18+', 
            'ВСЕ 14-59': 'All 14-59', 
            'Ж 25-59': 'W 25-59', 
            'ВСЕ 25-54': 'All 25-54', 
            'ВСЕ 6-54': 'All 6-54', 
            'ВСЕ 10-45': 'All 10-45', 
            'ВСЕ 14-54': 'All 14-54', 
            'ВСЕ 14-44': 'All 14-44', 
            'ВСЕ 4-45': 'All 4-45', 
            'Ж 14-44': 'W 14-44', 
            'ВСЕ 25-49': 'All 25-49', 
        }
        data_anal['БЦА'] = data_anal['БЦА'].map(gender_map)
        data_anal['tvCompanyName'] = data_anal['Телеканал'] + ' (' + data_anal['Город'] + ')'
        data_anal = data_anal[['tvCompanyName', 'БЦА', period_column]]

        # Переименовываем названия локальных каналов
        data_anal['tvCompanyName'].replace(
                        {
                            'ТЕЛЕКАНАЛ 78 САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)': 'ТЕЛЕКАНАЛ 78 (САНКТ-ПЕТЕРБУРГ)', 
                            'ТЕЛЕКАНАЛ САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)': 'САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)'
                            }, 
                        inplace = True)

        # Создаем словарь, в котором в дальнейшем будем выстраивать нужный порядок столбцов
        bca_unique = data_anal['БЦА'].unique()
        results_data_anal = {}
        for bca in bca_unique:
            df = data_anal[data_anal['БЦА'] == bca].reset_index(drop = True)
            df.drop('БЦА', axis = 1, inplace = True)
            df = df.T
            
            headers = df.iloc[0].tolist()  # берем всю вторую строку
            
            # Третья строка (индекс 2) - это данные
            
            data = df.iloc[1].tolist()     # берем всю третью строку
            # Создаем DataFrame с одной строкой данных
            
            result = pd.DataFrame([data], columns = headers)
            
            result.insert(0, 'Date', period_column_new)
            
            results_data_anal[bca] = result


        dict_data_new = Dict_Operations(results_data_anal).rename_columns_in_dict_with_df(loaded_dict_columns)

        # Сохранение данных в файл
        data = File(filepath_by_months).from_file(0, 0)
        data_old = Dict_Operations(data).replace_keys_in_dict(list_of_replacements = DataConfig.BCA_LIST)

        for bca, df in data_old.items():
            # Проверка на то, что последние две строчки в исходном DataFrame различны
            if len(data_old[bca]) >= 2 and data_old[bca].iloc[-1][0] == data_old[bca].iloc[-2][0]:
                data_old[bca] = data_old[bca].iloc[:-1]  # Удаляем только последнюю дублирующуюся строку, а не две
            
            # Join выгрузки и исходного DataFrame
            if bca in dict_data_new:  # Проверяем, что ключ существует в новых данных
                if len(data_old[bca]) > 0 and len(dict_data_new[bca]) > 0:
                    # Проверяем, не дублируется ли последняя дата
                    if data_old[bca].iloc[-1][0] == dict_data_new[bca].iloc[0][0]:
                        # Если даты совпадают, объединяем без дубликата
                        res = pd.concat([data_old[bca].iloc[:-1], dict_data_new[bca]], ignore_index=True)
                    else:
                        # Если даты разные, просто объединяем
                        res = pd.concat([data_old[bca], dict_data_new[bca]], ignore_index=True)
                    
                    # Сохраняем результат обратно в словарь
                    data_old[bca] = res
                    
        #Сохранение в файл
        File(filepath_by_months).to_file(data_old)

        #Придание внешнего вида итоговой таблице
        df_dict = File(filepath_by_months).from_file(0, 0)
        data = Dict_Operations(df_dict).replace_keys_in_dict(list_of_replacements = DataConfig.BCA_LIST)
        try:
            writer = pd.ExcelWriter(filepath_by_months, engine = 'xlsxwriter')
            for key, df in data.items():
                Table(df = df).make_style_of_table(writer = writer, 
                                                sheet_name = key, 
                                                width_col_1 = 4.5, 
                                                width_col_2 = 13.43, 
                                                width_col_3 = 13.0)
            writer.close()
            print("✅ Файл успешно сохранен")

        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")
        
        return data_old