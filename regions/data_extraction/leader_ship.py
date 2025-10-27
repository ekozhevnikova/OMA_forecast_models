import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import pymorphy3 as pmrph
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce


import sys
sys.path.append('C:/')
import OMA_tools
from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.io_data.dates import Dates_Operations

import os
import re
import json
import time
import openpyxl
from IPython.display import JSON


from mediascope_api.core import net as mscore
from mediascope_api.mediavortex import tasks as cwt
from mediascope_api.mediavortex import catalogs as cwc

# Настраиваем отображение

# Включаем отображение всех колонокv
pd.set_option('display.max_columns', None)

# Cоздаем объекты для работы с TVI API
mnet = mscore.MediascopeApiNetwork()
mtask = cwt.MediaVortexTask()
cats = cwc.MediaVortexCats()



#Класс, который мьютит все принты в консоли
class WrapperNoPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

        

class LeaderShipDataExtractor:
    """
        Класс для выгрузки данных для руководителей групп, а также для куратора регионов
    """
    @staticmethod
    def preprocess_constant_files(regions_id_path, table_sample_path, sheet_name_1 = 'for_comments', sheet_name_2 = 'for_team_lead'):
        """
            Функция для обработки файлов, в которых прописан порядок каналов, а также их ID
        """
        data = File(regions_id_path).from_file(skiprows = 0)
        data_ = Dict_Operations(data).replace_keys_in_dict(['All 18+', 'All 14-59', 'All 10-45', 'All 14-44', 'All 14-54', 
                                                            'All 25-49', 'All 25-54', 'All 4-45', 'All 6-54', 'W 14-44', 'W 25-59'])
        dict_data = Dict_Operations.generate_list_or_dict_from_df(dict_with_df = data_, col_name = 'ID', cond = 'dict')
        #Чтение заготовки для составления таблицы для генерации комментариев
        share_table = pd.read_excel(table_sample_path, sheet_name_1)
        #Чтение заготовки для составления таблицы для руководителей групп
        team_lead_table = pd.read_excel(table_sample_path, sheet_name_2)
        #Приведение к верхнему регистру значений в колонках
        columns = ['Channel', 'City', 'BCA']
        for i in columns:
            share_table[i] = share_table[i].str.upper()
        share_table['tvCompanyName'] = share_table['Channel'] + ' (' + share_table['City'] + ')'
        return dict_data, share_table, team_lead_table
    
    @staticmethod
    def get_output(date_filter, data, share_table, statistic = 'Доля'):
        """
            Вспомогательная функция для придания внешнего вида выходной таблицы
        """
        df_new = Table(data).delete_rows_with_substring(col_name = 'tvCompanyName', substring = ' / ДОП ')
        df_new = df_new.rename(columns = {'regionName': 'City'})
        split_df = Table(df_new).split_column(split = '   ', col_name = 'prj_name', col_name_new = ['City', 'BCA'])
        split_df['prj_name'] = split_df['City'] + '   ' + split_df['BCA']
        data_df_fact_month = pd.merge(split_df, df_new, on = ['City', 'prj_name']).drop_duplicates()
        data_output = pd.merge(share_table, data_df_fact_month, on = ['tvCompanyName', 'City', 'BCA'], how = 'left')
        output = data_output.drop(columns = ['prj_name', 'tvCompanyName'])
        output['Share'] = output['Share'].round(5)
        output_ = output.rename(columns = {
            'Channel': 'Телеканал', 
            'City': 'Город', 
            'BCA': 'БЦА', 
            'Share': Dates_Operations(date_filter).generate_date_interval(statistic)})
        return output_
    
    
    @staticmethod
    def sum_2_channels(df1, df2, date_column):
        """
            Args:
                df1: Локальный канал
                df2: Отфильтрованный канал из основной массы
            Return:
        """
        output_df1 = df1.rename(columns = {'regionName': 'Город'})[['Город', 'tvCompanyName', 'Share']]
        data = pd.merge(df2, output_df1, on = 'Город', how = 'left')[['Телеканал', 'Город', 'БЦА', date_column, 'Share']]
        data['SHARE'] = data[[date_column, 'Share']].sum(axis = 1)
        data_updated = data[['Телеканал', 'Город', 'БЦА', 'SHARE']]
        Data = data_updated.rename(columns = {'SHARE': date_column})
        return Data

    
    @staticmethod
    def make_output(df_1, df_2, full, share_table):
        broken_channels = pd.concat([df_1, df_2])
        data_updated = pd.concat([full, broken_channels])
        share_ = share_table.rename(columns  = {'Channel': 'Телеканал', 'City': 'Город', 'BCA': 'БЦА'})
        example_share = share_[list(share_.columns)[:-1]]
        output_data = pd.merge(example_share, data_updated, on = ['Телеканал', 'Город', 'БЦА'], how = 'left')
        return output_data
    
    
    @staticmethod
    def get_data(date_filter, company_filter, basedemo_filter, regions_id,
            time_filter = 'timeBand1 >= 50000 AND timeBand1 < 290000',
            statistics = ['Share'],
            slices = ['regionName', #регион
                      'tvCompanyName' #телесеть
                     ],
            sortings = {'tvCompanyName':'ASC'}, #Указываем сортировки'''
            options = {
                        "kitId": 3, #TV Index Cities  
                        "totalType": "TotalChannels" #база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
                      },
             location_filter = None, #Если None, то Дом и Дача
             weekday_filter = None, #Задаем дни недели
             daytype_filter = None, #Задаем тип дня
             targetdemo_filter = None, #Дополнительный фильтр на ЦА для расчета Affinity
             add_city_to_basedemo_from_region = True,
             add_city_to_targetdemo_from_region = True):
        tasks = []

        # Для каждого региона формируем задание и отправляем на расчет
        for reg_id, reg_name in regions_id.items():

            project_name = reg_name

            #Передаем id региона в company_filter:
            init_company_filter = company_filter

            if company_filter is not None:
                company_filter = company_filter + f' AND regionId IN ({reg_id})'

            else:
                company_filter = f'regionId IN ({reg_id})'

            # Формируем задание для API TV Index в формате JSON
            task_json = mtask.build_timeband_task(date_filter=date_filter, 
                                         weekday_filter=weekday_filter, daytype_filter=daytype_filter, 
                                         company_filter=company_filter, time_filter=time_filter, 
                                         basedemo_filter=basedemo_filter, targetdemo_filter=targetdemo_filter,
                                         location_filter=location_filter, slices=slices, sortings=sortings,
                                         statistics=statistics, options=options, 
                                         add_city_to_basedemo_from_region=True,
                                         add_city_to_targetdemo_from_region=True
                                        )

            # Для каждого этапа цикла формируем словарь с параметрами и отправленным заданием на расчет
            tsk = {}
            tsk['project_name'] = project_name
            tsk['task'] = mtask.send_timeband_task(task_json)
            tasks.append(tsk)
            time.sleep(3)
            company_filter = init_company_filter

        tsks = mtask.wait_task(tasks)

        # Получаем результат
        results = []
        for t in tasks:
            tsk = t['task'] 
            df_result = mtask.result2table(mtask.get_result(tsk), project_name = t['project_name'])        
            results.append(df_result)
        df = pd.concat(results)

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[['prj_name'] + slices + statistics]
        df['prj_name'] = df['prj_name'].str.upper()
        return df
    
    
    @staticmethod
    def by_months_parallel_main_part(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names):
        """
        Функция для расчета долей для руководителей групп. ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ
        """
        def get_data_wrapper(args):
            """
            Вспомогательная функция, принимающая кортеж аргументов
            """
            date_filt, company, basedemo, regions, name = args
            try:
                df = LeaderShipDataExtractor.get_data(date_filt, company, basedemo, regions)
                return name, df
            except Exception as e:
                print(f"Ошибка при получении данных для {name}: {e}")
                return name, pd.DataFrame()  # возвращаем пустой DataFrame в случае ошибки

        # Подготавливаем список аргументов
        args_list = [
            (date_filter, company_name_list[j], basedemo_filter_list[j], regions_dict_list[j], bca_list_names[j])
            for j in range(len(bca_list_names))
        ]

        # Используем ThreadPoolExecutor для параллельного выполнения
        with ThreadPoolExecutor(max_workers=min(10, len(args_list))) as executor:
            # Вариант 1: Используем map для простоты и безопасности
            results = list(executor.map(get_data_wrapper, args_list))

        # Обрабатываем результаты
        res = {}
        for name, df in results:
            if not df.empty:  # добавляем только непустые DataFrame
                res[name] = df

        # Объединяем все DataFrame
        if res:
            result_dfs = list(res.values())
            data = pd.concat(result_dfs, ignore_index=True)
        else:
            data = pd.DataFrame()

        return data


        
        
    @staticmethod
    def get_data_through_api_per_team_lead(date_filter,
                             share_table,
                             bca_list_names,
                             company_name_list,
                             company_filter_list_local_channels,
                             basedemo_filter_list, 
                             regions_dict_list,
                             company_gtrk,
                             vgtrk 
                                          ):
        """
            Функция для выгрузки данных через API.
        """
        ############################################################### Выгрузка основной массы каналов-городов ##################################################################################################
        res_tasks = LeaderShipDataExtractor.by_months_parallel_main_part(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names)
        
        ############################################################### Выгрузка для Телеканала 78 (Санкт-Петербург) и Санкт-Петербург(Санкт-Петербург) ###############################################################
        tasks_local_channels = []
        for i in range(len(company_filter_list_local_channels)):
            t = LeaderShipDataExtractor.get_data(date_filter, company_filter_list_local_channels[i], 'age >= 18', {2: 'САНКТ-ПЕТЕРБУРГ   ВСЕ 18+'})
            tasks_local_channels.append(t)   
        res_local_channels = pd.concat(tasks_local_channels , ignore_index = True)


        #Переименовывание Локальных каналов
        l = list(res_local_channels['tvCompanyName'])
        l_converted = []
        for i in range(len(l)):
            if l[i] == 'ТЕЛЕКАНАЛ 78 (САНКТ-ПЕТЕРБУРГ)':
                l_converted.append('ТЕЛЕКАНАЛ 78 САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)')
            if l[i] == 'САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)':
                l_converted.append('ТЕЛЕКАНАЛ САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)')
        res_local_channels['tvCompanyName'] = res_local_channels['tvCompanyName'].replace(l, l_converted)


        ############################################################### Выгрузка для Четвертый канал (Екатеринбург) ##################################################################################################
        channel_4 = {12: 'ЕКАТЕРИНБУРГ   ВСЕ 14-44'}
        task_4_channel = LeaderShipDataExtractor.get_data(date_filter, 'tvCompanyId = 4654', 'age >= 14 AND age <= 44', channel_4)

        
        ############################################################################ Выгрузка для ГТРК ################################################################################################################
        from concurrent.futures import ThreadPoolExecutor

        date_filter = date_filter
        def process_item(i):
            return LeaderShipDataExtractor.get_data(date_filter, company_gtrk[i], 'age >= 18', vgtrk[i])

        # Параллельное выполнение
        with ThreadPoolExecutor() as executor:
            tasks_vgtrk = list(executor.map(process_item, range(len(vgtrk))))
        res_vgtrk = pd.concat(tasks_vgtrk, ignore_index=True)

        #Объединение результатов между собой    
        df_fact_month = pd.concat([res_tasks, res_local_channels], ignore_index = True)
        output_fact_month = LeaderShipDataExtractor.get_output(date_filter, df_fact_month, share_table)
        #Название колонки с временным периодом
        date_column = output_fact_month.columns[3]
        
        #Отбор России 1
        filtered_df = output_fact_month[(output_fact_month['Телеканал'] == 'РОССИЯ 1')]
        senza_russia_1 = output_fact_month[(output_fact_month['Телеканал'] != 'РОССИЯ 1')]

        #Суммирование значений долей по каналу Пятница Екатеринбург
        output_fact_month_ = output_fact_month.copy()
        output_fact_month_['Теканал+Город+БЦА']= output_fact_month_['Телеканал']+ ' ' + output_fact_month_['Город'] + ' '+ output_fact_month_['БЦА']
        senza_pyatniza_ekb = output_fact_month_[(output_fact_month_['Теканал+Город+БЦА'] != 'ПЯТНИЦА ЕКАТЕРИНБУРГ ВСЕ 14-44')][list(output_fact_month_.columns)[:-1]]
        filtered_pyatniza = output_fact_month_[(output_fact_month_['Теканал+Город+БЦА'] == 'ПЯТНИЦА ЕКАТЕРИНБУРГ ВСЕ 14-44')][list(output_fact_month_.columns)[:-1]]

        filtered_full = output_fact_month_[(output_fact_month_['Теканал+Город+БЦА'] != 'ПЯТНИЦА ЕКАТЕРИНБУРГ ВСЕ 14-44') & (output_fact_month_['Телеканал'] != 'РОССИЯ 1')][list(output_fact_month_.columns)[:-1]]
        Russia_1 = LeaderShipDataExtractor.sum_2_channels(res_vgtrk, filtered_df, date_column)
        Channel_4 = LeaderShipDataExtractor.sum_2_channels(task_4_channel, filtered_pyatniza, date_column)
        output_data = LeaderShipDataExtractor.make_output(Russia_1, Channel_4, filtered_full, share_table)
        return output_data