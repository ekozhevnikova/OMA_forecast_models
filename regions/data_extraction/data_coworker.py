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
from OMA_tools.regions.data_extraction.leader_ship import WrapperNoPrints

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



class TeamDataExportService:
    """
        Класс для выгрузки данных для сотрудников
    """
    @staticmethod
    def get_output(df):
        df = df[df.columns.drop(list(df.filter(regex = ' / ДОП ')))]
        df = df.rename(columns = {'regionName': 'City'})
        result_df = pd.pivot_table(df, values = 'Share',
                                index = None, 
                                columns = ['tvCompanyName'])
        result_df = result_df[result_df.columns.drop(list(result_df.filter(regex = ' / ДОП ')))]
        return result_df
    
    
    @staticmethod
    def sum_2_channels(df_channel_1, df_channel_2, regions: list):
        """
            Функция для суммирования значений по двум каналам. Данная функция нужна для расчета доли по Р1 (Р1+ГТРК), Пятница ЕКБ (Пятница + 4 канал)
            Args:
                df_channel_1: DataFrame с городом-каналом, к которому будем плюсовать значения
                df_channel_2: DataFrame с городом-каналом, значения которого будет прибавлять к df_channel_1
                regions: список из регионов, для которых необходимо произвести суммирование
            Return:
                merged_channels: DataFrame с просуммированными значениями долей. Порядок каналов как в df_channel_1
        """
        df_channel_1['Date'] = pd.to_datetime(df_channel_1['Date'])
        df_channel_2['Date'] = pd.to_datetime(df_channel_2['Date'])
        full = []
        for i in range(len(regions)):
            data_1 = df_channel_1[['Date', list(df_channel_1.loc[:, df_channel_1.columns.str.contains(regions[i])].columns)[0]]]
            data_2 = df_channel_2[['Date', list(df_channel_2.loc[:, df_channel_2.columns.str.contains(regions[i])].columns)[0]]]
            data = pd.merge(data_1, data_2, on = 'Date', how = 'outer')
            data['total'] = data[[data.columns[1], data.columns[2]]].astype(float).sum(1)
            data_ = data[['Date', 'total']]
            data_final = data_.rename(columns = {'total': df_channel_1.columns[i+1]})
            full.append(data_final)
        merged_channels = reduce(lambda left, right: pd.merge(left, right, on = 'Date', how = 'outer'), full)
        return merged_channels
    
    
    @staticmethod
    def get_data_by_dates(date_filter, company_filter, basedemo_filter, regions_id, flag = True,
            time_filter = 'timeBand1 >= 50000 AND timeBand1 < 290000',
            statistics = ['Share'],
            slices = ['regionName', #регион
                      'tvCompanyName',
                      'researchDate'],
            sortings = {'tvCompanyName':'ASC', 'researchDate': 'ASC'},
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
            tsk['task'] = mtask.send_timeband_task(task_json)
            tasks.append(tsk)
            time.sleep(3)


            company_filter = init_company_filter


        tsks = mtask.wait_task(tasks)

        # Получаем результат
        results = []
        for t in tasks:
            tsk = t['task'] 
            df_result = mtask.result2table(mtask.get_result(tsk))        
            results.append(df_result)
        df = pd.concat(results)

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[slices + statistics]
        df_= pd.pivot_table(df, values = statistics,
                            index = ['researchDate'], 
                            columns = ['tvCompanyName'])
        if flag == True:
            df_new = Table(df_).make_table(column_name = 'Date')
            df_new = df_new[df_new.columns.drop(list(df_new.filter(regex = ' ДОП ')))]
            return df_new
        else:
            return df_

    
    @staticmethod
    def get_data_by_months(date_filter, company_filter, basedemo_filter, regions_id, flag = True,
            time_filter = 'timeBand1 >= 50000 AND timeBand1 < 290000',
            statistics = ['Share'],
            slices = ['regionName', #регион
                      'tvCompanyName',
                      'researchMonth'],
            sortings = {'tvCompanyName':'ASC', 'researchMonth': 'ASC'},
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

            #project_name = reg_name

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
            #tsk['project_name'] = project_name
            tsk['task'] = mtask.send_timeband_task(task_json)
            tasks.append(tsk)
            time.sleep(3)

            company_filter = init_company_filter

        # Ждем выполнения
        tsks = mtask.wait_task(tasks)
        # Получаем результат
        results = []
        for t in tasks:
            tsk = t['task'] 
            df_result = mtask.result2table(mtask.get_result(tsk))        
            results.append(df_result)
        df = pd.concat(results)

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[slices + statistics]
        #df['prj_name'] = df['prj_name'].str.upper()
        df_= pd.pivot_table(df, values = statistics,
                            index = ['researchMonth'], 
                            columns = ['tvCompanyName'])
        if flag == True:
            table = Table(df_)
            df_new = table.make_table(column_name = 'Date')
            df_new['Date'] = df_new['Date'].apply(lambda x: pd.to_datetime(x))
            df_new = df_new[df_new.columns.drop(list(df_new.filter(regex = ' ДОП ')))]
            return df_new
        else:
            return df_

    
    
    @staticmethod
    def get_data_per_fact_month(date_filter, company_filter, basedemo_filter, regions_id, flag = True,
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
        #print("Отправляем задания на расчет")

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
        #print('Собираем таблицу')
        for t in tasks:
            tsk = t['task'] 
            df_result = mtask.result2table(mtask.get_result(tsk), project_name = t['project_name'])
            results.append(df_result)
            print('.', end = '')
        df = pd.concat(results)
        
        if flag == True:
            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[['prj_name'] + slices + statistics]
            df['prj_name'] = df['prj_name'].str.upper()
            df_new = TeamDataExportService.get_output(df)
            df_new = df_new.reset_index()
            df_new = df_new.rename(columns = {df_new.columns[0]: 'Date'})
            df_new_ = df_new.replace(list(df_new['Date']), date_filter[0][0])
            df_new_['Date'] = pd.to_datetime(df_new_['Date'])
            return df_new_
        else:
            df = df[slices + statistics]
            result_df = pd.pivot_table(df, values = statistics,
                                index = None, 
                                columns = ['tvCompanyName'])
            result_df['Date'] = date_filter[0][0]
            result_df['Date'] = pd.to_datetime(result_df['Date'])
            return result_df

    
    @staticmethod
    def parallel_main_part_by_dates(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names):
        """
            Функция для расчета долей для руководителей групп. ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ
        """
        def get_data_wrapper(date_filter, company, basedemo, regions, name):
            """
                Вспомогательная функция
            """
            df = TeamDataExportService.get_data_by_dates(date_filter, company, basedemo, regions)
            return name, df
        
        res = {}
        args_list = [
            (date_filter, company_name_list[j], basedemo_filter_list[j], regions_dict_list[j], bca_list_names[j])
            for j in range(len(bca_list_names))
        ]

        with ThreadPoolExecutor(max_workers = 10) as executor:
            future_to_name = {
                executor.submit(get_data_wrapper, date_filter, company, basedemo, regions, name): name
                for date_filter, company, basedemo, regions, name in args_list
            }

            for future in as_completed(future_to_name):
                name, df = future.result()
                res[name] = df
        return res
    
    
    @staticmethod
    def parallel_main_part_by_months(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names):
        """
            Функция для расчета долей для руководителей групп. ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ
        """
        
        def get_data_wrapper(date_filter, company, basedemo, regions, name):
            """
                Вспомогательная функция
            """
            df = TeamDataExportService.get_data_by_months(date_filter, company, basedemo, regions)
            return name, df
        
        res = {}
        args_list = [
            (date_filter, company_name_list[j], basedemo_filter_list[j], regions_dict_list[j], bca_list_names[j])
            for j in range(len(bca_list_names))
        ]

        with ThreadPoolExecutor(max_workers = 10) as executor:
            future_to_name = {
                executor.submit(get_data_wrapper, date_filter, company, basedemo, regions, name): name
                for date_filter, company, basedemo, regions, name in args_list
            }

            for future in as_completed(future_to_name):
                name, df = future.result()
                res[name] = df
        return res
    
    
    @staticmethod
    def parallel_main_part_fact(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names):
        """
            Функция для расчета долей для руководителей групп. ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ
        """
        
        def get_data_wrapper(date_filter, company, basedemo, regions, name):
            """
                Вспомогательная функция
            """
            df = TeamDataExportService.get_data_per_fact_month(date_filter, company, basedemo, regions)
            return name, df
        
        res = {}
        args_list = [
            (date_filter, company_name_list[j], basedemo_filter_list[j], regions_dict_list[j], bca_list_names[j])
            for j in range(len(bca_list_names))
        ]

        with ThreadPoolExecutor(max_workers = 10) as executor:
            future_to_name = {
                executor.submit(get_data_wrapper, date_filter, company, basedemo, regions, name): name
                for date_filter, company, basedemo, regions, name in args_list
            }

            for future in as_completed(future_to_name):
                name, df = future.result()
                res[name] = df
        return res
    
    
    @staticmethod
    def get_data_through_api_per_employees_by_days(date_filter: list,
                                                   bca_list_names: list,
                                                   company_name_list: list,
                                                   company_filter_list_local_channels: list,
                                                   basedemo_filter_list: list, 
                                                   regions_dict_list: list,
                                                   vgtrk: list,
                                                   company_gtrk: list,
                                                   columns_order: str,
                                                   filepath_by_days: str,
                                                   LAST_N_DAYS: int = 28,
                                                   date_column: str = 'Date'
                                                  ):
        """
            Функция для выгрузки данных через API.
            Args:
                date_filter: временной период для выгрузки
                bca_list_names: список из БЦА
                company_name_list: список из телекомпаний
                company_filter_list_local_channels: список из локальных телекомпаний
                basedemo_filter_list: список из БЦА
                regions_dict_list: список из словарей регионов и их ID
                columns_order: str: путь к файлу .pkl с порядком столбцов
                filepath_by_days: str: путь к файлу, куда будем сохранять выгрузку
                LAST_N_DAYS = 28: критерий на то, сколько последних N дней будем сохранять в выходной файл
                date_column = 'Date': название колонки с датой
            Return:
                df_dict_tail: причесанный словарь из DataFrameов.
        """
        loaded_dict_columns = Dict_Operations.load_pkl_file(columns_order)
        
        result = TeamDataExportService.parallel_main_part_by_dates(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names)
        
        #Выделение России 1
        result['All 18+']['Date'] = pd.to_datetime(result['All 18+']['Date'])
        Russia_1 = result['All 18+'][['Date', 'РОССИЯ 1 (БАРНАУЛ)', 'РОССИЯ 1 (ВЛАДИВОСТОК)', 'РОССИЯ 1 (ВОЛГОГРАД)',  'РОССИЯ 1 (ВОРОНЕЖ)',
                                     'РОССИЯ 1 (ЕКАТЕРИНБУРГ)',  'РОССИЯ 1 (ИРКУТСК)',  'РОССИЯ 1 (КАЗАНЬ)',  'РОССИЯ 1 (КЕМЕРОВО)',
                                     'РОССИЯ 1 (КРАСНОДАР)',  'РОССИЯ 1 (КРАСНОЯРСК)',  'РОССИЯ 1 (НИЖНИЙ НОВГОРОД)',  'РОССИЯ 1 (НОВОСИБИРСК)',
                                     'РОССИЯ 1 (ОМСК)',  'РОССИЯ 1 (ПЕРМЬ)',  'РОССИЯ 1 (РОСТОВ-НА-ДОНУ)',  'РОССИЯ 1 (САМАРА)',
                                     'РОССИЯ 1 (САНКТ-ПЕТЕРБУРГ)',  'РОССИЯ 1 (САРАТОВ)',  'РОССИЯ 1 (ТЮМЕНЬ)',  'РОССИЯ 1 (УФА)',
                                     'РОССИЯ 1 (ХАБАРОВСК)',  'РОССИЯ 1 (ЧЕЛЯБИНСК)',  'РОССИЯ 1 (ЯРОСЛАВЛЬ)',  'РОССИЯ 1 (СТАВРОПОЛЬ)',
                                     'РОССИЯ 1 (ТВЕРЬ)', 'РОССИЯ 1 (ТОМСК)']]

        #Удаление России 1 из Датафрейма
        data_all_18_senza_russia_1 = result['All 18+'].drop([col for col in result['All 18+'].columns if 'РОССИЯ 1' in col], axis = 1)
        
        
        result['All 14-44']['Date'] = pd.to_datetime(result['All 14-44']['Date'])
        #Выделение Пятница (Екатеринбург)
        pyatniza = result['All 14-44'][['Date', 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)']]
        #Удаление Пятница (Екатеринбург) из Датафрейма
        data_all_14_44_senza_pyatniza_ekb = result['All 14-44'].drop([col for col in result['All 14-44'].columns if 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)' in col], axis = 1)

        #Выгрузка для Телеканала 78 (Санкт-Петербург) и Санкт-Петербург(Санкт-Петербург)    
        tasks_local_channels = []
        for i in range(len(company_filter_list_local_channels)):
            t = TeamDataExportService.get_data_by_dates(date_filter, 
                                                        company_filter_list_local_channels[i], 
                                                        'age >= 18', 
                                                        {2: 'САНКТ-ПЕТЕРБУРГ   ВСЕ 18+'},
                                                        flag = False)
            t_new = Table(t).make_table(column_name = 'Date')
            t_new['Date'] = pd.to_datetime(t_new['Date'])
            tasks_local_channels.append(t_new)

        
        #Выгрузка каналов ГТРК
        from concurrent.futures import ThreadPoolExecutor

        def process_single_company(args):
            i, date_filter, company, age_filter, vgtrk_data = args
            t = TeamDataExportService.get_data_by_dates(date_filter, company, 'age >= 18', vgtrk_data, flag = False)
            t_new = Table(t).make_table(column_name='Date')
            t_new['Date'] = pd.to_datetime(t_new['Date'])
            column = t_new.columns[1]
            column_name = f"ГТРК {list(vgtrk_data.values())[0]}"
            return t_new.rename(columns={column: column_name})

        # Параллельная обработка
        def optimized_version(vgtrk, company_gtrk, date_filter):
            with ThreadPoolExecutor() as executor:
                args = [(i, date_filter, company_gtrk[i], 'age >= 18', vgtrk[i]) 
                        for i in range(len(vgtrk))]
                tasks_vgtrk = list(executor.map(process_single_company, args))

            # Более эффективное объединение
            merged_vgtrk = tasks_vgtrk[0]
            for df in tasks_vgtrk[1:]:
                merged_vgtrk = merged_vgtrk.merge(df, on='Date', how='outer')
            return merged_vgtrk
        merged_vgtrk = optimized_version(vgtrk, company_gtrk, date_filter)
        merged_vgtrk['Date'] = pd.to_datetime(merged_vgtrk['Date'])
        full_vgtrk = merged_vgtrk[['Date', 'ГТРК БАРНАУЛ   ВСЕ 18+', 'ГТРК ВЛАДИВОСТОК   ВСЕ 18+',
                                        'ГТРК ВОЛГОГРАД   ВСЕ 18+', 'ГТРК ВОРОНЕЖ   ВСЕ 18+', 'ГТРК ЕКАТЕРИНБУРГ   ВСЕ 18+',
                                        'ГТРК ИРКУТСК   ВСЕ 18+', 'ГТРК КАЗАНЬ   ВСЕ 18+', 'ГТРК КЕМЕРОВО   ВСЕ 18+',
                                        'ГТРК КРАСНОДАР   ВСЕ 18+', 'ГТРК КРАСНОЯРСК   ВСЕ 18+', 'ГТРК НИЖНИЙ НОВГОРОД   ВСЕ 18+',
                                        'ГТРК НОВОСИБИРСК   ВСЕ 18+', 'ГТРК ОМСК   ВСЕ 18+', 'ГТРК ПЕРМЬ   ВСЕ 18+',
                                        'ГТРК РОСТОВ-НА-ДОНУ   ВСЕ 18+', 'ГТРК САМАРА   ВСЕ 18+', 'ГТРК САНКТ-ПЕТЕРБУРГ   ВСЕ 18+',
                                        'ГТРК САРАТОВ   ВСЕ 18+', 'ГТРК ТЮМЕНЬ   ВСЕ 18+', 'ГТРК УФА   ВСЕ 18+',
                                        'ГТРК ХАБАРОВСК   ВСЕ 18+', 'ГТРК ЧЕЛЯБИНСК  ВСЕ 18+', 'ГТРК ЯРОСЛАВЛЬ   ВСЕ 18+',
                                        'ГТРК СТАВРОПОЛЬ   ВСЕ 18+', 'ГТРК ТВЕРЬ   ВСЕ 18+', 'ГТРК ТОМСК   ВСЕ 18+']]

        
        #Выгрузка 4 канала (Екатеринбург)
        channel_4 = {12: 'ЕКАТЕРИНБУРГ   ВСЕ 14-44'}
        task_4_channel = TeamDataExportService.get_data_by_dates(date_filter, 'tvCompanyId = 4654', 'age >= 14 AND age <= 44', channel_4, flag = False)
        t = Table(task_4_channel).make_table(column_name = 'Date')
        t['Date'] = pd.to_datetime(t['Date'])
        column = t.columns[1]
        column_name = 'ЧЕТВЕРТЫЙ КАНАЛ ' + str(list(channel_4.values())[0])
        df = t.rename(columns = {f'{column}': f'{column_name}'})
        df_channel_4 = df[['Date', 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44']]
        
        
        #Суммирование значений по России 1 и ГТРК
        Russia_1_updated = TeamDataExportService.sum_2_channels(Russia_1, 
                                  full_vgtrk, 
                                  ['БАРНАУЛ', 'ВЛАДИВОСТОК', 'ВОЛГОГРАД', 'ВОРОНЕЖ', 
                                    'ЕКАТЕРИНБУРГ', 'ИРКУТСК', 'КАЗАНЬ', 'КЕМЕРОВО', 
                                    'КРАСНОДАР', 'КРАСНОЯРСК', 'НИЖНИЙ НОВГОРОД', 'НОВОСИБИРСК', 
                                    'ОМСК', 'ПЕРМЬ', 'РОСТОВ-НА-ДОНУ', 'САМАРА', 
                                    'САНКТ-ПЕТЕРБУРГ', 'САРАТОВ', 'ТЮМЕНЬ',
                                    'УФА', 'ХАБАРОВСК', 'ЧЕЛЯБИНСК', 'ЯРОСЛАВЛЬ', 
                                   'СТАВРОПОЛЬ', 'ТВЕРЬ', 'ТОМСК'
                                   ]                               
                                 )
        #Обновление датафрейма для Все 18+
        new_data_all_18 = pd.merge(data_all_18_senza_russia_1, Russia_1_updated, how = 'left', on = 'Date')
        
        result['All 18+'] = Table.make_left_join(new_data_all_18, tasks_local_channels[0], tasks_local_channels[1], 'Date')
        result['All 18+'] = result['All 18+'][loaded_dict_columns['All 18+']]
        
        
        #Суммирование значений долей Пятница(Екатеринбург) с Четвертым каналом (Екатеринбург)
        pyatniza_updated = TeamDataExportService.sum_2_channels(pyatniza, df_channel_4, ['ЕКАТЕРИНБУРГ'])
        #Обновление датафрейма для Все 14-44
        
        new_data_all_14_44 = pd.merge(data_all_14_44_senza_pyatniza_ekb, pyatniza_updated, how = 'left', on = 'Date')
        result['All 14-44'] = new_data_all_14_44[loaded_dict_columns['All 14-44']]
        
        #Добавление локальных каналов к аудитории Все 18+
        #dict_data['All 18+'] = Table.make_left_join(dict_data['All 18+'], tasks_local_channels[0], tasks_local_channels[1], 'Date')
        dict_data = {'All 18+': result['All 18+'], 
                 'All 14-59': result['All 14-59'], 
                 'All 10-45': result['All 10-45'], 
                 'All 14-44': result['All 14-44'], 
                 'All 14-54': result['All 14-54'], 
                 'All 25-49': result['All 25-49'], 
                 'All 25-54': result['All 25-54'], 
                 'All 4-45': result['All 4-45'], 
                 'All 6-54': result['All 6-54'], 
                 'W 14-44': result['W 14-44'], 
                 'W 25-59': result['W 25-59']}
        
        df_dict = Dict_Operations(dict_data).rename_columns_in_dict_with_df(loaded_dict_columns)
        for key, df in df_dict.items():
            df_dict[key][date_column] = df_dict[key][date_column].apply(lambda x: pd.to_datetime(x))
        data_old = File(filepath_by_days).from_file(0)
        data_new = File(filename = filepath_by_days).update_file(dataframe = df_dict,
                                                                              column_name = date_column, 
                                                                              list_of_replacements = ['All 18+', 'All 14-59', 'All 10-45', 'All 14-44', 
                                                                                                      'All 14-54', 'All 25-49', 'All 25-54', 'All 4-45', 
                                                                                                      'All 6-54', 'W 14-44', 'W 25-59']
                                                                              )
        df_dict_tail = {}
        for bca, df in df_dict.items():
            if len(list(df_dict['All 18+'])) < LAST_N_DAYS:
                raise ValueError('Количество выгружаемых дней не соответствует количеству дней, записываемых в файл. Выберите другой временной период')
            df_dict_tail[bca] = df.tail(LAST_N_DAYS) #Записывает последние 30 значений из выгрузки
        return data_new, df_dict_tail
        
    
    
    
    @staticmethod
    def get_data_through_api_per_employees_by_months(date_filter: list,
                                                   bca_list_names: list,
                                                   company_name_list: list,
                                                   company_filter_list_local_channels: list,
                                                   basedemo_filter_list: list, 
                                                   regions_dict_list: list,
                                                   vgtrk: list,
                                                   company_gtrk: list,
                                                   columns_order: str,
                                                   date_column: str = 'Date'
                                                  ):
        """
            Функция для выгрузки данных через API.
            Args:
                date_filter: временной период для выгрузки
                bca_list_names: список из БЦА
                company_name_list: список из телекомпаний
                company_filter_list_local_channels: список из локальных телекомпаний
                basedemo_filter_list: список из БЦА
                regions_dict_list: список из словарей регионов и их ID
                columns_order: str: путь к файлу .pkl с порядком столбцов
                filepath_by_days: str: путь к файлу, куда будем сохранять выгрузку
                LAST_N_DAYS = 28: критерий на то, сколько последних N дней будем сохранять в выходной файл
                date_column = 'Date': название колонки с датой
            Return:
                df_dict_tail: причесанный словарь из DataFrameов.
        """
        loaded_dict_columns = Dict_Operations.load_pkl_file(columns_order)
        
        ###################################################################################### Выгрузка основной массы каналов-городов ######################################################################################
        result = TeamDataExportService.parallel_main_part_by_months(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names)
        
        #Выделение России 1
        Russia_1 = result['All 18+'][['Date', 'РОССИЯ 1 (БАРНАУЛ)', 'РОССИЯ 1 (ВЛАДИВОСТОК)', 'РОССИЯ 1 (ВОЛГОГРАД)',  'РОССИЯ 1 (ВОРОНЕЖ)',
                                     'РОССИЯ 1 (ЕКАТЕРИНБУРГ)',  'РОССИЯ 1 (ИРКУТСК)',  'РОССИЯ 1 (КАЗАНЬ)',  'РОССИЯ 1 (КЕМЕРОВО)',
                                     'РОССИЯ 1 (КРАСНОДАР)',  'РОССИЯ 1 (КРАСНОЯРСК)',  'РОССИЯ 1 (НИЖНИЙ НОВГОРОД)',  'РОССИЯ 1 (НОВОСИБИРСК)',
                                     'РОССИЯ 1 (ОМСК)',  'РОССИЯ 1 (ПЕРМЬ)',  'РОССИЯ 1 (РОСТОВ-НА-ДОНУ)',  'РОССИЯ 1 (САМАРА)',
                                     'РОССИЯ 1 (САНКТ-ПЕТЕРБУРГ)',  'РОССИЯ 1 (САРАТОВ)',  'РОССИЯ 1 (ТЮМЕНЬ)',  'РОССИЯ 1 (УФА)',
                                     'РОССИЯ 1 (ХАБАРОВСК)',  'РОССИЯ 1 (ЧЕЛЯБИНСК)',  'РОССИЯ 1 (ЯРОСЛАВЛЬ)',  'РОССИЯ 1 (СТАВРОПОЛЬ)',
                                     'РОССИЯ 1 (ТВЕРЬ)', 'РОССИЯ 1 (ТОМСК)']]
        
        #Удаление России 1 из Датафрейма
        data_all_18_senza_russia_1 = result['All 18+'].drop([col for col in result['All 18+'].columns if 'РОССИЯ 1' in col], axis = 1)
        
        

        #Выделение Пятница (Екатеринбург)
        pyatniza = result['All 14-44'][['Date', 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)']]
        #Удаление Пятница (Екатеринбург) из Датафрейма
        data_all_14_44_senza_pyatniza_ekb = result['All 14-44'].drop([col for col in result['All 14-44'].columns if 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)' in col], axis = 1)


        ###################################################################################### Выгрузка для Телеканала 78 (Санкт-Петербург) и Санкт-Петербург(Санкт-Петербург) ######################################################################################
        tasks_local_channels = []
        for i in range(len(company_filter_list_local_channels)):
            t = TeamDataExportService.get_data_by_months(date_filter = date_filter, company_filter = company_filter_list_local_channels[i], basedemo_filter = 'age >= 18', regions_id = {2: 'САНКТ-ПЕТЕРБУРГ   ВСЕ 18+'}, flag = False)
            t_new = Table(t).make_table(column_name = 'Date')
            t_new['Date'] = t_new['Date'].apply(lambda x: pd.to_datetime(x))
            tasks_local_channels.append(t_new)

    
        ####################################################################################### Выгрузка каналов ГТРК ######################################################################################
        from concurrent.futures import ThreadPoolExecutor

        def process_single_company(args):
            i, date_filter, company, age_filter, vgtrk_data = args
            t = TeamDataExportService.get_data_by_months(date_filter = date_filter, company_filter = company, basedemo_filter = 'age >= 18', regions_id = vgtrk_data, flag = False)
            t_new = Table(t).make_table(column_name='Date')
            column = t_new.columns[1]
            column_name = f"ГТРК {list(vgtrk_data.values())[0]}"
            return t_new.rename(columns={column: column_name})

        # Параллельная обработка
        def optimized_version(vgtrk, company_gtrk, date_filter):
            with ThreadPoolExecutor() as executor:
                args = [(i, date_filter, company_gtrk[i], 'age >= 18', vgtrk[i]) 
                        for i in range(len(vgtrk))]
                tasks_vgtrk = list(executor.map(process_single_company, args))

            # Более эффективное объединение
            merged_vgtrk = tasks_vgtrk[0]
            for df in tasks_vgtrk[1:]:
                merged_vgtrk = merged_vgtrk.merge(df, on='Date', how='outer')
            return merged_vgtrk
        merged_vgtrk = optimized_version(vgtrk, company_gtrk, date_filter)
        full_vgtrk = merged_vgtrk[['Date', 'ГТРК БАРНАУЛ   ВСЕ 18+', 'ГТРК ВЛАДИВОСТОК   ВСЕ 18+',
                                        'ГТРК ВОЛГОГРАД   ВСЕ 18+', 'ГТРК ВОРОНЕЖ   ВСЕ 18+', 'ГТРК ЕКАТЕРИНБУРГ   ВСЕ 18+',
                                        'ГТРК ИРКУТСК   ВСЕ 18+', 'ГТРК КАЗАНЬ   ВСЕ 18+', 'ГТРК КЕМЕРОВО   ВСЕ 18+',
                                        'ГТРК КРАСНОДАР   ВСЕ 18+', 'ГТРК КРАСНОЯРСК   ВСЕ 18+', 'ГТРК НИЖНИЙ НОВГОРОД   ВСЕ 18+',
                                        'ГТРК НОВОСИБИРСК   ВСЕ 18+', 'ГТРК ОМСК   ВСЕ 18+', 'ГТРК ПЕРМЬ   ВСЕ 18+',
                                        'ГТРК РОСТОВ-НА-ДОНУ   ВСЕ 18+', 'ГТРК САМАРА   ВСЕ 18+', 'ГТРК САНКТ-ПЕТЕРБУРГ   ВСЕ 18+',
                                        'ГТРК САРАТОВ   ВСЕ 18+', 'ГТРК ТЮМЕНЬ   ВСЕ 18+', 'ГТРК УФА   ВСЕ 18+',
                                        'ГТРК ХАБАРОВСК   ВСЕ 18+', 'ГТРК ЧЕЛЯБИНСК  ВСЕ 18+', 'ГТРК ЯРОСЛАВЛЬ   ВСЕ 18+',
                                        'ГТРК СТАВРОПОЛЬ   ВСЕ 18+', 'ГТРК ТВЕРЬ   ВСЕ 18+', 'ГТРК ТОМСК   ВСЕ 18+']]
        
        
        ###################################################################################### Выгрузка 4 канала (Екатеринбург) ######################################################################################
        channel_4 = {12: 'ЕКАТЕРИНБУРГ   ВСЕ 14-44'}
        task_4_channel = TeamDataExportService.get_data_by_months(date_filter = date_filter, company_filter = 'tvCompanyId = 4654', basedemo_filter = 'age >= 14 AND age <= 44', regions_id = channel_4, flag = False)
        t = Table(task_4_channel).make_table(column_name = 'Date')
        column = t.columns[1]
        column_name = 'ЧЕТВЕРТЫЙ КАНАЛ ' + str(list(channel_4.values())[0])
        df = t.rename(columns = {f'{column}': f'{column_name}'})
        df_channel_4 = df[['Date', 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44']]
        
        #Суммирование значений по России 1 и ГТРК
        Russia_1_updated = TeamDataExportService.sum_2_channels(Russia_1, 
                                  full_vgtrk, 
                                  ['БАРНАУЛ', 'ВЛАДИВОСТОК', 'ВОЛГОГРАД', 'ВОРОНЕЖ', 
                                    'ЕКАТЕРИНБУРГ', 'ИРКУТСК', 'КАЗАНЬ', 'КЕМЕРОВО', 
                                    'КРАСНОДАР', 'КРАСНОЯРСК', 'НИЖНИЙ НОВГОРОД', 'НОВОСИБИРСК', 
                                    'ОМСК', 'ПЕРМЬ', 'РОСТОВ-НА-ДОНУ', 'САМАРА', 
                                    'САНКТ-ПЕТЕРБУРГ', 'САРАТОВ', 'ТЮМЕНЬ',
                                    'УФА', 'ХАБАРОВСК', 'ЧЕЛЯБИНСК', 'ЯРОСЛАВЛЬ', 
                                   'СТАВРОПОЛЬ', 'ТВЕРЬ', 'ТОМСК'
                                   ]                                 
                                 )
        #Обновление датафрейма для Все 18+
        new_data_all_18 = pd.merge(data_all_18_senza_russia_1, Russia_1_updated, how = 'left', on = 'Date')
        new_data_all_18['Date'] = pd.to_datetime(new_data_all_18['Date'])
        result['All 18+'] = Table.make_left_join(new_data_all_18, tasks_local_channels[0], tasks_local_channels[1], 'Date')
        result['All 18+'] = result['All 18+'][loaded_dict_columns['All 18+']]
        
        
        #Суммирование значений долей Пятница(Екатеринбург) с Четвертым каналом (Екатеринбург)
        pyatniza_updated = TeamDataExportService.sum_2_channels(pyatniza, df_channel_4, ['ЕКАТЕРИНБУРГ'])
        
        #Обновление датафрейма для Все 14-44
        new_data_all_14_44 = pd.merge(data_all_14_44_senza_pyatniza_ekb, pyatniza_updated, how = 'left', on = 'Date')
        result['All 14-44'] = new_data_all_14_44[loaded_dict_columns['All 14-44']]
        
        dict_data = {'All 18+': result['All 18+'], 
                 'All 14-59': result['All 14-59'], 
                 'All 10-45': result['All 10-45'], 
                 'All 14-44': result['All 14-44'], 
                 'All 14-54': result['All 14-54'], 
                 'All 25-49': result['All 25-49'], 
                 'All 25-54': result['All 25-54'], 
                 'All 4-45': result['All 4-45'], 
                 'All 6-54': result['All 6-54'], 
                 'W 14-44': result['W 14-44'], 
                 'W 25-59': result['W 25-59']}
        
        df_dict = Dict_Operations(dict_data).rename_columns_in_dict_with_df(loaded_dict_columns)
        
        #Добавление локальных каналов к аудитории Все 18+
        #dict_data['All 18+'] = Table.make_left_join(dict_data['All 18+'], tasks_local_channels[0], tasks_local_channels[1], 'Date')
        dict_data_new = Dict_Operations(df_dict).convert_column_with_date('Date')
        return dict_data_new
    
    
    
    @staticmethod
    def get_data_through_api_per_employees_fact(date_filter: list,
                                                bca_list_names: list,
                                                company_name_list: list,
                                                company_filter_list_local_channels: list,
                                                basedemo_filter_list: list, 
                                                regions_dict_list: list,
                                                vgtrk: list,
                                                company_gtrk: list,
                                                columns_order: str,
                                                date_column: str = 'Date'
                                                  ):
        """
            Функция для выгрузки данных через API.
            Args:
                date_filter: временной период для выгрузки
                bca_list_names: список из БЦА
                company_name_list: список из телекомпаний
                company_filter_list_local_channels: список из локальных телекомпаний
                basedemo_filter_list: список из БЦА
                regions_dict_list: список из словарей регионов и их ID
                columns_order: str: путь к файлу .pkl с порядком столбцов
                date_column = 'Date': название колонки с датой
            Return:
                df_dict_tail: причесанный словарь из DataFrameов.
        """
        loaded_dict_columns = Dict_Operations.load_pkl_file(columns_order)
        
        
        #################################################### Выгрузка основной массы каналов-городов ####################################################
        result = TeamDataExportService.parallel_main_part_fact(date_filter, company_name_list, basedemo_filter_list, regions_dict_list, bca_list_names)
                
        ###################################Выгрузка для Телеканала 78 (Санкт-Петербург) и Санкт-Петербург(Санкт-Петербург)##################################    
        
        tasks_local_channels = []
        for i in range(len(company_filter_list_local_channels)):
            t = TeamDataExportService.get_data_per_fact_month(date_filter = date_filter, company_filter = company_filter_list_local_channels[i], basedemo_filter = 'age >= 18', regions_id = {2: 'САНКТ-ПЕТЕРБУРГ   ВСЕ 18+'})
            tasks_local_channels.append(t)
        
        ###############################################Выгрузка каналов ГТРК###############################################
        from concurrent.futures import ThreadPoolExecutor

        def process_single_company(args):
            i, date_filter, company, age_filter, vgtrk_data = args
            t = TeamDataExportService.get_data_per_fact_month(date_filter = date_filter, company_filter = company, basedemo_filter = 'age >= 18', regions_id = vgtrk_data, flag = False)
            #print(t)

            # Вместо использования Table, обрабатываем данные напрямую
            if not t.empty:
                # Группируем по дате и считаем количество
                result = t.groupby('Date').size().reset_index(name=f"ГТРК {list(vgtrk_data.values())[0]}")
                return result
            else:
                # Возвращаем пустой DataFrame с правильной структурой
                return pd.DataFrame({'Date': [], f"ГТРК {list(vgtrk_data.values())[0]}": []})

        def optimized_version(vgtrk, company_gtrk, date_filter):
            with ThreadPoolExecutor() as executor:
                args = [(i, date_filter, company_gtrk[i], 'age >= 18', vgtrk[i])
                        for i in range(len(vgtrk))]
                tasks_vgtrk = list(executor.map(process_single_company, args))

            # Объединяем все DataFrame по колонке Date
            merged_vgtrk = tasks_vgtrk[0]
            for df in tasks_vgtrk[1:]:
                merged_vgtrk = pd.merge(merged_vgtrk, df, on='Date', how='outer')

            return merged_vgtrk.fillna(0)  # Заменяем NaN на 0

        merged_vgtrk = optimized_version(vgtrk, company_gtrk, date_filter)
        
        full_vgtrk = merged_vgtrk[['Date', 'ГТРК БАРНАУЛ   ВСЕ 18+', 'ГТРК ВЛАДИВОСТОК   ВСЕ 18+',
                                       'ГТРК ВОЛГОГРАД   ВСЕ 18+', 'ГТРК ВОРОНЕЖ   ВСЕ 18+', 'ГТРК ЕКАТЕРИНБУРГ   ВСЕ 18+',
                                       'ГТРК ИРКУТСК   ВСЕ 18+', 'ГТРК КАЗАНЬ   ВСЕ 18+', 'ГТРК КЕМЕРОВО   ВСЕ 18+',
                                       'ГТРК КРАСНОДАР   ВСЕ 18+', 'ГТРК КРАСНОЯРСК   ВСЕ 18+', 'ГТРК НИЖНИЙ НОВГОРОД   ВСЕ 18+',
                                       'ГТРК НОВОСИБИРСК   ВСЕ 18+', 'ГТРК ОМСК   ВСЕ 18+', 'ГТРК ПЕРМЬ   ВСЕ 18+',
                                       'ГТРК РОСТОВ-НА-ДОНУ   ВСЕ 18+', 'ГТРК САМАРА   ВСЕ 18+', 'ГТРК САНКТ-ПЕТЕРБУРГ   ВСЕ 18+',
                                       'ГТРК САРАТОВ   ВСЕ 18+', 'ГТРК ТЮМЕНЬ   ВСЕ 18+', 'ГТРК УФА   ВСЕ 18+',
                                       'ГТРК ХАБАРОВСК   ВСЕ 18+', 'ГТРК ЧЕЛЯБИНСК  ВСЕ 18+', 'ГТРК ЯРОСЛАВЛЬ   ВСЕ 18+',
                                       'ГТРК СТАВРОПОЛЬ   ВСЕ 18+', 'ГТРК ТВЕРЬ   ВСЕ 18+', 'ГТРК ТОМСК   ВСЕ 18+']]       
        ###############################################Выгрузка Четвертый канал Екатеринбург###############################################
        channel_4 = {12: 'ЕКАТЕРИНБУРГ   ВСЕ 14-44'}
        task_4_channel = TeamDataExportService.get_data_per_fact_month(date_filter = date_filter, company_filter = 'tvCompanyId = 4654', basedemo_filter = 'age >= 14 AND age <= 44', regions_id = channel_4, flag = False)
        #t = Table(task_4_channel).make_table(column_name = 'Date')
        column = task_4_channel.columns[0]
        column_name = 'ЧЕТВЕРТЫЙ КАНАЛ ' + str(list(channel_4.values())[0])
        df = task_4_channel.rename(columns = {f'{column}': f'{column_name}'})
        df_channel_4 = df[['Date', 'ЧЕТВЕРТЫЙ КАНАЛ ЕКАТЕРИНБУРГ   ВСЕ 14-44']]
        #Совмещение результатов между собой
        
        #Выделение России 1
        Russia_1 = result['All 18+'][['Date', 'РОССИЯ 1 (БАРНАУЛ)', 'РОССИЯ 1 (ВЛАДИВОСТОК)', 'РОССИЯ 1 (ВОЛГОГРАД)',  'РОССИЯ 1 (ВОРОНЕЖ)',
                                     'РОССИЯ 1 (ЕКАТЕРИНБУРГ)',  'РОССИЯ 1 (ИРКУТСК)',  'РОССИЯ 1 (КАЗАНЬ)',  'РОССИЯ 1 (КЕМЕРОВО)',
                                     'РОССИЯ 1 (КРАСНОДАР)',  'РОССИЯ 1 (КРАСНОЯРСК)',  'РОССИЯ 1 (НИЖНИЙ НОВГОРОД)',  'РОССИЯ 1 (НОВОСИБИРСК)',
                                     'РОССИЯ 1 (ОМСК)',  'РОССИЯ 1 (ПЕРМЬ)',  'РОССИЯ 1 (РОСТОВ-НА-ДОНУ)',  'РОССИЯ 1 (САМАРА)',
                                     'РОССИЯ 1 (САНКТ-ПЕТЕРБУРГ)',  'РОССИЯ 1 (САРАТОВ)',  'РОССИЯ 1 (ТЮМЕНЬ)',  'РОССИЯ 1 (УФА)',
                                     'РОССИЯ 1 (ХАБАРОВСК)',  'РОССИЯ 1 (ЧЕЛЯБИНСК)',  'РОССИЯ 1 (ЯРОСЛАВЛЬ)',  'РОССИЯ 1 (СТАВРОПОЛЬ)',
                                     'РОССИЯ 1 (ТВЕРЬ)', 'РОССИЯ 1 (ТОМСК)']]

        #Удаление России 1 из Датафрейма
        data_all_18_senza_russia_1 = result['All 18+'].drop([col for col in result['All 18+'].columns if 'РОССИЯ 1' in col], axis = 1)
        
        

        #Выделение Пятница (Екатеринбург)
        pyatniza = result['All 14-44'][['Date', 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)']]
        #Удаление Пятница (Екатеринбург) из Датафрейма
        data_all_14_44_senza_pyatniza_ekb = result['All 14-44'].drop([col for col in result['All 14-44'].columns if 'ПЯТНИЦА (ЕКАТЕРИНБУРГ)' in col], axis = 1)
        
        #Суммирование значений по России 1 и ГТРК
        Russia_1_updated = TeamDataExportService.sum_2_channels(Russia_1, 
                                  full_vgtrk, 
                                  ['БАРНАУЛ', 'ВЛАДИВОСТОК', 'ВОЛГОГРАД', 'ВОРОНЕЖ', 
                                    'ЕКАТЕРИНБУРГ', 'ИРКУТСК', 'КАЗАНЬ', 'КЕМЕРОВО', 
                                    'КРАСНОДАР', 'КРАСНОЯРСК', 'НИЖНИЙ НОВГОРОД', 'НОВОСИБИРСК', 
                                    'ОМСК', 'ПЕРМЬ', 'РОСТОВ-НА-ДОНУ', 'САМАРА', 
                                    'САНКТ-ПЕТЕРБУРГ', 'САРАТОВ', 'ТЮМЕНЬ',
                                    'УФА', 'ХАБАРОВСК', 'ЧЕЛЯБИНСК', 'ЯРОСЛАВЛЬ', 
                                   'СТАВРОПОЛЬ', 'ТВЕРЬ', 'ТОМСК'
                                   ]                               
                                 )
        #Обновление датафрейма для Все 18+
        new_data_all_18 = pd.merge(data_all_18_senza_russia_1, Russia_1_updated, how = 'left', on = 'Date')
        result['All 18+'] = Table.make_left_join(new_data_all_18, tasks_local_channels[0], tasks_local_channels[1], 'Date')
        result['All 18+'] = result['All 18+'][loaded_dict_columns['All 18+']]
        
        
        #Суммирование значений долей Пятница(Екатеринбург) с Четвертым каналом (Екатеринбург)
        pyatniza_updated = TeamDataExportService.sum_2_channels(pyatniza, df_channel_4, ['ЕКАТЕРИНБУРГ'])
        #Обновление датафрейма для Все 14-44
        new_data_all_14_44 = pd.merge(data_all_14_44_senza_pyatniza_ekb, pyatniza_updated, how = 'left', on = 'Date')
        result['All 14-44'] = new_data_all_14_44[loaded_dict_columns['All 14-44']]
        
        dict_data = {'All 18+': result['All 18+'], 
         'All 14-59': result['All 14-59'], 
         'All 10-45': result['All 10-45'], 
         'All 14-44': result['All 14-44'], 
         'All 14-54': result['All 14-54'], 
         'All 25-49': result['All 25-49'], 
         'All 25-54': result['All 25-54'], 
         'All 4-45': result['All 4-45'], 
         'All 6-54': result['All 6-54'], 
         'W 14-44': result['W 14-44'], 
         'W 25-59': result['W 25-59']}
        df_dict = Dict_Operations(dict_data).rename_columns_in_dict_with_df(loaded_dict_columns)

        replacements = {'Share': datetime.today().strftime('%B %Y')}
        for bca, df in df_dict.items():
            df_dict[bca].set_index('Date', inplace = True)
            df_dict[bca].index = [replacements.get(x, x) for x in df_dict[bca].index]
            df_dict[bca] = df_dict[bca].reset_index()
            df_dict[bca] = df_dict[bca].rename(columns = {df_dict[bca].columns[0]: 'Date'})
        dict_data_new = Dict_Operations(df_dict).convert_column_with_date('Date')
        return dict_data_new