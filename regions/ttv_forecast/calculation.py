import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
import os
import re
import json
import time
import openpyxl
from IPython.display import JSON

from mediascope_api.core import net as mscore
from mediascope_api.mediavortex import tasks as cwt
from mediascope_api.mediavortex import catalogs as cwc

from OMA_tools.regions.ttv_forecast.constants import Constants_Calculation, Constants__Columns
from OMA_tools.io_data.operations import File, Table, Dict_Operations

# Настраиваем отображение

# Включаем отображение всех колонокv
pd.set_option('display.max_columns', None)

# Cоздаем объекты для работы с TVI API
mnet = mscore.MediascopeApiNetwork()
mtask = cwt.MediaVortexTask()
cats = cwc.MediaVortexCats()


company_filter_list = ['tvNetId IN (40)', #Все 4-45
                       'tvNetId IN (11)', #Все 6-54
                       'tvNetId IN (1)', #Все 14-54
                       'tvNetId IN (1)', #Все 18+
                       'tvCompanyId = 3736 AND regionId IN (12)', #Екатерингбург
                       'tvCompanyId = 3796 AND regionId IN (19)', #Казань
                       'tvCompanyId = 3340 AND regionId IN (4)', #Нижний Новгород
                       'tvCompanyId = 3360 AND regionId IN (15)', #Новосибирск
                       'tvCompanyId = 3448 AND regionId IN (2)' #Санкт-Петербург
]

basedemo_filter_list = ['age >= 4 AND age <= 45', #Все 4-45
                        'age >= 6 AND age <= 54', #Все 6-54
                        'age >= 14 AND age <= 54', #Все 14-54
                        'age >= 18', #Все 18+
                        None, #Екатерингбург
                        'age >= 10 AND age <= 45', #Казань
                        None, #Нижний Новгород
                        None, #Новосибирск
                        None #Санкт-Петербург
] 

regions_dict_list = [Constants_Calculation.regions_dict_4_45, 
                     Constants_Calculation.regions_dict_6_54, 
                     Constants_Calculation.regions_dict_14_54, 
                     Constants_Calculation.regions_dict_18, 
                     None, #Екатерингбург
                     None, #Казань
                     None, #Нижний Новгород
                     None, #Новосибирск
                     None #Санкт-Петербург
]

targets_list = [None, #Все 4-45
                None, #Все 6-54
                None, #Все 14-54
                None, #Все 18+
                Constants_Calculation.targets_ekaterinburg, #Екатерингбург
                None, #Казань
                Constants_Calculation.targets_nizniy_novgorod, #Нижний Новгород
                Constants_Calculation.targets_novosibirsk, #Новосибирск
                Constants_Calculation.targets_saint_petersburg #Санкт-Петербург
]


bca = ['All 4-45', 'All 6-54', 'All 14-54', 'All 18+', 'Ekaterinburg', 'Kazan', 'Nizniy_Novgorod', 'Novosibirsk', 'SaintPetersburg']



def ttv_fact_month(
                   BCA: str, 
                   basedemo_filter: str, 
                   date_filter,
                   regions_dict, 
                   targets,
                   company_filter = 'tvNetId IN (1)',
                   statistics = ['TTVRtgPer'],
                   slices = ['regionName'],
                   sortings = {'regionName': 'ASC'},
                   time_filter = 'timeBand1 >= 50000 AND timeBand1 < 290000', # 5:00 - 29:00
                   options = {
                                "kitId": 3, #TV Index Cities  
                                "totalType": "TotalChannels" #база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
                            },
                   weekday_filter = None, 
                   daytype_filter = None, 
                   targetdemo_filter = None, 
                   location_filter = None):
    if basedemo_filter is not None and regions_dict is not None and targets == None:
        tasks = []
        print("Отправляем задания на расчет")

        # Для каждого региона формируем задание и отправляем на расчет
        for reg_id, reg_name in regions_dict.items():

            project_name = reg_name

            #Передаем id региона в company_filter:
            init_company_filter = company_filter

            if company_filter is not None:
                company_filter = company_filter + f' AND regionId IN ({reg_id})'

            else:
                company_filter = f'regionId IN ({reg_id})'

            # Формируем задание для API TV Index в формате JSON
            task_json = mtask.build_timeband_task(date_filter = date_filter, 
                                         weekday_filter = weekday_filter, daytype_filter = daytype_filter, 
                                         company_filter = company_filter, time_filter = time_filter, 
                                         basedemo_filter = basedemo_filter, targetdemo_filter = targetdemo_filter,
                                         location_filter = location_filter, slices = slices, sortings = sortings,
                                         statistics = statistics, options = options, 
                                         add_city_to_basedemo_from_region = True,
                                         add_city_to_targetdemo_from_region = True
                                        )

            # Для каждого этапа цикла формируем словарь с параметрами и отправленным заданием на расчет
            tsk = {}
            tsk['project_name'] = project_name
            tsk['task'] = mtask.send_timeband_task(task_json)
            tasks.append(tsk)
            time.sleep(3)
            print('.', end = '')

            company_filter = init_company_filter

        print(f"\nid: {[i['task']['taskId'] for i in tasks]}") 

        print('')
        # Ждем выполнения
        print('Ждем выполнения')
        tsks = mtask.wait_task(tasks)
        print('Расчет завершен, получаем результат')

        # Получаем результат
        results = []
        print('Собираем таблицу')
        for t in tasks:
            tsk = t['task'] 
            df_result = mtask.result2table(mtask.get_result(tsk), project_name = t['project_name'])        
            results.append(df_result)
            print('.', end = '')
        df = pd.concat(results)

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[['prj_name']+slices+statistics]

        data = df[['prj_name', 'TTVRtgPer']]
        data = data[data['TTVRtgPer'] != 0]
        data.rename(columns = {'prj_name': 'Регион', 'TTVRtgPer': 'TTV'}, inplace = True)
        #data['BCA'] = BCA
        return data
    elif basedemo_filter == None and regions_dict is not None and targets is not None:
        tasks = []
        print("Отправляем задания на расчет")
        
        company_filter = company_filter + f' AND regionId IN ({regions_dict})'            
        for target, syntax in targets.items():

            # Подставляем значения словаря в параметры
            project_name = target
            basedemo_filter = syntax
            # Формируем задание для API TV Index в формате JSON
            task_json = mtask.build_timeband_task(date_filter = date_filter, 
                                         weekday_filter = weekday_filter, daytype_filter = daytype_filter, 
                                         company_filter = company_filter, time_filter = time_filter, 
                                         basedemo_filter = basedemo_filter, targetdemo_filter = targetdemo_filter,
                                         location_filter = location_filter, slices = slices, sortings = sortings,
                                         statistics = statistics, options = options, 
                                         add_city_to_basedemo_from_region = True,
                                         add_city_to_targetdemo_from_region = True
                                        )

            # Для каждого этапа цикла формируем словарь с параметрами и отправленным заданием на расчет
            tsk = {}
            tsk['project_name'] = project_name
            tsk['task'] = mtask.send_timeband_task(task_json)
            #tsk['BCA'] = target
            #print(tsk.columns)
            tasks.append(tsk)
            time.sleep(3)
            print('.', end = '')

            #company_filter = init_company_filter

        print(f"\nid: {[i['task']['taskId'] for i in tasks]}")
        print('')
        # Ждем выполнения
        print('Ждем выполнения')
        tsks = mtask.wait_task(tasks)
        print('Расчет завершен, получаем результат')

        # Получаем результат
        results = []
        print('Собираем таблицу')
        for t in tasks:
            tsk = t['task'] 
            df_result = mtask.result2table(mtask.get_result(tsk), project_name = t['project_name'])        
            results.append(df_result)
            print('.', end = '')
        df = pd.concat(results)

        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[['prj_name']+slices+statistics]

        data = df[['prj_name', 'TTVRtgPer']]
        data = data[data['TTVRtgPer'] != 0]
        data.rename(columns = {'prj_name': 'Регион', 'TTVRtgPer': 'TTV'}, inplace = True)
        return data


        
class Calculation:
    """
    Class to make calculation through API
    """
    def __init__(self):
        self.date_filter = []
        self.constants_calc = Constants_Calculation()
        self.constants_col = Constants__Columns()
        
    
    def set_date_filter(self, number_of_previous_days = []):
        """
        Receive range of dates in format Year-Month-Day.

        Args:
            number_of_previous_days: list of 2 items. 
            First one is related to the quantity of calculation days. Second one refers to detect to last fact date in calendar. Usually, it's date now - 3 days.
        Returns:
            date_filter: list of 2 dates: start date and stop date of calculation
        """
        date_start = datetime.now() + timedelta(days = number_of_previous_days[0])
        start_date = date_start.strftime('%Y-%m-%d')

        date_stop = datetime.now() + timedelta(days = number_of_previous_days[1])
        stop_date = date_stop.strftime('%Y-%m-%d')

        self.date_filter = [(start_date, stop_date)]


    def make_calculation(self, BCA, company_filter, basedemo_filter, regions_dict, targets, columns_new,
                              weekday_filter = None, daytype_filter = None, location_filter = None,
                              time_filter = 'timeBand1 >= 50000 AND timeBand1 < 290000', # 5:00 - 29:00
                              targetdemo_filter = None, statistics = ['TTVRtgPer'], slices = ['researchDate'], sortings = {'researchDate': 'ASC'},
                              options = {
                                        "kitId": 3, #TV Index Cities  
                                        "totalType": "TotalChannels" #база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
                                        }
                            ):
        
        """
        Receive data from database through API

        Args:
            company_filter: channel ID 
            basedemo_filter: BCA filter
            regions_dict: dict of cities with regions id
            targets: dict of bca
            weekday_filter: weekdays or not
            daytype_filter: filter on types of days in week
            location_filter: None refers to home and country hous
            time_filter: broadcast time
            targetdemo_filter: filter on BCA
            statistics: statistics for calculation
            slices: slices by weeks, months, dates
            sortings: sort in ascending order or not
            options: calculation database

        Returns:
            data_api: list of DataFrames with received data through API
        """
        if regions_dict is not None and targets == None and basedemo_filter is not None and columns_new is not None:
            tasks = []
            print("Отправляем задания на расчет")
            # Для каждого региона формируем задание и отправляем на расчет
            for reg_id, reg_name in regions_dict.items():

                project_name = reg_name

                #Передаем id региона в company_filter:
                init_company_filter = company_filter

                if company_filter is not None:
                    company_filter = company_filter + f' AND regionId IN ({reg_id})'

                else:
                    company_filter = f'regionId IN ({reg_id})'

                # Формируем задание для API TV Index в формате JSON
                task_json = mtask.build_timeband_task(date_filter = self.date_filter, 
                                            weekday_filter = weekday_filter, daytype_filter = daytype_filter, 
                                            company_filter = company_filter, time_filter = time_filter, 
                                            basedemo_filter = basedemo_filter, targetdemo_filter = targetdemo_filter,
                                            location_filter = location_filter, slices = slices, sortings = sortings,
                                            statistics = statistics, options = options, 
                                            add_city_to_basedemo_from_region = True,
                                            add_city_to_targetdemo_from_region = True
                                            )

                # Для каждого этапа цикла формируем словарь с параметрами и отправленным заданием на расчет
                tsk = {}
                tsk['project_name'] = project_name
                tsk['task'] = mtask.send_timeband_task(task_json)
                tasks.append(tsk)
                time.sleep(3)
                print('.', end = '')

                company_filter = init_company_filter

            print(f"\nid: {[i['task']['taskId'] for i in tasks]}") 

            print('')
            # Ждем выполнения
            print('Ждем выполнения')
            tsks = mtask.wait_task(tasks)
            print('Расчет завершен, получаем результат')

            # Получаем результат
            results = []
            print('Собираем таблицу')
            for t in tasks:
                tsk = t['task'] 
                df_result = mtask.result2table(mtask.get_result(tsk), project_name = t['project_name'])        
                results.append(df_result)
                print('.', end = '')
            df = pd.concat(results)

            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[['prj_name']+slices+statistics]
            df_= pd.pivot_table(df, values = statistics,
                                index = ['researchDate'], 
                                columns = ['prj_name'])
            data = df_.rename_axis(None, axis = 0)
            data.columns = data.columns.droplevel(0)
            #columns = columns_new
            data_ = data[columns_new]
            data_.reset_index(inplace = True)
            data_ = data_.rename(columns = {'index': 'date'})
            data_['date'] = data_['date'].apply(lambda x: pd.to_datetime(x))
            return data_
            #solutions.append({BCA: data_})

        elif targets is not None and regions_dict == None and basedemo_filter == None and columns_new is not None:
            # Посчитаем задания в цикле
            tasks = []
            print("Отправляем задания на расчет")

            # Для каждой ЦА формируем задание и отправляем на расчет
            for target, syntax in targets.items():

                # Подставляем значения словаря в параметры
                project_name = target
                basedemo_filter = syntax

                # Формируем задание для API TV Index в формате JSON
                task_json = mtask.build_timeband_task(task_name = project_name, date_filter = self.date_filter, 
                                                    weekday_filter = weekday_filter, daytype_filter = daytype_filter, 
                                                    company_filter = company_filter, time_filter = time_filter, 
                                                    basedemo_filter = basedemo_filter, targetdemo_filter = targetdemo_filter,
                                                    location_filter = location_filter, slices = slices, 
                                                    statistics = statistics, sortings = sortings, options = options,
                                                    add_city_to_basedemo_from_region = True,
                                                    add_city_to_targetdemo_from_region = True)

                # Для каждого этапа цикла формируем словарь с параметрами и отправленным заданием на расчет
                tsk = {}
                tsk['project_name'] = project_name    
                tsk['task'] = mtask.send_timeband_task(task_json)
                tasks.append(tsk)
                time.sleep(2)
                print('.', end = '')

            print(f"\nid: {[i['task']['taskId'] for i in tasks]}") 

            print('')
            # Ждем выполнения
            print('Ждем выполнения')
            tsks = mtask.wait_task(tasks)
            print('Расчет завершен, получаем результат')

            # Получаем результат
            results = []
            print('Собираем таблицу')
            for t in tasks:
                tsk = t['task'] 
                df_result = mtask.result2table(mtask.get_result(tsk), project_name = t['project_name'])        
                results.append(df_result)
                print('.', end = '')
            df = pd.concat(results)

            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[['prj_name']+slices+statistics]
            df_= pd.pivot_table(df, values = statistics,
                                index = ['researchDate'], 
                                columns = ['prj_name'])

            data = df_.rename_axis(None, axis = 0)
            data.columns = data.columns.droplevel(0)
            #columns = columns_new
            data_ = data[columns_new]
            data_.reset_index(inplace = True)
            data_ = data_.rename(columns = {'index': 'date'})
            data_['date'] = data_['date'].apply(lambda x: pd.to_datetime(x))
            return data_
            #solutions.append({BCA: data_})

        elif regions_dict == None and targets == None and columns_new == None and basedemo_filter is not None:
            # Формируем задание для API TV Index в формате JSON
            task_json = mtask.build_timeband_task(date_filter = self.date_filter, weekday_filter = weekday_filter, 
                                                daytype_filter = daytype_filter, company_filter = company_filter, 
                                                time_filter = time_filter, basedemo_filter = basedemo_filter, 
                                                targetdemo_filter = targetdemo_filter,location_filter = location_filter, 
                                                slices = slices, statistics = statistics, sortings = sortings, options = options,
                                                add_city_to_basedemo_from_region = True, 
                                                add_city_to_targetdemo_from_region = True)

            # Отправляем задание на расчет и ждем выполнения
            task_timeband = mtask.wait_task(mtask.send_timeband_task(task_json))

            # Получаем результат
            df = mtask.result2table(mtask.get_result(task_timeband), project_name='Total. Ind')

            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[['prj_name']+slices+statistics]
            df = df.drop(columns = ['prj_name'])
            df.rename(columns = {'researchDate': 'date', 'TTVRtgPer': 'КАЗАНЬ 10-45'}, inplace = True)
            df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
            return df
            #solutions.append({BCA: df})
    
            
    def parallel_main(self, bca_list, company_filter_list, basedemo_filter_list, regions_dict_list, targets_list, columns_list):
        """
            Функция для расчета долей для руководителей групп. ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ
        """

        def get_data_wrapper(args):
            """
                Вспомогательная функция
            """
            bca, company, basedemo, regions, target, columns = args
            try:
                df = self.make_calculation(bca, company, basedemo, regions, target, columns)
                return bca, df
            except Exception as e:
                print(f"Ошибка при получении данных для {bca}: {e}")
                return bca, pd.DataFrame()  # возвращаем пустой DataFrame в случае ошибки

        # Подготавливаем список аргументов
        args_list = [
            (bca_list[j], company_filter_list[j], basedemo_filter_list[j], regions_dict_list[j], targets_list[j], columns_list[j])
            for j in range(len(bca_list))
        ]

        # Используем ThreadPoolExecutor для параллельного выполнения
        with ThreadPoolExecutor(max_workers=min(10, len(args_list))) as executor:
            # Теперь функция принимает один аргумент - кортеж, и распаковывает его
            results = list(executor.map(get_data_wrapper, args_list))

        # Обрабатываем результаты
        res = {}
        for bca, df in results:
            if not df.empty:  # добавляем только непустые DataFrame
                res[bca] = df
        return res


            
    def get__data_through_API(self, 
                              columns_list,
                              bca = bca, 
                              company_filter_list = company_filter_list, 
                              basedemo_filter_list = basedemo_filter_list, 
                              regions_dict_list = regions_dict_list, 
                              targets_list = targets_list
                              ):
        """
        Make DataFrames for each BCA using API
        """
        result = self.parallel_main(bca, company_filter_list, basedemo_filter_list, regions_dict_list, targets_list, columns_list)
        
        dict_data_api = {'All 4-45': result['All 4-45'], 
                 'All 6-54': result['All 6-54'], 
                 'All 14-54': result['All 14-54'], 
                 'All 18+': result['All 18+'], 
                 'EKB_NN_KZN': Table.make_left_join(df1 = result['Ekaterinburg'], df2 = result['Kazan'], df3 = result['Nizniy_Novgorod'], key = 'date'), 
                 'Novosibirsk': result['Novosibirsk'], 
                 'SaintPetersburg': result['SaintPetersburg']}
        return dict_data_api
    
  
    @staticmethod
    def parallel_main_part(date_filter, basedemo_filter_list, regions_dict_list, targets_list, bca_list):
        """
        Функция для расчета долей для руководителей групп. ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ
        """
        def get_data_wrapper(args):
            """
            Вспомогательная функция, принимающая кортеж аргументов
            """
            date_filter, basedemo, regions, target, bca = args
            try:
                df = ttv_fact_month(bca, basedemo, date_filter, regions, target)
                return bca, df
            except Exception as e:
                print(f"Ошибка при получении данных для {bca}: {e}")
                return bca, pd.DataFrame()  # возвращаем пустой DataFrame в случае ошибки

        # Подготавливаем список аргументов
        args_list = [
            (date_filter, basedemo_filter_list[j], regions_dict_list[j], targets_list[j], bca_list[j])
            for j in range(len(bca_list))
        ]

        # Используем ThreadPoolExecutor для параллельного выполнения
        with ThreadPoolExecutor(max_workers = min(10, len(args_list))) as executor:
            # Вариант 1: Используем map для простоты и безопасности
            results = list(executor.map(get_data_wrapper, args_list))

        # Обрабатываем результаты
        res = {}
        for bca, df in results:
            if not df.empty:  # добавляем только непустые DataFrame
                res[bca] = df
        # Объединяем все DataFrame
        if res:
            result_dfs = list(res.values())
            data = pd.concat(result_dfs, ignore_index=True)
        else:
            data = pd.DataFrame()
        return data
   
    
    
    @staticmethod
    def calculate_fact_month(filename, date_filter, girls_cities: str):
        basedemo_filter_list = ['age >= 4 AND age <= 45', #Все 4-45
                            'age >= 6 AND age <= 54', #Все 6-54
                            'age >= 14 AND age <= 54', #Все 14-54
                            'age >= 18', #Все 18+
                            None, #Екатеринбург
                            'age >= 10 AND age <= 45', #Казань
                            None, #Нижний Новгород
                            None, #Новосибирск
                            None #Санкт-Петербург
        ] 

        regions_dict_list = [Constants_Calculation.regions_dict_4_45, 
                             Constants_Calculation.regions_dict_6_54, 
                             Constants_Calculation.regions_dict_14_54, 
                             Constants_Calculation.regions_dict_18, 
                             12,
                             {19: 'КАЗАНЬ 10-45'}, #Казань
                             4,
                             15,
                             2
                            ]

        targets_list = [None, #Все 4-45
                        None, #Все 6-54
                        None, #Все 14-54
                        None, #Все 18+
                        Constants_Calculation.targets_ekaterinburg, #Екатерингбург
                        None, #Казань
                        Constants_Calculation.targets_nizniy_novgorod, #Нижний Новгород
                        Constants_Calculation.targets_novosibirsk, #Новосибирск
                        Constants_Calculation.targets_saint_petersburg #Санкт-Петербург
        ]

        bca = ['All 4-45', 'All 6-54', 'All 14-54', 'All 18+', 'Ekaterinburg', 'Kazan', 'Nizniy_Novgorod', 'Novosibirsk', 'SaintPetersburg']
        
     
        #def parallel_main_part(bca_list, basedemo_filter_list, date_filter, regions_dict_list, targets_list):
        #    """
        #        Функция для расчета долей для руководителей групп. ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ
        #    """
#
        #    def get_data_wrapper(bca, basedemo, date_filter, regions, target):
        #        """
        #            Вспомогательная функция
        #        """
        #        df = ttv_fact_month(bca, basedemo, date_filter, regions, target)
        #        return bca, df
#
        #    res = {}
        #    args_list = [
        #        (bca_list[j], basedemo_filter_list[j], date_filter, regions_dict_list[j], targets_list[j])
        #        for j in range(len(bca_list))
        #    ]
#
        #    with ThreadPoolExecutor(max_workers=10) as executor:
        #        future_to_name = {
        #            executor.submit(get_data_wrapper, bca, basedemo, date_filter, regions, target): bca
        #            for bca, basedemo, date_filter, regions, target in args_list
        #        }
#
        #        for future in as_completed(future_to_name):
        #            bca, df = future.result()
        #            res[bca] = df
        #    return res
#
        
        full_data = Calculation.parallel_main_part(date_filter, basedemo_filter_list, regions_dict_list, targets_list, bca)

        df_dict = File(girls_cities).from_file(0)
        girls = Dict_Operations(df_dict).replace_keys_in_dict(list_of_replacements = ['Tatyana', 'Maria', 'Kseniia'])

        new_dict = {}
        for girl, ttv in girls.items():
            ttv['Регион'] = ttv['Регион'].str.upper()
            df = pd.merge(ttv, full_data, how = 'inner', on = 'Регион')
            df.drop_duplicates(inplace = True)
            df_ = df.set_index('Регион').T
            new_dict[girl] = df_

        File(filename).to_file(new_dict)