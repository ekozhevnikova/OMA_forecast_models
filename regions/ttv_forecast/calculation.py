import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta


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

from regions_forecast.constants import Constants_Calculation, Constants__Columns
from OMA_tools.io_data.operations import File, Table, Dict_Operations

# Настраиваем отображение

# Включаем отображение всех колонокv
pd.set_option('display.max_columns', None)

# Cоздаем объекты для работы с TVI API
mnet = mscore.MediascopeApiNetwork()
mtask = cwt.MediaVortexTask()
cats = cwc.MediaVortexCats()


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


    def make_calculation(self, company_filter, basedemo_filter, regions_dict, targets,
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
                            if regions_dict is not None and targets == None and basedemo_filter is not None:
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
                                return df_

                            elif targets is not None and regions_dict == None and basedemo_filter == None:
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
                                return df_

                            elif regions_dict == None and targets == None and basedemo_filter is not None:
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

    @staticmethod
    def make_table(df, columns_new: list):
        """
        Makes a table with columns as in files with fact data
        """
        df_ = df.rename_axis(None, axis = 0)
        df_.columns = df_.columns.droplevel(0)
        columns = columns_new
        data = df_[columns]
        data.reset_index(inplace = True)
        data = data.rename(columns = {'index': 'date'})
        data['date'] = data['date'].apply(lambda x: pd.to_datetime(x))
        return data

    '''
    @staticmethod
    def make_left_join(df1, df2, df3):
        """
        Make left join for 3 DataFrames
        """
        joined_df = pd.merge(df1, df2, on = 'date', how = 'left')
        left_joined_df = pd.merge(joined_df, df3, on = 'date', how = 'left')
        #left_joined_df['date'] = left_joined_df['date'].apply(lambda x: pd.to_datetime(x))

        return left_joined_df
    '''
    
    
    def get__data_through_API(self):
        """
        Make DataFrames for each BCA using API
        """
        df_4_45 = self.make_calculation(company_filter = 'tvNetId IN (40)', 
                                        basedemo_filter = 'age >= 4 AND age <= 45', 
                                        regions_dict = self.constants_calc.regions_dict_4_45,
                                        targets = None)
        data_4_45 = Calculation.make_table(df = df_4_45, columns_new = self.constants_col.columns_new_4_45)

        df_6_54 = self.make_calculation(company_filter = 'tvNetId IN (11)', 
                                        basedemo_filter = 'age >= 6 AND age <= 54', 
                                        regions_dict = self.constants_calc.regions_dict_6_54,
                                        targets = None)
        data_6_54 = Calculation.make_table(df = df_6_54, columns_new = self.constants_col.columns_new_6_54)

        df_14_54 = self.make_calculation(company_filter = 'tvNetId IN (1)', 
                                        basedemo_filter = 'age >= 14 AND age <= 54', 
                                        regions_dict = self.constants_calc.regions_dict_14_54,
                                        targets = None)
        data_14_54 = Calculation.make_table(df = df_14_54, columns_new = self.constants_col.columns_new_14_54)

        df_18 = self.make_calculation(company_filter = 'tvNetId IN (1)', 
                                        basedemo_filter = 'age >= 18', 
                                        regions_dict = self.constants_calc.regions_dict_18,
                                        targets = None)
        data_18 = Calculation.make_table(df = df_18, columns_new = self.constants_col.columns_new_18)

        df_ekb = self.make_calculation(company_filter = 'tvCompanyId = 3736 AND regionId IN (12)', 
                                    basedemo_filter = None, 
                                    targets = self.constants_calc.targets_ekaterinburg,
                                    regions_dict = None)
        data_ekb = Calculation.make_table(df = df_ekb, columns_new = self.constants_col.columns_new_ekb)

        df_kzn = self.make_calculation(company_filter = 'tvCompanyId = 3796 AND regionId IN (19)', 
                                    basedemo_filter = 'age >= 10 AND age <= 45', 
                                    targets = None, 
                                    regions_dict = None)

        df_nn = self.make_calculation(company_filter = 'tvCompanyId = 3340 AND regionId IN (4)', 
                                    basedemo_filter = None, 
                                    targets = self.constants_calc.targets_nizniy_novgorod,
                                    regions_dict = None)
        data_nn = Calculation.make_table(df = df_nn, columns_new = self.constants_col.columns_new_nn)

        data_ekb_kzn_nn = Table.make_left_join(df1 = data_ekb, df2 = df_kzn, df3 = data_nn, key = 'date')

        df_novosib = self.make_calculation(company_filter = 'tvCompanyId = 3360 AND regionId IN (15)', 
                                        basedemo_filter = None, 
                                        targets = self.constants_calc.targets_novosibirsk,
                                        regions_dict = None)
        data_novosib = Calculation.make_table(df = df_novosib, columns_new = self.constants_col.columns_new_novosibirsk)

        df_spb = self.make_calculation(company_filter = 'tvCompanyId = 3448 AND regionId IN (2)', 
                                    basedemo_filter = None, 
                                    targets = self.constants_calc.targets_saint_petersburg,
                                    regions_dict = None)
        data_spb = Calculation.make_table(df = df_spb, columns_new = self.constants_col.columns_new_saint_petersburg)

        data_api = [data_4_45, data_6_54, data_14_54, data_18, data_ekb_kzn_nn, data_novosib, data_spb]
        
        bca_list_names = ['All 4-45', 'All 6-54', 'All 14-54', 'All 18+', 'EKB_NN_KZN', 'Novosibirsk', 'SaintPetersburg']
        dict_data_api = dict(zip(bca_list_names, data_api))

        return dict_data_api