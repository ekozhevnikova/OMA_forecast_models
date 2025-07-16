import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


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

# Настраиваем отображение

# Включаем отображение всех колонок
pd.set_option('display.max_columns', None)

# Cоздаем объекты для работы с TVI API
mnet = mscore.MediascopeApiNetwork()
mtask = cwt.MediaVortexTask()
cats = cwc.MediaVortexCats()

class TTV_Federal_Forecast:
    last_fact_date = None
    def __init__(self, date_filter, last_predict_date, holidays):
        self.date_filter = date_filter
        self.last_predict_date = last_predict_date
        self.holidays = holidays
        #self.constants = Constants()
        
    def API_calculation(self,
                        time_filter,
                        targets, 
                        col_name = '', 
                        statistics = ['TTVRtgPer'], # Указываем список статистик для расчета
                        slices = ['researchDate'], #Разбиваем по дням
                        sortings = {'researchDate':'ASC'}, # Задаем условия сортировки: дата (по возрастанию)
                        company_filter = f'tvCompanyId IN (1873)', # Задаем каналы: РЕН-ТВ
                        options = {
                                    "kitId": 1, #TV Index Russia all 
                                    "totalType": "TotalChannels" #база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
                                },
                        weekday_filter = None, 
                        daytype_filter = None, 
                        basedemo_filter = None, 
                        targetdemo_filter = None, 
                        location_filter = None
                        ):
        if basedemo_filter == None and targets != None:
            tasks = []
            print("Отправляем задания на расчет")

            # Для каждой ЦА формируем задание и отправляем на расчет
            # Посчитаем задания в цикле
            tasks = []
            print("Отправляем задания на расчет")

            # Для каждой ЦА формируем задание и отправляем на расчет
            for target, syntax in targets.items():

                # Подставляем значения словаря в параметры
                project_name = target
                basedemo_filter = syntax

                # Формируем задание для API TV Index в формате JSON
                task_json = mtask.build_timeband_task(task_name = project_name, 
                                                      date_filter = self.date_filter, 
                                                      weekday_filter = weekday_filter, 
                                                      daytype_filter = daytype_filter, 
                                                      company_filter = company_filter, 
                                                      time_filter = time_filter, 
                                                      basedemo_filter = basedemo_filter, 
                                                      targetdemo_filter = targetdemo_filter,
                                                      location_filter = location_filter, 
                                                      slices = slices, 
                                                      statistics = statistics, 
                                                      sortings = sortings, 
                                                      options = options)

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
            df = df[['prj_name'] + slices + statistics]

            df_= pd.pivot_table(df, values = statistics,
                                index = ['researchDate'], 
                                columns = ['prj_name'])
            return df_
        
        else:
            # Формируем задание для API TV Index в формате JSON
            task_json = mtask.build_timeband_task(date_filter = self.date_filter,
                                                weekday_filter = weekday_filter, 
                                                daytype_filter = daytype_filter, 
                                                company_filter = company_filter, 
                                                time_filter = time_filter,
                                                basedemo_filter = basedemo_filter, 
                                                targetdemo_filter = targetdemo_filter, 
                                                location_filter = location_filter, 
                                                slices = slices, 
                                                statistics = statistics, 
                                                sortings = sortings, 
                                                options = options)

            # Отправляем задание на расчет и ждем выполнения
            task_timeband = mtask.wait_task(mtask.send_timeband_task(task_json))

            # Получаем результат
            df = mtask.result2table(mtask.get_result(task_timeband))
            # Приводим порядок столбцов в соответствие с условиями расчета
            return df
            
            
    
    def delete_zeros(self, df):
        '''
        Convert date from D.M.Y 00:00:00 to D.M.Y
        '''
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
        df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        return df

    
    def get_data_new_API(self):
        '''
        Get format of dataframe as in file TTV_Federal_fact
        '''
        df_ = self.API_calculation()
        df_ = df_.rename_axis(None, axis = 0)
        df_.columns = df_.columns.droplevel(0)
        columns = ['All 18+', 'All 18-44', 'All 25-54', 'M 18+', 'W 14-44',
                   'All 11-34', 'All 14-59', 'All 25-59', 'W 25-59', 'all 10-45',
                   'All 25-49', 'all 14-44', 'all 4-45', 'm 14-59', 'All 4+', 'W 18-45', 'All 22-55']
        data = df_[columns]
        data.reset_index(inplace = True)
        data = data.rename(columns = {'index': 'date'})
        data['date'] = data['date'].apply(lambda x: pd.to_datetime(x))
        return data
    
    
    def get_ttv_update(self, data_api, filename):
        '''
        Update your dataframe with TTV and save it
        '''
        #data_api = self.get_data_new_API()
        total = pd.read_excel(f'{filename}')
        total = total.drop(['Unnamed: 0'], axis = 1)

        for i, row_i in data_api.iterrows():
            total[total['date'] == row_i[0]] = data_api[data_api['date'] == row_i[0]]

        if total.iloc[-1]['date'] == data_api.iloc[0]['date']: 
                total.iloc[-1] = data_api.iloc[0]
                total = pd.concat([total, data_api.iloc[1:]])
        else:
            total = pd.concat([total, data_api])
        total['date'] = total['date'].apply(lambda x: pd.to_datetime(x))
        total = total.dropna()
        total = total.reset_index(drop = True)
        total.to_excel(filename)
        return total

    
    def __get_predictions(self, total):
        '''
        Get the number of predict days
        '''
        self.last_fact_date = total.date.max()
        #последняя дата предсказываемого периода
        self.last_predict_date = pd.to_datetime(self.last_predict_date, format = '%d.%m.%Y')
        self.last_fact_date = pd.to_datetime(self.last_fact_date, format = '%Y-%m-%d')
        predictions = self.last_predict_date - self.last_fact_date
        predictions = predictions.days
        
        return predictions
    
    def get_cond(self, df, bca):
        ds = pd.to_datetime(df.ds, format = '%Y-%m-%d')
        if bca == 'All 18+' or \
        bca == 'All 25-54' or \
        bca == 'M 18+' or \
        bca == 'All 14-59' or \
        bca == 'All 25-59' or \
        bca == 'All 25-49' or \
        bca == 'All 14-44' or \
        bca == 'M 14-59' or \
        bca == 'All 4+':
            return (ds <= self.last_fact_date) & (ds >= dt.datetime(self.last_fact_date.year - 4, 1, 1))
        else:
            return (ds <= self.last_fact_date)

    @staticmethod
    def to_file(path, bca, result_df):
        '''
        Save your forecasts for each BCA to file 
        '''
        name = path + bca + '.xlsx'
        with pd.ExcelWriter(name, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
                result_df.to_excel(writer)
                
    def get_forecast(self, total, path_to_save, bca_params):
        '''
        Forecast process
        '''
        pred = self.__get_predictions(total)
        for key, value in bca_params.items():
            df = total[['date', key]]
            df = df.rename(columns = {'date': 'ds', key: 'y'})
            cond = self.get_cond(df, key)

            train_df = df[cond]
            model = Prophet(holidays = self.holidays,
                            seasonality_prior_scale = value[0], 
                            holidays_prior_scale = value[1],
                            changepoint_prior_scale = 0.001)
            model.fit(train_df)

            future = model.make_future_dataframe(periods = pred)
            forecast = model.predict(future)

            #во фрейме дата добавляем нижнюю и верхнюю границы 
            train_df.columns = ['ds', 'yhat']
            train_df['yhat_lower'] = train_df['yhat']
            train_df['yhat_upper'] = train_df['yhat']

            tmp = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_cut = tmp[tmp.ds > self.last_fact_date]
            train_df = train_df[pd.to_datetime(train_df.ds, format = '%Y-%m-%d') >= dt.datetime(self.last_fact_date.year - 3, 1, 1)]
            result = pd.concat([train_df, forecast_cut], axis = 0)
            result['bca'] = key

            TTV_Federal_Forecast.to_file(path = path_to_save, bca = key, result_df = result)