# In[1]:
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta

from prophet import Prophet
import sys
import os
import re
import json
import time
import openpyxl
from IPython.display import JSON

# Импорт Mediascope API
sys.path.append('C:/')
from mediascope_api.core import net as mscore
from mediascope_api.mediavortex import tasks as cwt
from mediascope_api.mediavortex import catalogs as cwc

# Настраиваем отображение
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Создаем объекты для работы с TVI API
mnet = mscore.MediascopeApiNetwork()
mtask = cwt.MediaVortexTask()
cats = cwc.MediaVortexCats()

# In[54]:

class TTV:
    last_fact_date = None
    
    def __init__(self, date_filter, last_predict_date):
        self.date_filter = date_filter
        self.last_predict_date = last_predict_date
        
    def API_calculation(self, time_filter, company_filter, options, targets,
                     statistics = ['TTVRtgPer'],
                     sortings = {'researchDate': 'ASC'},
                     slices = ['researchDate'],
                     add_city_to_basedemo_from_region = False, 
                     add_city_to_targetdemo_from_region = False,
                     weekday_filter = None,
                     targetdemo_filter = None,
                     location_filter = None,
                     daytype_filter = None,
                     basedemo_filter = None):
        #start_time = time.time()
        #constants = Constants()
        tasks = []
        print("Отправляем задания на расчет")
        
        if targets is not None:
            # Для каждой ЦА формируем задание и отправляем на расчет
            for target, syntax in targets.items():

                # Подставляем значения словаря в параметры 
                project_name = target
                basedemo_filter = syntax
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
                                      options = options,
                                      add_city_to_basedemo_from_region = add_city_to_basedemo_from_region, #Обязательно нужно указать этот параметр, тк работаем в базе городов
                                      add_city_to_targetdemo_from_region = add_city_to_targetdemo_from_region #Обязательно нужно указать этот параметр, тк работаем в базе городов
                                     )
                              # Для каждого этапа цикла формируем словарь с параметрами и отправленным заданием на расчет
                tsk = {}
                tsk['project_name'] = project_name    
                tsk['task'] = mtask.send_timeband_task(task_json)
                tasks.append(tsk)
                time.sleep(2)
                print('.', end = '')

            print(f"\nid: {[i['task']['taskId'] for i in tasks if i is not None and 'task' in i and i['task'] is not None]}") 

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
            df = df[['prj_name'] + slices + statistics]
            return df
        
        elif targets == None:
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
            
            
    def get_predictions(self, total, date_column):
    
        # Получаем последнюю фактическую дату
        self.last_fact_date = total[date_column].max()

        # Преобразуем даты к единому формату
        self.last_predict_date = pd.to_datetime(self.last_predict_date, format='%d.%m.%Y')
        self.last_fact_date = pd.to_datetime(self.last_fact_date, format='%Y-%m-%d')

        # Вычисляем количество дней для предсказания
        predictions = self.last_predict_date - self.last_fact_date
        predictions = predictions.days

        # Возвращаем те же данные, что и вторая функция для совместимости
        return self.last_fact_date, predictions

    
    def get_cond(self, df, bca, bca_list, n):
        ds = pd.to_datetime(df.ds, format='%Y-%m-%d')
    
        if bca in bca_list:
            return (ds <= self.last_fact_date) & (ds >= dt.datetime(self.last_fact_date.year - n, 1, 1))
        else:
            return (ds <= self.last_fact_date)
      
        
    @staticmethod
    def to_file(path, bca, result_df):
        '''
            Save your forecasts for each BCA to file
            Args:
        '''
        name = path + bca + '.xlsx'
        with pd.ExcelWriter(name, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
                result_df.to_excel(writer)

                
    def get_forecast(self, total, path_to_save, bca_params, last_predict_date, date_column, holidays, bca_list, n):
        '''
        Forecast process
        '''
        self.last_fact_date, pred = self.get_predictions(total, date_column)
        for key, value in bca_params.items():
            df = total[[date_column, key]]
            df = df.rename(columns = {date_column: 'ds', key: 'y'})
            cond = self.get_cond(df, key, bca_list, n)

            train_df = df[cond]
            
            model = Prophet(holidays=holidays,
                            changepoint_prior_scale=float(value[0]), 
                            seasonality_prior_scale=float(value[1]),
                            holidays_prior_scale=float(value[2]))


            model.fit(train_df)
            future = model.make_future_dataframe(periods = pred)
            forecast = model.predict(future)

            #во фрейме дата добавляем нижнюю и верхнюю границы 
            train_df.columns = ['ds', 'yhat']
            train_df['yhat_lower'] = train_df['yhat']
            train_df['yhat_upper'] = train_df['yhat']

            tmp = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_cut = tmp[tmp.ds > self.last_fact_date]
            train_df = train_df[pd.to_datetime(train_df.ds, format = '%Y-%m-%d') >= dt.datetime(self.last_fact_date.year - 1, 1, 1)]
            result = pd.concat([train_df, forecast_cut], axis = 0)
            result['bca'] = key
            TTV.to_file(path_to_save, key, result_df = result)