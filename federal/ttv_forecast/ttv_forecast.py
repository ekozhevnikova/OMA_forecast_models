import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta

from OMA_tools.regions.data_extraction.task_builder import BaseDataService
from OMA_tools.io_data.colors import *

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger('log').setLevel(logging.INFO)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("log").propagate = False
logging.getLogger("cmdstanpy").disabled = True

class FederalConstants:
    def __init__(self):
        pass


    @property
    def company_filter_fed(self):
        """
            Словарь БЦА Тематика
        """
        return f'tvCompanyId IN (1873)'
    

    @property
    def options_fed(self):
        """
            Словарь БЦА Тематика
        """
        return {
            "kitId": 1,                     # TV Index Russia all 
            "totalType": "TotalChannels"    # база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
        }


    @property
    def fed_bca(self):
        """
            Словарь БЦА Федеральное ТВ
        """
        return {
            'All 18+': 'age >= 18',
            'All 18-44': 'age >= 18 AND age <= 44',
            'All 25-54': 'age >= 25 AND age <= 54',
            'M 18+': 'age >= 18 AND sex = 1',
            'W 14-44': 'age >= 14 AND age <= 44 AND sex = 2',
            'All 11-34': 'age >= 11 AND age <= 34',
            'All 14-59': 'age >= 14 AND age <= 59',
            'All 25-59': 'age >= 25 AND age <= 59',
            'W 25-59': 'age >= 25 AND age <= 59 AND sex = 2',
            'all 10-45': 'age >= 10 AND age <= 45',
            'All 25-49': 'age >= 25 AND age <= 49',
            'all 14-44': 'age >= 14 AND age <= 44',
            'all 4-45': 'age >= 4 AND age <= 45',
            'm 14-59': 'age >= 14 AND age <= 59 AND sex = 1',
            'All 4+': 'age >= 4',
            'W 18-45': 'age >= 18 AND age <= 45 AND sex = 2',
            'All 22-55': 'age >= 22 AND age <= 55'
        }

    
    @property
    def them_bca(self):
        """
            Словарь БЦА Тематика
        """
        return {
            'All 25-49 Them': 'age >= 25 AND age <= 49',
            'W 25-49 Them': 'age >= 25 AND age <= 49 AND sex = 2',
            'M 25-49 Them': 'age >= 25 AND age <= 49 AND sex = 1'
        }
    

    @property
    def forecast_params_fed(self):
        """
            Словарь с параметрами для построения прогноза
        """
        return {
            'All 14-59': [5.0, 5.0], 
            'All 11-34': [5.0, 10.0], 
            'W 25-59' : [5.0, 10.0],
            'All 4+': [10.0, 5.0],
            'all 10-45': [5.0, 20.0], 
            'all 4-45': [5.0, 20.0],
            'M 18+': [10.0, 20.0],
            'All 25-54': [20.0, 5.0], 
            'm 14-59': [20.0, 5.0],
            'All 25-59': [20.0, 10.0],
            'All 18+': [20.0, 20.0], 
            'All 25-49': [20.0, 20.0], 
            'all 14-44': [20.0, 20.0],
            'All 18-44': [20.0, 20.0], 
            'W 14-44': [20.0, 20.0], 
            'W 18-45': [20.0, 20.0],
            'All 22-55': [20.0, 20.0],
            'All 25-49 Them': [10.0, 10.0], 
            'W 25-49 Them': [10.0, 10.0],
            'M 25-49 Them': [10.0, 10.0],
            'All 4-40 Them': [10.0, 10.0]
        }
    

    @property
    def bca_list_per_cond(self):
        """
            Словарь с параметрами для построения прогноза
        """
        return ['All 18+', 'All 25-54', 'M 18+', 'All 14-59', 'All 25-59', 'All 25-49', 'All 14-44', 'M 14-59', 'All 4+']


class MoscowConstants:
    def __init__(self):
        pass


    @property
    def company_filter_moscow(self):
        """
            Словарь БЦА Тематика
        """
        return 'tvCompanyId = 3322 AND regionId IN (1)'
    

    @property
    def options_moscow(self):
        """
            Словарь БЦА Тематика
        """
        return {
            "kitId": 3,                     # TV Index Russia all 
            "totalType": "TotalChannels"    # база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
        }


    @property
    def moscow_bca(self):
        """
            Словарь БЦА Федеральное ТВ
        """
        return {
            'Все 18+': 'age >= 18',
            'Все 25-54': 'age >= 25 AND age <= 54',
            'ж 14-44': 'age >= 14 AND age <= 44 AND sex = 2',
            'Все 14-59': 'age >= 14 AND age <= 59',
            'Все 25-59': 'age >= 25 AND age <= 59',
            'ж 25-59': 'age >= 25 AND age <= 59 AND sex = 2',
            'Все 10-45': 'age >= 10 AND age <= 45',
            'Все 25-49': 'age >= 25 AND age <= 49',
            'Все 4-45': 'age >= 4 AND age <= 45',
            'Все 22-55': 'age >= 22 AND age <= 55',
            'Все 14-44': 'age >= 14 AND age <= 44'
        }
    

    @property
    def forecast_params_moscow(self):
        """
            Словарь с параметрами для построения прогноза
        """
        return {
            'Все 18+': [0.01, 10, 10],
            'Все 25-54': [0.01, 5, 10],
            'ж 14-44': [0.01, 10, 10], 
            'Все 14-59': [0.01, 20, 20],
            'Все 25-59': [0.01, 20, 20], 
            'ж 25-59': [0.01, 5, 10],
            'Все 10-45': [0.01, 10, 10], 
            'Все 25-49': [0.01, 20, 5], 
            'Все 4-45': [0.01, 20, 20],
            'Все 22-55': [0.01, 5, 10], 
            'Все 14-44': [0.01, 10, 10]
        }
    

    @property
    def bca_list_per_cond(self):
        """
            Словарь с параметрами для построения прогноза
        """
        return [
            'Все 14-44', 'Все 14-59', 'Все 22-55','Все 25-49', 'Все 25-54', 
            'Все 25-59', 'ж 14-44', 'ж 25-59', 'Все 10-45', 'Все 4-45',  'Все 18+'
        ]




class TTV:
    """
        Класс для выгрузки и построения прогноза Федерального, Тематического, а также Московского TTV
    """ 

    def __init__(
        self, 
        date_filter, 
        last_predict_date,
        company_filter,
        holidays,
        add_city_to_basedemo_from_region = False, 
        add_city_to_targetdemo_from_region = False
        ):
        self.date_filter = date_filter
        self.last_predict_date = last_predict_date
        self.company_filter = company_filter
        self.holidays = holidays
        self.add_city_to_basedemo_from_region = add_city_to_basedemo_from_region
        self.add_city_to_targetdemo_from_region = add_city_to_targetdemo_from_region


        self.statistics = ['TTVRtgPer']                 # Указываем список статистик для расчета
        self.slices = ['researchDate']                  # Разбиваем по дням
        self.sortings = {'researchDate':'ASC'}          # Задаем условия сортировки: дата (по возрастанию)
        self.weekday_filter = None 
        self.daytype_filter = None 
        self.targetdemo_filter = None 
        self.location_filter = None
    


    def make_api_calculation(self, targets, time_filter, options, basedemo_filter = None):
        """
            Метод для выгрузки данных из БД
        """
        # Формируем задачи в формате json
        tasks = BaseDataService._build_timeband_common_params(
                                                        date_filter = self.date_filter, company_filter = self.company_filter, 
                                                        basedemo_filter = basedemo_filter, regions_id = None,          # работаем в Федеральной Базе
                                                        targets = targets, time_filter = time_filter, 
                                                        statistics = self.statistics, slices = self.slices, 
                                                        sortings = self.sortings, options = options,
                                                        location_filter = self.location_filter, weekday_filter = self.weekday_filter,
                                                        daytype_filter = self.daytype_filter, targetdemo_filter = self.targetdemo_filter,
                                                        add_city_to_basedemo_from_region = self.add_city_to_basedemo_from_region,
                                                        add_city_to_targetdemo_from_region = self.add_city_to_targetdemo_from_region
                                                    )
        # Отправляем задачи на расчет
        df = BaseDataService._execute_tasks(tasks)
        return df
    

    def get_ttv_update(self, data_api, filename):
        """
            Обновляет файл с фактическими данными
        """
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
    

    def make_table_output(self, df, column_order: list):
        df_pivot = pd.pivot_table(df, values = self.statistics,
                                index = ['researchDate'], 
                                columns = ['prj_name'])
        
        df_ = df_pivot.rename_axis(None, axis = 0)
        df_.columns = df_.columns.droplevel(0)
        #columns = ['All 18+', 'All 18-44', 'All 25-54', 'M 18+', 'W 14-44',
        #           'All 11-34', 'All 14-59', 'All 25-59', 'W 25-59', 'all 10-45',
        #           'All 25-49', 'all 14-44', 'all 4-45', 'm 14-59', 'All 4+', 'W 18-45', 'All 22-55']
        output_df = df_ [column_order]
        output_df.reset_index(inplace = True)
        output_df = output_df.rename(columns = {'index': 'date'})
        output_df['date'] = output_df['date'].apply(lambda x: pd.to_datetime(x))
        return output_df
    

    def __get_predictions(self, total):
        """
            Генерация количества прогнозных дней
        """
        self.last_fact_date = total.date.max()
        #последняя дата предсказываемого периода
        self.last_predict_date = pd.to_datetime(self.last_predict_date, format = '%d.%m.%Y')
        self.last_fact_date = pd.to_datetime(self.last_fact_date, format = '%Y-%m-%d')
        predictions = self.last_predict_date - self.last_fact_date
        predictions = predictions.days
        
        return predictions
    

    def get_cond(self, df, bca, bca_list, n = 4):
        ds = pd.to_datetime(df.ds, format = '%Y-%m-%d')

        if bca in bca_list:
            return (ds <= self.last_fact_date) & (ds >= dt.datetime(self.last_fact_date.year - n, 1, 1))
        
        else:
            return (ds <= self.last_fact_date)
    

    @staticmethod
    def to_file(path, bca, result_df):
        """
            Метод сохраняет прогноз для каждого bca в отдельный файл
        """
        name = path + bca + '.xlsx'
        with pd.ExcelWriter(name, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
                result_df.to_excel(writer)

                
    def get_forecast(self, total, path_to_save, bca_params, bca_list, num_of_years_in_train = 3):
        """
            Построение прогноза
        """
        pred = self.__get_predictions(total)

        forecast_results = {}
        for key, value in bca_params.items():
            df = total[['date', key]]
            df = df.rename(columns = {'date': 'ds', key: 'y'})
            cond = self.get_cond(df, key, bca_list)

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
            train_df = train_df[pd.to_datetime(train_df.ds, format = '%Y-%m-%d') >= dt.datetime(self.last_fact_date.year - num_of_years_in_train, 1, 1)]
            result = pd.concat([train_df, forecast_cut], axis = 0)
            result['bca'] = key

            TTV.to_file(path = path_to_save, bca = key, result_df = result)

            forecast_results[key] = result
        
        return forecast_results

    

    def ttv_pipeline_fed(
        self, 
        filename_with_fact: str,
        path_to_save_results: str,
        forecast_params: dict, 
        bca_list_per_cond: list, 
        fed_targets: dict, 
        them_targets: dict, 
        them_time_filter: str,  
        children_time_filter: str = 'timeBand1 >= 60000 AND timeBand1 < 220000',
        fed_time_filter: str = 'timeBand1 >= 50000 AND timeBand1 < 290000',
        children_basedemo_filter: str = 'age >= 4 AND age <= 40',
        fed_options = {
            "kitId": 1,                     # TV Index Russia all 
            "totalType": "TotalChannels"    # база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
        },
        columns_order: list = [
            'date', 'All 18+', 'All 18-44', 'All 25-54', 'M 18+', 'W 14-44',
            'All 11-34', 'All 14-59', 'All 25-59', 'W 25-59', 'all 10-45',
            'All 25-49', 'all 14-44', 'all 4-45', 'm 14-59', 'All 4+', 'W 18-45', 
            'All 22-55', 'All 25-49 Them', 'W 25-49 Them', 'M 25-49 Them', 'All 4-40 Them'
        ]
        ):
        """
            Полный пайплайн для выгрузки данных по TTV и построению прогноза.

            Параметры:
            ----------
                filename_with_fact: str
                    Путь к файлу с фактическими данными
                path_to_save_results: str
                    Путь к папке с результатами прогноза
                fed_targets: dict
                    Словарь из выгружаемых БЦА для Федерального ТВ
                them_targets: dict
                    Словарь из выгружаемых БЦА для Тематического ТВ
                them_time_filter: str
                    Эфирные сутки для Тематического ТВ
                children_time_filter: str
                    Эфирные сутки для Детской тематической аудитории
                fed_time_filter: str
                    Эфирные сутки для Федерального ТВ 
                children_basedemo_filter: str
                    Детская тематическая аудитория
                fed_options: dict
                    Словарь подключения к федеральной базе Russia0+
        """

        # ШАГ 1. ВЫГРУЗКА ДАННЫХ ДЛЯ ФЕДЕРАЛЬНОГО ТВ
        fed_output = self.make_api_calculation(fed_targets, fed_time_filter, fed_options)
        fed_df = self.make_table_output( fed_output, list(fed_targets.keys()) )

        print(Color.VIOLET + 'Данные по Федеральному ТВ выгружены.' + Color.END)

        ## ШАГ 2. ВЫГРУЗКА ДАННЫХ ТЕМАТИКИ
        them_output = self.make_api_calculation(them_targets, them_time_filter, fed_options)
        them_df = self.make_table_output( them_output, list(them_targets.keys()) )

        # ШАГ 3. ВЫГРУЗКА ДАННЫХ ДЕТСКОЙ АУДИТОРИИ ТЕМАТИКИ
        child_them_output = self.make_api_calculation(None, children_time_filter, fed_options, children_basedemo_filter)
        child_them_output.rename(columns = {'TTVRtgPer': 'All 4-40 Them', 'researchDate': 'date'}, inplace = True)
        child_them_output = child_them_output[['date', 'All 4-40 Them']]
        child_them_output['date'] = child_them_output['date'].apply(lambda x: pd.to_datetime(x))

        # ШАГ 4. СОЗДАНИЕ ДАТАФРЕЙМА ДЛЯ ТЕМАТИЧЕСКОГО ТВ
        thematic_df = pd.merge(them_df, child_them_output, on = 'date', how = 'left')

        print(Color.GREEN + 'Данные по Тематическому ТВ выгружены.' + Color.END)

        # ШАГ 5. ОБЪЕДИНЕНИЕ ФЕДЕРАЛЬНОЙ ВЫГРУЗКИ И ТЕМАТИЧЕСКОЙ
        data = pd.merge(fed_df, thematic_df, on = 'date', how = 'left')
        full_data = data[columns_order]
        
        # ШАГ 6. ОБНОВЛЕНИЕ ФАЙЛА С ФАКТИЧЕСКИМИ ДАННЫМИ
        total_df = self.get_ttv_update(full_data, filename_with_fact)

        print(Color.ORANGE + '=== 🚀 НАЧИНАЮ МАШИННЫЙ ПРОГНОЗ TTV ===' + Color.END)

        # ШАГ 7. ПОСТРОЕНИЕ ПРОГНОЗА
        forecast_results = self.get_forecast(total_df, path_to_save_results, forecast_params, bca_list_per_cond)

        print(Color.BOLD + Color.PURPLE + '=== Данные выгружены и сохранены. Спасибо за ваше ожидание! 😊 ===' + Color.END)

        return full_data, forecast_results
    


    def ttv_pipeline_moscow(
        self,
        filename_with_fact: str,
        path_to_save_results: str,
        forecast_params: dict, 
        bca_list_per_cond: list, 
        moscow_targets: dict, 
        time_filter: str = 'timeBand1 >= 50000 AND timeBand1 < 290000',
        moscow_options = {
            "kitId": 3,                     # TV Index Russia all 
            "totalType": "TotalChannels"    # база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
        },
        columns_order = [
            'date', 'Все 14-44', 'Все 14-59', 'Все 22-55','Все 25-49', 'Все 25-54', 
            'Все 25-59', 'ж 14-44', 'ж 25-59', 'Все 10-45', 'Все 4-45',  'Все 18+'
        ]
    ):
        # ШАГ 1. ВЫГРУЗКА ДАННЫХ ДЛЯ ФЕДЕРАЛЬНОГО ТВ
        moscow_output = self.make_api_calculation(moscow_targets, time_filter, moscow_options)
        moscow_df = self.make_table_output( moscow_output, list(moscow_targets.keys()) )
        moscow_df = moscow_df[columns_order]

        print(Color.GREEN + 'Данные по Москве выгружены.' + Color.END)

        # ШАГ 2/ ОБНОВЛЕНИЕ ФАЙЛА С ФАКТИЧЕСКИМИ ДАННЫМИ
        total_df = self.get_ttv_update(moscow_df, filename_with_fact)


        # ШАГ 3. ПОСТРОЕНИЕ ПРОГНОЗА
        print(Color.ORANGE + '=== 🚀 НАЧИНАЮ МАШИННЫЙ ПРОГНОЗ TTV ===' + Color.END)

        forecast_results = self.get_forecast(total_df, path_to_save_results, forecast_params, bca_list_per_cond)

        print(Color.BOLD + Color.PURPLE + '=== Данные выгружены и сохранены. Спасибо за ваше ожидание! 😊 ===' + Color.END)

        return moscow_df, forecast_results