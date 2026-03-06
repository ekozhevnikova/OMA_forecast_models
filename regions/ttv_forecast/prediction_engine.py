import pandas as pd
import numpy as np
import xlsxwriter
import copy
import threading
from threading import Lock
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.io_data.dates import Dates_Operations

from OMA_tools.regions.data_extraction.task_builder import *
from OMA_tools.regions.data_extraction.data_coworker import EmployeeExportService
from OMA_tools.regions.data_extraction.leader_ship import LeaderShipDataExtractor
from OMA_tools.regions.ttv_forecast.ttv_constants import Holidays, Prophet_Constants

from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger('log').setLevel(logging.INFO)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("log").propagate = False
logging.getLogger("cmdstanpy").disabled = True
from OMA_tools.regions.ttv_forecast.constants import Holidays, Prophet_Constants

from OMA_tools.io_data.colors import *


import warnings
warnings.filterwarnings('ignore')


class TTV_Calculation:
    def __init__(self, date_filter, tasks_json):
        self.date_filter = date_filter
        self.tasks_json = tasks_json
    

    def get_data(self):
        """
            Метод для выгрузки данных из API.
        """
        #with WrapperNoPrints():
        df = BaseDataService._execute_tasks(self.tasks_json)
        return df
    

    def main_api_calculation(
                        self,
                        filename_fact_data: str,
                        columns_order_dict: dict,
                        slices = ['researchDate'],
                        statistics = ['TTVRtgPer']
                        ):
        """
            Осуществление выгрузки по дням для дальнейшего прогнозирования.
            Args:
                filename_fact_data: путь к файлу с фактическими данными.
            Returns:
                total: Датафрейм с историческими данными.
        """
        # 1. Выгрузка по всем данным
        results = {}
        for group, json_tasks_full in copy.deepcopy(self.tasks_json).items():
            results[group] = EmployeeExportService.process_tasks_parallel(
                                                        json_tasks_full, 
                                                        lambda x: TTV_Calculation(self.date_filter, x).get_data()
                                                    )
        
        # 2. Приведение данных к нужному виду
        results_new = {}
        for bca, df in results['main'].items():
            
            if bca != 'Kazan':
                # Приводим порядок столбцов в соответствие с условиями расчета
                df = df[['prj_name'] + slices + statistics]
                df_= pd.pivot_table(df, values = statistics,
                                    index = ['researchDate'], 
                                    columns = ['prj_name'])

                data = df_.rename_axis(None, axis = 0)
                data.columns = data.columns.droplevel(0)
                data_ = data[columns_order_dict[bca]]
                data_.reset_index(inplace = True)
                data_ = data_.rename(columns = {'index': 'date'})
                data_['date'] = data_['date'].apply(lambda x: pd.to_datetime(x))
                results_new[bca] = data_

            elif bca == 'Kazan':
                # Приводим порядок столбцов в соответствие с условиями расчета
                df = df[['prj_name'] + slices + statistics]
                df = df.drop(columns = ['prj_name'])
                df.rename(columns = {'researchDate': 'date', 'TTVRtgPer': 'КАЗАНЬ 10-45'}, inplace = True)
                df['date'] = df['date'].apply(lambda x: pd.to_datetime(x))
                results_new[bca] = df
        
        # 3. Формирование выходного словаря для дальнейшей работы
        dict_data_api = {
            'All 4-45': results_new['All 4-45'], 
            'All 6-54': results_new['All 6-54'], 
            'All 14-54': results_new['All 14-54'], 
            'All 18+': results_new['All 18+'], 
            'EKB_NN_KZN': Table.make_left_join(
                                    results_new['Ekaterinburg'], 
                                    results_new['Kazan'], 
                                    results_new['Nizniy_Novgorod'], 
                                    key = 'date'
                                    ), 
            'Novosibirsk': results_new['Novosibirsk'], 
            'SaintPetersburg': results_new['SaintPetersburg']}

        # 4. Чтение файла с историческими данными и его обновление
        fact = File(filename_fact_data).from_file(skiprows = 0, index_col = 0)
        total_old = Dict_Operations(fact).replace_keys_in_dict(['All 4-45', 
                                                                'All 6-54', 
                                                                'All 14-54', 
                                                                'All 18+', 
                                                                'EKB_NN_KZN', 
                                                                'Novosibirsk', 
                                                                'SaintPetersburg'])
        for bca, df in total_old.items():
            total_old[bca] = Table.update_table(total_old[bca], dict_data_api[bca], 'date')
        
        # 5. Сохранение последних 28 дней в файл
        File(filename = filename_fact_data).to_file(total_old)

        try:
            # Установка внешнего вида итоговой таблицы по дням за последние 28 дней
            writer = pd.ExcelWriter(filename_fact_data, engine = 'xlsxwriter')
            for key, df in total_old.items():
                Table(df = df).make_style_of_table(writer = writer, sheet_name = key, width_col_1 = 4.5, width_col_2 = 17.57, width_col_3 = 15.86)
            writer.close()
            print("✅ Файл успешно сохранен")
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")
        return total_old
    

    def api_calculation_fact_month(
                            self, 
                            girls_cities: str, 
                            filename_output: str, 
                            statistics = ['TTVRtgPer'], 
                            slices = ['regionName']
                            ):
        """
            Осуществление выгрузки фактической части месяца.
            Args:
                girls_cities: путь к файлу, в котором отражен список каналов-городов, закрепленный за каждой из ответственных.
                filename_output: путь к файлу, в котором будет храниться выгрузка по фактическим данным.
            Returns:
        """
        print(Color.BOLD + Color.BLUE + '=== 🕑 НАЧИНАЮ ВЫГРУЗКУ ФАКТИЧЕСКОЙ ЧАСТИ МЕСЯЦА ДЛЯ TTV ===' + Color.END)
        # 1. Выгрузка данных для всевозможных групп
        data_api = LeaderShipDataExtractor.process_tasks_with_validation(
                                self.tasks_json,
                                lambda x: TTV_Calculation(self.date_filter, x).get_data()
                            )
        # Приводим порядок столбцов в соответствие с условиями расчета
        data_api = data_api[['prj_name'] + slices + statistics]

        full_data = data_api[['prj_name', 'TTVRtgPer']]
        full_data = full_data[full_data['TTVRtgPer'] != 0]
        full_data.rename(columns = {'prj_name': 'Регион', 'TTVRtgPer': 'TTV'}, inplace = True)

        # 2. Формирование выходной таблицы с фактическими данными, исходя из распределения БЦА и каналов-городов между ответственными девочками.
        df_dict = File(girls_cities).from_file(0)
        girls = Dict_Operations(df_dict).replace_keys_in_dict(list_of_replacements = ['Tatyana', 'Maria', 'Kseniia'])

        new_dict = {}
        for girl, ttv in girls.items():
            ttv['Регион'] = ttv['Регион'].str.upper()
            df = pd.merge(ttv, full_data, how = 'inner', on = 'Регион')
            df.drop_duplicates(inplace = True)
            df_ = df.set_index('Регион').T
            new_dict[girl] = df_

        try:
            File(filename_output).to_file(new_dict)
            print('✅ Файл успешно сохранен')
            
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении: {e}")

        return new_dict


class TTV_Forecast:
    """
        Класс для прогнозирования Регионального TTV с использованием модели Prophet.
    """
    def __init__(self, last_predict_date: str, total_dict: dict):
        """
            Атрибут класса
            Args:
                last_predict_date: str: последняя дата предсказания.
        """
        self.last_predict_date = last_predict_date
        self.total_dict = total_dict
        self.proph_consts = Prophet_Constants()
    

    def get_predictions(self, key: str):
        """
            Генерация количества предсказываемых дней.
            
            Args:
                df: словарь из датафрейм с TTV по дням.
            Returns:
                predictions: int: Количество предсказываемых дней 
                last fact date (last_fact_date): последняя фактическая дата.
        """
        last_fact_date = self.total_dict[key].date.max()
        #последняя дата предсказываемого периода
        self.last_predict_date = pd.to_datetime(self.last_predict_date, format = '%d.%m.%Y')
        last_fact_date = pd.to_datetime(last_fact_date, format = '%Y-%m-%d')
        predictions = self.last_predict_date - last_fact_date
        predictions = predictions.days
        return predictions, last_fact_date
    

    def convert_columns_to_prophet_format(self):
        """
            Метод для конвертации столбцов в датафреймах в формат Prophet ('ds' и 'y').
            
            Args:
                dict_of_total: словарь из датафреймов с TTV по дням.
            Returns:
                dict_of_total: словарь из датафреймов с TTV по дням с переименованными колонками 'ds' and 'y'.
                ds_dict: словарь из дат
        """
        dict_of_total = self.total_dict
        ds_dict = {}
    
        for key, df in dict_of_total.items():
            # Создаем новый DataFrame с переименованными колонками
            new_df = df.rename(columns = {'date': 'ds'})
            
            # Сохраняем даты отдельно
            ds_dict[key] = pd.to_datetime(new_df['ds'], format = '%Y-%m-%d')
            
            # Обновляем словарь
            dict_of_total[key] = new_df
    
        return dict_of_total, ds_dict
    

    def get_cond__and__train_df(self, ds_dict, dict_of_total, last_fact_date):
        """
        This function returns condition and datasets of training data which were selected by condition.
        
        Args:
            ds_dict: DataFrame with dates only
            dict_of_total: dict of total DataFrames, where column 'date' was renamed to 'ds' and column 'ГОРОД БЦА' was renamed to 'y'.
            last_fact_date: the last fact date in the datasets.
            
        Returns:
            cond_df: dict of condition on choosing the right history data
            train_df: dict of train_df DataFrames depending on condition (cond_df).
        """
        # Предварительная подготовка ключей и дат
        cond_keys = ['All 4-45', 'All 6-54', 'All 14-54', 'All 18+', 
                    'EKB_NN_KZN', 'Novosibirsk', 'SaintPetersburg']
        
        cond_df = {key: [] for key in cond_keys}
        train_df = {key: [] for key in cond_keys}
        
        # Конвертируем один раз
        ds_dict_converted = Dict_Operations(ds_dict).replace_keys_in_dict(cond_keys)
        dict_of_total_converted = Dict_Operations(dict_of_total).replace_keys_in_dict(cond_keys)
        
        # Предварительно конвертируем last_fact_date
        last_fact_date_dt = pd.to_datetime(last_fact_date)
        
        # Один проход по данным
        for bca_num, (idx, dates) in enumerate(self.proph_consts.cond_date_count.items()):
            for date_num, date in enumerate(dates):
                date_dt = pd.to_datetime(date)  # Конвертируем дату один раз
                
                # Создаем условие
                condition = (ds_dict_converted[idx] <= last_fact_date_dt) & (ds_dict_converted[idx] >= date_dt)
                
                # Применяем условие
                cond_df[idx] = condition
                train_df[idx] = dict_of_total_converted[idx][condition]
        
        return train_df
    

    def get_forecast(self, train_df, predictions, last_fact_date):
        """
            Осуществление процесса прогнозирвоания для каждой БЦА.
        """
        def process_bca(bca_data):
            bca, df = bca_data
            results = pd.DataFrame()
            for icol, col in enumerate(df.columns[1:]):
                tmp_df = pd.concat([df['ds'], df.iloc[:, icol + 1]], axis = 1, keys = ['ds', 'y'])
                
                m = Prophet()
                if self.proph_consts.cond_holidays[bca]:
                    m = Prophet(holidays = Holidays().holidays)
                m.fit(tmp_df)
                
                future = m.make_future_dataframe(periods = predictions)
                forecast = m.predict(future)
                
                tmp_df.columns = ['ds', 'yhat']
                tmp_df['yhat_lower'] = tmp_df['yhat']
                tmp_df['yhat_upper'] = tmp_df['yhat']
                
                tmp = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                forecast_cut = tmp[tmp.ds > last_fact_date]
                tmp_df = tmp_df[tmp_df.ds >= dt.datetime(last_fact_date.year - 4, 1, 1)]
                result = pd.concat([tmp_df, forecast_cut], axis=0)
                result['bca'] = df.columns[icol + 1]
                
                results = pd.concat([result, results]).reset_index(drop = True)
            return bca, results
        
        # Используем многопоточность для параллельной обработки BCA
        with ThreadPoolExecutor(max_workers = mp.cpu_count()) as executor:
            results = list(executor.map(process_bca, train_df.items()))
        
        results_df = {bca: result for bca, result in results}
        return results_df
    

    def make_forecast(self, path: str, file_names: list):
        """
            Пайплайн для прогнозирования регионального TTV
        """
        print(Color.BOLD + Color.VIOLET + '=== 🚀 НАЧИНАЮ МАШИННЫЙ ПРОГНОЗ TTV ===' + Color.END)
        print('\n')

        predictions, last_fact_date = self.get_predictions('All 4-45')
        df, ds_dict = self.convert_columns_to_prophet_format()
        train_df = self.get_cond__and__train_df(
                                                ds_dict = ds_dict,
                                                dict_of_total = df,
                                                last_fact_date = last_fact_date)
        results_df = self.get_forecast(
                                train_df = train_df,
                                predictions = predictions,
                                last_fact_date = last_fact_date)  
        
        print('Прогноз завершён! Сохраняю результаты в выходные файлы. Проверьте соответствующую папку. Файлы должны обновиться.')
        File.to_file_from_dict(
                        path = path,
                        dict_data = results_df,
                        file_names = file_names)
        
        print(Color.BOLD + Color.VIOLET + '=== Данные выгружены и сохранены. Спасибо за ваше ожидание! 😊 ===' + Color.END)

