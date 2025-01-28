import numpy as np
import pandas as pd
import json
import pymannkendall as mk
import threading
from OMA_tools.ml_models.preprocessing import Preprocessing
from OMA_tools.ml_models.processing import Forecast_Models
from OMA_tools.ml_models.postprocessing import Postprocessing


class GROUPS():
    """
        Класс для процессинга ML-моделей.
    """

    def __init__(self, df):
        self.df = df


    def initiate_group(self):
        """
            Функция для определения принадлежности к GROUP_1 / GROUP_2 / GROUP_3 / GROUP_4
                Для определения принадлежности к той или иной группе используется две различных характеристики:
                    - Коррелляция и проверка на наличие тренда
                    Если коррелляция >= 0.7 и присутствует тренд => GROUP_1: ВР, в котором есть сезонность и тренд
                    Если коррелляция < 0.7 и присутствует тренд => GROUP_2: ВР с трендом без сезонности
                    Если коррелляция >= 0.7 и нет тренда => GROUP_3: ВР с сезонностью без тренда
                    Если коррелляция < 0.7 и нет тренда => GROUP_4: ВР без сезонности и без тренда
                Returns:
                    group_1, group_2, groups_3, group_4
        """
        df_list_1 = []
        df_list_2 = []
        df_list_3 = []
        df_list_4 = []

        for column in self.df.columns:
            time_series = pd.Series(self.df[column])
            lagged_series = time_series.shift(12)
            correlation = time_series.corr(lagged_series)
            trend_test_result = mk.original_test(time_series)

            #Есть сезонность и есть тренд
            if (correlation >= 0.7) and trend_test_result.h == True:
                df_list_1.append(self.df[column])

            #Нет сезонности, но есть тренд
            if (correlation < 0.7) and trend_test_result.h == True:
                df_list_2.append(self.df[column])

            #Есть сезонность, но нет тренда
            if (correlation >= 0.7) and trend_test_result.h == False:
                df_list_3.append(self.df[column])

            #Нет сезонности и нет тренда
            if (correlation < 0.7) and trend_test_result.h == False:
                df_list_4.append(self.df[column])

        group_1 = pd.DataFrame(df_list_1).T
        group_2 = pd.DataFrame(df_list_2).T
        group_3 = pd.DataFrame(df_list_3).T
        group_4 = pd.DataFrame(df_list_4).T

        return group_1, group_2, group_3, group_4


    def process_group(self,
                    forecast_periods,
                    column_name_with_date,
                    type_of_group,
                    weights_filepath,
                    error_dir: str = None,
                    plots_dir = None,
                    plots: bool = False,
                    test: bool = False):
        """
            Функция для обработки группы моделей
            Args:
                type_of_group: Тип группы (GROUP_1, GROUP_2, GROUP_3, GROUP_4);
                weights_filepath: Полный путь к config-файлу с весами для каждой из ML-моделей;
                plots_dir: Путь к директории, куда будут сохраняться графики;
                plots: Переменная типа bool. Если True, графики строятся, в противном случае нет;
                test: Переменная типа bool. Если True, тестинг моделей проводится, в противном случае нет.
            Returns:
                forecasts: список из прогнозов для каждой модели с определённым весом.
        """
        #поиск последнего месяца в исходном DataFrame
        last_month_in_df = Preprocessing(self.df).search_last_fact_data()[1]

        #Чтение config.json для корректного указания веса каждой из моделей
        with open(f'{weights_filepath}') as f:
            file_content = f.read()
            groups = json.loads(file_content)

        #Определение типа группы в зависимости от последнего месяца в исходном DataFrame
        group_key = f'{type_of_group}_{"not_december" if last_month_in_df != 12 else "december"}'
        if group_key not in groups.keys():
            raise ValueError(f"Такой группы: '{group_key}' не существует! Выберите другую группу.")

        list_of_model_names = list(groups[group_key].keys())

        #Формирование папки для сохранения графиков с прогнозами
        path_to_save = None
        path_to_save_errors = None
        if type_of_group == 'GROUP_1' and plots_dir is not None:
            path_to_save = f'{plots_dir}/Сезонность и тренд'
        if type_of_group == 'GROUP_1' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/Сезонность и тренд'

        if type_of_group == 'GROUP_2' and plots_dir is not None:
            path_to_save = f'{plots_dir}/Тренд без сезонности'
        if type_of_group == 'GROUP_2' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/Тренд без сезонности'

        if type_of_group == 'GROUP_3' and plots_dir is not None:
            path_to_save = f'{plots_dir}/Сезонность без тренда'
        if type_of_group == 'GROUP_3' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/Сезонность без тренда'

        if type_of_group == 'GROUP_4' and plots_dir is not None:
            path_to_save = f'{plots_dir}/Без сезонности и без тренда'
        if type_of_group == 'GROUP_4' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/Без сезонности и без тренда'

        #Обработка группы с моделями
        #forecasts = []
        #for model_name in list_of_model_names:
        #    if model_name not in groups[group_key]:
        #            raise ValueError(f"Модель '{model_name}' не найдена в интересующей группе! Выберите другую модель.")
        #    else:
        #        forecast_df = Forecast_Models(self.df, forecast_periods, column_name_with_date).process_model(model_name = model_name,
        #                                         plots_dir = path_to_save,
        #                                         plots = plots,
        #                                         test = test)
        #        forecasts.append(forecast_df * groups[group_key][model_name])
        #Обработка группы с моделями
        threads = []
        forecasts = []
        for model_name in list_of_model_names:
             if model_name not in groups[group_key]:
                    raise ValueError(f"Модель '{model_name}' не найдена в интересующей группе! Выберите другую модель.")
             else:
                t = threading.Thread(target = Forecast_Models(self.df, forecast_periods, column_name_with_date).process_model, 
                                    args = (forecasts, groups[group_key][model_name], model_name, path_to_save_errors, path_to_save, plots, test))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()

        avg_forecast = Postprocessing.calculate_average_forecast(forecasts)
        return avg_forecast.round(4)