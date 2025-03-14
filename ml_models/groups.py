import os
import numpy as np
import pandas as pd
import json
import pymannkendall as mk
import threading
from OMA_tools.ml_models.preprocessing import Preprocessing
from OMA_tools.ml_models.processing import Forecast_Models
from OMA_tools.ml_models.postprocessing import Postprocessing
from OMA_tools.io_data.operations import Table



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
            if (correlation >= 0.65) and trend_test_result.h == True:
                df_list_1.append(self.df[column])

            #Нет сезонности, но есть тренд
            if (correlation < 0.65) and trend_test_result.h == True:
                df_list_2.append(self.df[column])

            #Есть сезонность, но нет тренда
            if (correlation >= 0.5) and trend_test_result.h == False:
                df_list_3.append(self.df[column])

            #Нет сезонности и нет тренда
            if (correlation < 0.65) and trend_test_result.h == False:
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
                name_of_group_1: Имя группы 1
                name_of_group_2: Имя группы 2
                name_of_group_3: Имя группы 3
                name_of_group_4: Имя группы 4
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
        #создание папки для сохранения выходных файлов с ошибками
        #os.makedirs(path_to_save_errors, exist_ok = True)
        if type_of_group == 'GROUP_1' and plots_dir is not None:
            path_to_save = f'{plots_dir}/GROUP_1'
        if type_of_group == 'GROUP_1' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/GROUP_1'

        if type_of_group == 'GROUP_2' and plots_dir is not None:
            path_to_save = f'{plots_dir}/GROUP_2'
        if type_of_group == 'GROUP_2' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/GROUP_2'

        if type_of_group == 'GROUP_3' and plots_dir is not None:
            path_to_save = f'{plots_dir}/GROUP_3'
        if type_of_group == 'GROUP_3' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/GROUP_3'

        if type_of_group == 'GROUP_4' and plots_dir is not None:
            path_to_save = f'{plots_dir}/GROUP_4'
        if type_of_group == 'GROUP_4' and error_dir is not None:
            path_to_save_errors = f'{error_dir}/GROUP_4'

        #Обработка группы с моделями
        threads = []
        forecasts = []
        tests = []
        trains = []
        forecasts_with_weight = []
        for model_name in list_of_model_names:
             if model_name not in groups[group_key]:
                    raise ValueError(f"Модель '{model_name}' не найдена в интересующей группе! Выберите другую модель.")
             else:
                t = threading.Thread(target = Forecast_Models(self.df.copy(), forecast_periods, column_name_with_date).process_model_PARALLEL, 
                                    args = (forecasts, tests, trains, model_name, path_to_save_errors, path_to_save, plots, test))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()
        
        for model_name_forecast_df in forecasts:
            model_name = list(model_name_forecast_df.keys())[0]
            forecast_df = list(model_name_forecast_df.values())[0]
            
            forecasts_with_weight.append(forecast_df * groups[group_key][model_name])

            #Если задан параметр test == True
            test_data = None
            if test:
                train_data = list(list(filter(lambda x: model_name in list(x.keys()), trains))[0].values())[0]
                test_data = list(list(filter(lambda x: model_name in list(x.keys()), tests))[0].values())[0]

            # Если задан параметр plots == True
            if plots and plots_dir is not None:
                Postprocessing(self.df, forecast_df).get_plot(column_name_with_date = column_name_with_date,
                                                                save_dir = f'{path_to_save}/{model_name}', test_data = test_data)
            # Если задан параметр test == True    
            if test:
                error_df = Postprocessing.calculate_forecast_error(
                                    forecast_df = forecast_df,
                                    test_data = test_data
                                )
                error_df.to_excel(f'{path_to_save_errors}/{model_name}_MAPE(%).xlsx')
                
                #Установка внешнего вида итоговых таблиц
                writer = pd.ExcelWriter(f'{path_to_save_errors}/{model_name}_MAPE(%).xlsx', engine = 'xlsxwriter')
                self.df.to_excel(writer, sheet_name = 'Sheet1', startrow = 1, header = False)

                workbook = writer.book
                worksheet = writer.sheets['Sheet1']

                header_format = workbook.add_format({
                                                    'bold': True,
                                                    'text_wrap': True, #перенос текста
                                                    'align': 'center' #выравнение текста в ячейке
                                                })
                table_fmt = workbook.add_format({'num_format': '0.0', 
                                                'align': 'center' #выравнение текста в ячейке
                                                })

                for col_num, value in enumerate(self.df.columns.values):
                    worksheet.write(0, col_num + 1, value, header_format)

                for col in range(len(self.df.columns)):
                    writer.sheets['Sheet1'].set_column(1,  col + 1, 17.0, table_fmt)
                    #writer.sheets[sheet_name].set_column(column_start, col + 1, width_col_3, table_fmt)
                worksheet.set_column('A:A', 17.0)
                #worksheet.set_column('B:B', width_col_2)
                writer.close()


        avg_forecast = Postprocessing.calculate_average_forecast(forecasts_with_weight)
        #avg_forecast = Postprocessing.calculate_average_forecast(list(map(lambda x: list(x.values())[0], forecasts)))
        #return forecasts, trains, tests, avg_forecast.round(4)
        return avg_forecast.round(4)