import numpy as np
import pandas as pd
from datetime import date, datetime
from OMA_tools.io_data.dates import Dates_Operations
from OMA_tools.io_data.operations import File, Table
from OMA_tools.ml_models.preprocessing import Preprocessing
from OMA_tools.ml_models.groups import GROUPS
from OMA_tools.ml_models.postprocessing import Postprocessing

from OMA_tools.regions.data_extraction.data_coworker import WrapperNoPrints

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger('log').setLevel(logging.INFO)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("log").propagate = False
logging.getLogger("cmdstanpy").disabled = True

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



class Ensemble_Pipeline:
    """
        Класс для запуска ансамблей ML-моделей
    """
    def __init__(self, df, forecast_periods, column_name_with_date):
        """
            df: pd.DataFrame: входной DataFrame, для которого будем строить прогноз
            forecast_periods: кол-во предсказываемых периодов
        """
        self.df = df
        self.forecast_periods = forecast_periods
        self.column_name_with_date = column_name_with_date
    
    def main(
        self, 
        weights_filepath, 
        error_dir = None,
        plots_dir = None, 
        save_dir = None, 
        plots: bool = False, 
        test: bool = False):
        """
            Функция для запуска прогноза ансамбля ML-моделей
            Args:
                filename: Полный путь к файлу с исходными данными
                list_of_replacements: Список из листов, находящихся в файле с исходными данными
                column_name_with_date: Название столбца с датой
                weights_filepath: Полный путь к файлу с весами для каждой модели
                plots_dir: Путь к директории, куда будут сохраняться графики для каждой модели
                save_dir: Путь к директории, куда будут сохраняться итоговые графики с прогнозами
                plots: Переменная типа bool. Если True, то графики строятся. В противном случае нет.
                test: Переменная типа bool. Если True, то тестинг проводится. В противном случае нет.
            Returns:
        """
        #Определение к какой группе относятся данные по тому или иному каналу
        group_1, group_2, group_3, group_4 = GROUPS(self.df).initiate_group()

        avg_forecasts = []

        #GROUP_1
        if not group_1.empty:
            #print('', color.BOLD + color.BLUE + 'Результаты работы различных методов для ТВ-каналов с сезонностью и трендом' + color.END, sep = '\n', end = '\n')
            with WrapperNoPrints():
                avg_forecast_1 = GROUPS(group_1).process_group(self.forecast_periods,
                                                            self.column_name_with_date,
                                                            type_of_group = 'GROUP_1', 
                                                            weights_filepath = weights_filepath, 
                                                            error_dir = error_dir,
                                                            plots_dir = plots_dir, 
                                                            plots = plots, 
                                                            test = test)
            #print(color.BOLD + '== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ С СЕЗОННОСТЬЮ И ТРЕНДОМ ==' + color.END, avg_forecast_1, sep = '\n', end = '\n')
            if plots:
                Postprocessing(group_1, avg_forecast_1).get_plot(self.column_name_with_date, 
                                                                 f'{save_dir}/Cезонность и тренд')
            avg_forecasts.append(avg_forecast_1)

        #GROUP_2
        if not group_2.empty:
            #print('', color.BOLD + color.BLUE + 'Результаты работы различных методов для ТВ-каналов с трендом без сезонности' + color.END, sep = '\n', end = '\n')
            with WrapperNoPrints():
                avg_forecast_2 = GROUPS(group_2).process_group(self.forecast_periods,
                                                            self.column_name_with_date,
                                                            type_of_group = 'GROUP_2', 
                                                            weights_filepath = weights_filepath, 
                                                            error_dir = error_dir,
                                                            plots_dir = plots_dir, 
                                                            plots = plots, 
                                                            test = test)
            #print(color.BOLD + '== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ С ТРЕНДОМ БЕЗ СЕЗОННОСТИ ==' + color.END, avg_forecast_2, sep = '\n', end = '\n')
            if plots:
                Postprocessing(group_2, avg_forecast_2).get_plot(self.column_name_with_date, 
                                                                 f'{save_dir}/Тренд без сезонности')
            avg_forecasts.append(avg_forecast_2)

        #GROUP_3
        if not group_3.empty:
            #print('', color.BOLD + color.BLUE + 'Результаты работы различных методов для ТВ-каналов с сезонностью без тренда' + color.END, sep = '\n', end = '\n')
            with WrapperNoPrints():
                avg_forecast_3 = GROUPS(group_3).process_group(self.forecast_periods,
                                                            self.column_name_with_date,
                                                            type_of_group = 'GROUP_3', 
                                                            weights_filepath = weights_filepath, 
                                                            error_dir = error_dir,
                                                            plots_dir = plots_dir, 
                                                            plots = plots, 
                                                            test = test)
            #print(color.BOLD + '== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ С СЕЗОННОСТЬЮ БЕЗ ТРЕНДА ==' + color.END, avg_forecast_3, sep = '\n', end = '\n')
            if plots:
                Postprocessing(group_3, avg_forecast_3).get_plot(self.column_name_with_date, 
                                                                 f'{save_dir}/Сезонность без тренда')
            avg_forecasts.append(avg_forecast_3)

        #GROUP_4
        if not group_4.empty:
            #print('', color.BOLD + color.BLUE + 'Результаты работы различных методов для ТВ-каналов без сезонности и без тренда'+ color.END, sep = '\n', end = '\n')
            with WrapperNoPrints():
                avg_forecast_4 = GROUPS(group_4).process_group(self.forecast_periods,
                                                            self.column_name_with_date,
                                                            type_of_group = 'GROUP_4', 
                                                            weights_filepath = weights_filepath, 
                                                            error_dir = error_dir,
                                                            plots_dir = plots_dir, 
                                                            plots = plots, 
                                                            test = test)
            #print(color.BOLD + '== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ БЕЗ СЕЗОННОСТИ И БЕЗ ТРЕНДА ==' + color.END, avg_forecast_4, sep = '\n', end ='\n')
            if plots:
                Postprocessing(group_4, avg_forecast_4).get_plot(self.column_name_with_date, 
                                                                 f'{save_dir}/Без сезонности и без тренда')
            avg_forecasts.append(avg_forecast_4)
        general_df = Postprocessing.ensemble_of_models(self.df, *avg_forecasts)
        return general_df
    

    def run_ensemble(self, config_filepath: str, output_filepath: str):
        """
            Метод для запуска процесса прогнозирования и сохранения прогноза в выходной файл формата .xlsx.
        """
        forecast_df = self.main(config_filepath)
        forecast_df = forecast_df.reset_index()
        forecast_df = forecast_df.rename(columns = {forecast_df.columns[0]: 'Date'})
        output_filename = File.generate_filename(output_filepath, '.xlsx')
        forecast_df.to_excel(output_filename)
        #Установка внешнего вида итоговой таблицы с прогнозом
        try:
            writer = pd.ExcelWriter(output_filename, engine = 'xlsxwriter')
            Table(df = forecast_df).make_style_of_table(writer = writer, sheet_name = 'Sheet1', width_col_1 = 5.0, width_col_2 = 17.57, width_col_3 = 13.0)
            writer.close()
            print("Файл успешно сохранен")
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
        return forecast_df
