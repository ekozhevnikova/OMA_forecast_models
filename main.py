import pandas as pd
import time
from io_data.dates import Dates_Operations
from ml_models.preprocessing import Preprocessing
from ml_models.groups import GROUPS
from ml_models.postprocessing import Postprocessing
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("cmdstanpy").disabled = True
import matplotlib.pyplot as plt
plt.ioff()

prefix = 'D:/работа/groups/data/'
prefix_plots = 'D:/работа/groups/TEST_PLOTS/'
prefix_output = 'D:/работа/groups/RESULTS_NEW/'
df = Preprocessing.get_data_for_forecast(filename = f'{prefix}Data_by_months_TEST.xlsx',
                  list_of_replacements = ['All 18+', 'All 14-59', 'All 10-45', 'All 14-44',
                      'All 14-54', 'All 25-49', 'All 25-54', 'All 4-45', 'All 6-54', 'W 14-44', 'W 25-59'], 
                                         column_name_with_date = 'Date')

start_date = pd.to_datetime(df.index[-1], format = '%d.%m.%Y') #нахождение последнего месяца в DataFrame
stop_date = '25.01.2026' # 31 декабря текущего года
forecast_periods = Dates_Operations.make_difference_in_months(start_date, stop_date)
forecast_periods

def main(df,
         forecast_periods,
         column_name_with_date, 
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
    group_1, group_2, group_3, group_4 = GROUPS(df).initiate_group()

    avg_forecasts = []
    
    #GROUP_1
    if not group_1.empty:
        print('', 'Результаты работы различных методов для ТВ-каналов с сезонностью и трендом', sep = '\n', end = '\n')
        avg_forecast_1 = GROUPS(group_1).process_group(forecast_periods,
                                                       column_name_with_date,
                                                       type_of_group = 'GROUP_1', 
                                                       weights_filepath = weights_filepath, 
                                                       error_dir = error_dir,
                                                       plots_dir = plots_dir, 
                                                       plots = plots, 
                                                       test = test)
        print('== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ С СЕЗОННОСТЬЮ И ТРЕНДОМ ==', avg_forecast_1, sep = '\n', end = '\n')
        if plots:
            Postprocessing(group_1, avg_forecast_1).get_plot(column_name_with_date, 
                                                             f'{save_dir}/Cезонность и тренд')
        avg_forecasts.append(avg_forecast_1)
        
    #GROUP_2
    if not group_2.empty:
        print('', 'Результаты работы различных методов для ТВ-каналов с трендом без сезонности', sep = '\n', end = '\n')
        avg_forecast_2 = GROUPS(group_2).process_group(forecast_periods,
                                                       column_name_with_date,
                                                       type_of_group = 'GROUP_2', 
                                                       weights_filepath = weights_filepath, 
                                                       error_dir = error_dir,
                                                       plots_dir = plots_dir, 
                                                       plots = plots, 
                                                       test = test)
        print('== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ С ТРЕНДОМ БЕЗ СЕЗОННОСТИ ==', avg_forecast_2, sep = '\n', end = '\n')
        if plots:
            Postprocessing(group_2, avg_forecast_2).get_plot(column_name_with_date, 
                                                             f'{save_dir}/Тренд без сезонности')
        avg_forecasts.append(avg_forecast_2)

    #GROUP_3
    if not group_3.empty:
        print('', 'Результаты работы различных методов для ТВ-каналов с сезонностью без тренда', sep = '\n', end = '\n')
        avg_forecast_3 = GROUPS(group_3).process_group(forecast_periods,
                                                       column_name_with_date,
                                                       type_of_group = 'GROUP_3', 
                                                       weights_filepath = weights_filepath, 
                                                       error_dir = error_dir,
                                                       plots_dir = plots_dir, 
                                                       plots = plots, 
                                                       test = test)
        print('== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ С СЕЗОННОСТЬЮ БЕЗ ТРЕНДА ==', avg_forecast_3, sep = '\n', end = '\n')
        if plots:
            Postprocessing(group_3, avg_forecast_3).get_plot(column_name_with_date, 
                                                             f'{save_dir}/Сезонность без тренда')
        avg_forecasts.append(avg_forecast_3)

    #GROUP_4
    if not group_4.empty:
        print('', 'Результаты работы различных методов для ТВ-каналов без сезонности и без тренда', sep = '\n', end = '\n')
        avg_forecast_4 = GROUPS(group_4).process_group(forecast_periods,
                                                       column_name_with_date,
                                                       type_of_group = 'GROUP_4', 
                                                       weights_filepath = weights_filepath, 
                                                       error_dir = error_dir,
                                                       plots_dir = plots_dir, 
                                                       plots = plots, 
                                                       test = test)
        print('== ИТОГОВЫЙ РЕЗУЛЬТАТ РАБОТЫ МЕТОДОВ ДЛЯ ТВ-КАНАЛОВ БЕЗ СЕЗОННОСТИ И БЕЗ ТРЕНДА ==', avg_forecast_4, sep = '\n', end ='\n')
        if plots:
            Postprocessing(group_4, avg_forecast_4).get_plot(column_name_with_date, 
                                                             f'{save_dir}/Без сезонности и без тренда')
        avg_forecasts.append(avg_forecast_4)
    general_df = Postprocessing.ensemble_of_models(df, *avg_forecasts)
    return general_df, test


# # **Расчет**


start = time.perf_counter()

general_df, test = main(df = df,
                  forecast_periods = forecast_periods, 
                  column_name_with_date = 'Date', 
                  weights_filepath = 'ml_models/config.json', 
                  error_dir = 'D:/работа/groups/ERRORS',
                  plots_dir = 'D:/работа/groups/PLOTS_NEW/',
                  save_dir = 'D:/работа/groups/RESULT_NEW/',
                  plots = True, 
                  test = True)

end = time.perf_counter()

print(f'ВРЕМЯ РАБОТЫ КОДА СОСТАВИЛО: {((end-start) / 60):0.2f} мин.')


general_df


# # **Вывод ошибок прогноза**


if test and forecast_periods <= 12:
    test_data =df.iloc[-forecast_periods:]
    general_MAPE = (general_df/test_data - 1) * 100
    general_MAPE = general_MAPE.reset_index()
    mean_values = general_MAPE.mean()
    # Добавляем новую строку с названием индекса "Mean"
    general_MAPE.loc['Mean'] = mean_values
    general_MAPE.to_excel(f'{prefix_output}forecast_error.xlsx')


general_df = general_df.reset_index()
general_df.to_excel(f'{prefix_output}forecast_output.xlsx')


