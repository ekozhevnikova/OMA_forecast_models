import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Arial')
import warnings
warnings.filterwarnings('ignore')

from OMA_tools.io_data.time_series import *
from OMA_tools.io_data.operations import Dates_Operations


class PipelineOrchestrator:
    """
        Класс, в котором реализованы пайплайны для запуска ML-моделей, а также некоторые вспомогательные фичи.
    """
    def __init__(self, ts):
        """
            Atributes:
                self.ts: DataFrame с временным рядом, в котором два столбца: Дата, Таргет.
        """
        self.ts = ts


    def corrected_prepare_timeseries(self, date_column='Date', target_column='Share', 
                               criteria=0.15, window_size=30):
        data = self.ts.copy()
        
        # 1. ПРАВИЛЬНАЯ обработка пропусков дат
        data = TimeSeriesTransformer.check_and_fix_dates(data, date_column, target_column)
        
        # 2. Сохраняем исходные значения для восстановления
        original_values = data[target_column].copy()
        original_dates = data[date_column].copy()
        
        # 3. Создаем ОДИН объект трансформера
        transformer = TimeSeriesTransformer(data[target_column])
        
        # 4. Последовательная предобработка
        # 4.1. Стационаризация
        stationary_series, log_transform = transformer.make_stationary()
        
        # 4.2. Обновляем данные
        data_preprocessed = data.copy()
        data_preprocessed[target_column] = stationary_series
        
        # 4.3. Масштабирование
        transformer_updated = TimeSeriesTransformer(data_preprocessed[target_column])
        scaled_values, scaler = transformer_updated.check_scale_and_modify_scale_if_need()
        
        data_preprocessed[target_column] = scaled_values
        
        # 5. Анализ тренда на ПРЕДОБРАБОТАННЫХ данных
        trend_analyze = TimeSeriesTrendAnalyze(data_preprocessed)
        tau = trend_analyze.analyze_trend_comprehensive(target_column=target_column)
        mean, ratio = trend_analyze.analyze_first_diff(target_column=target_column)
        
        # 6. Детрендирование если нужно
        if (tau > 0.5 and tau <= 1) or (tau < -0.5) or (ratio > criteria):
            print('ВЫДЕЛЯЮ ТРЕНД')
            detrend_series, trend = trend_analyze.extract_trend_with_ma(
                window_size=window_size, target_column=target_column
            )
            
            # Заполняем пропуски в тренде
            moving_avg = trend.ffill().bfill().values
            detrended_series_final = data_preprocessed[target_column] - moving_avg
            
            return data_preprocessed, detrended_series_final, moving_avg, log_transform, scaler
        else:
            # Если тренд не выделяем, возвращаем предобработанные данные
            return data_preprocessed, data_preprocessed[target_column], None, log_transform, scaler



    def prepare_timeseries(self, date_column: str = 'Date', target_column: str = 'Share', window_size: int = 7, criteria: float = 0.15):
        """
            Метод для комплексного анализа временных рядов. Здесь реализованы следующие шаги:
                - Поиск пропущенных значений
                - Выстраивание верной хронологии дат
                - Приведение ряда к стационарному виду, если требуется
                - Нормализация данных, если требуется
                - Игра с выбросами
                - Выделение тренда из ВР
            
            Args:
                date_column: str: название колонки с датой. По умолчанию Date.
                target_column: str: название колонки с таргет-переменной. По умолчанию Share.

        """
        data = self.ts.copy()
        # Проверка на наличие пропусков. Если пропуски есть, то ряд приводится к виду без них. В противном случае продолжаем анализ.
        gaps = TimeSeriesTransformer.find_missing_dates(list(data[date_column]))
        if gaps != 0:
            df = TimeSeriesTransformer.create_reverse_dates_from_target(data, date_column, target_column)
        elif gaps == 0:
            df = data

        # Приведение ряда к стационарному виду
        stationary_series, log_transform = TimeSeriesTransformer(df[target_column]).make_stationary()
        # Нормализация данных, если требуется
        target_values, scaler = TimeSeriesTransformer(stationary_series).check_scale_and_modify_scale_if_need()

        # Формирование входного DataFrame для анализа тренда
        df['target'] = target_values
        data_analysis = df[['Date', 'target']]

        # Анализ тренда
        trend_analyze = TimeSeriesTrendAnalyze(data_analysis)
        tau = trend_analyze.analyze_trend_comprehensive(target_column = 'target')
        mean, ratio = trend_analyze.analyze_first_diff(target_column = 'target')
        if (tau > 0.5 and tau <= 1) or (tau < - 0.5) or (ratio > criteria):
            print('ВЫДЕЛЯЮ ТРЕНД')
            detrend_series, trend = trend_analyze.extract_trend_with_ma(window_size = window_size, target_column = 'target')

            series = data_analysis['target']
            
            # Заполняем краевые значения
            moving_avg = trend.ffill().bfill().values
            
            # 2. Удаляем тренд (скользящую среднюю)
            detrended_series = series - moving_avg
            return data_analysis, detrended_series, moving_avg, log_transform, scaler
        else:
            detrended_series = None
            moving_avg = None
            return data_analysis, detrended_series, moving_avg, log_transform, scaler
        
    
    @staticmethod
    def plot_time_series_decomposition(original_data, detrended, trend, date_column: str = 'Date', target_column: str = 'Share'):
        """
            Вспомогательный метод для построения Декомпозиции временного ряда.
        """
        plt.figure(figsize = (10, 6))
        plt.tick_params(axis = 'both', which = 'major', labelsize = 13)

        plt.plot(original_data[date_column], original_data[target_column], color = 'black', linewidth = 1.5, label = 'Init Data')
        plt.plot(original_data[date_column], trend, color = 'green', linewidth = 1.5, label = 'Trend')
        plt.plot(original_data[date_column], detrended, color = 'magenta', linewidth = 1.5, label = 'Detrended')
        plt.xlabel(date_column, fontsize = 14, color = 'black')
        plt.ylabel(target_column, fontsize = 14, color = 'black')
        plt.xticks(rotation = 45)

        plt.legend(loc = 'upper left', ncols = 3, frameon = False, fontsize = 12)
        plt.tight_layout()
        plt.show()



    @staticmethod
    def forecast_share_not_found_programs(new_df, df_hist):
        """
            Функция для расчета прогноза доли для программ, для которых не было найдено похожей программы.
            Функция генерит последние 4 недели и делает поиск по слоту. Считается средняя доля программ для конкретного слота, 
            в котором шла программа. Если не было найдено совпадение по слоту, то считается среднее за последние 4 недели.
        """
        new_df_ = new_df.reset_index(drop = True)
        for i in range(len(new_df_)):
            start_date = new_df_.iloc[i]['Дата']
            slot = new_df_.iloc[i]['Время выхода']
            
            dates = Dates_Operations.get_last_4_weeks(start_date)
            timestamp_dates = [pd.Timestamp(d) for d in dates]
            df_dates = pd.DataFrame(timestamp_dates, columns = ['Дата'])
            #Join дат и исторического DataFrame
            merged = pd.merge(df_dates, df_hist, on = 'Дата', how = 'left')
            #Отбор по слоту
            slot_df = merged[merged['Время выхода'] == slot]
            #Если нашлась какая-то программа по тому же слоту
            if len(slot_df) != 0:
                drop_outlinear = TimeSeriesTransformer(slot_df['Share']).replace_outliers_with_median()
                share_mean = np.mean(drop_outlinear)
                new_df_.at[i, 'Forecast'] = share_mean
            #Если НЕ нашлась какая-то программа по тому же слоту
            elif len(slot_df) == 0:
                drop_outlinear = TimeSeriesTransformer(merged['Share']).replace_outliers_with_median()
                share_mean = np.mean(drop_outlinear)
                new_df_.at[i, 'Forecast'] = share_mean
        return new_df_