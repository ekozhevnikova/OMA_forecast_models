import numpy as np
import pandas as pd
import datetime
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
from OMA_tools.io_data.operations import File, Table, Dict_Operations
from pmdarima import auto_arima
import pymannkendall as mk
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from scipy import linalg
from contextlib import contextmanager
import os



class Forecast_Models:
    """
        Настоящий класс содержит ряд ML-моделей, применяемых для анализа и прогнозирования Временных Рядов.
    """
    def __init__(self, df, forecast_periods: int, column_name_with_date: str):
        """
            Args:
                df: DataFrame с исходными данными
                forecast_periods: количество предсказываемых периодов (измеряется в месяцах)
                column_name_with_date: название столбца с месяцем
        """
        self.df = df
        self.forecast_periods = forecast_periods
        self.column_name_with_date = column_name_with_date
    

    @contextmanager
    def suppress_stdout():
        with open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            sys.stdout = fnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


    def main(self, filename, list_of_replacements: list):
        """
            Функция для запуска ансамбля ML-моделей для прогнозирования ВР.
        """
        dict_data = File(filename).from_file(0, 0)
        dict_data_new = Dict_Operations(dict_data).replace_keys_in_dict(list_of_replacements)
        merged_df = Dict_Operations(dict_data_new).combine_dict_into_one_dataframe(self.column_name_with_date)
        merged_df[self.column_name_with_date] = pd.to_datetime(merged_df[self.column_name_with_date], format = '%d.%m.%Y')
        merged_df = merged_df.sort_values(by = self.column_name_with_date)
        merged_df.set_index(self.column_name_with_date, inplace = True)

        #TODO

    
    def regression_model(self, param: str):
        """
            Модель Регрессии
                Args:
                    param: Линейный тренд (linear trend) или Логистический тренд (logistic trend)
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        data = self.df.copy()
        # Сброс индекса и добавление названий месяцев
        data.reset_index(inplace = True)
        #Предварительная кодировка в признак
        data['month_name'] = data[self.column_name_with_date].dt.strftime("%B") #преобразование даты ('2020-01-01') в текстовый формат типа 'Январь'

        #С текстовыми данными работать не можем => применяется OneHotEncoder кодировщик для преобразования категориальных или текстовых данных в числа
        #Числа заменяются на единицы и нули, в зависимости от того, какому столбцу какое значение присуще.
        encoder = OneHotEncoder(categories = 'auto', drop = 'first', sparse_output = False)
        encoded_months = encoder.fit_transform(data[['month_name']])#конвертация в массив закодированного столбца

        # Матрица сезонных признаков
        encoded_df_0 = pd.DataFrame(encoded_months, columns = encoder.get_feature_names_out(['month_name']))

        # Колонка с трендом (наклон)
        if param == 'linear trend':
            encoded_df_trend = pd.DataFrame({'Linear_Trend': np.arange(1, data.shape[0] + 1)})
        elif param == 'logistic trend':
            encoded_df_trend = pd.DataFrame({'Log_Trend': np.log(np.arange(1, data.shape[0] + 1))})
        else:
            raise ValueError('Неверно выбран тип тренда.')

        # Свободный член в модели регресии (интерсепт)
        encoded_df_free_var = pd.DataFrame({'free_variable': np.ones(data.shape[0])})

        # Итоговая матрица признаков (сезонность, тренд, интерсепт)
        encoded_df = pd.concat([encoded_df_0, encoded_df_trend, encoded_df_free_var], axis = 1)

        # Новый DataFrame для хранения спрогнозированных значений
        predicted_df = pd.DataFrame({'Date': data[self.column_name_with_date]})

        # Словарь для хранения коэффициентов модели для каждого столбца
        model_coefficients_dict = {}

        # Прогнозирование для каждого столбца
        for column in self.df.columns[1:-1]:  # Пропускаем столбцы column_name_with_date и "month_name"
            A = encoded_df.values
            b = data[column].values.reshape((int(encoded_df.shape[0]), 1))

            # Решаем систему уравнений с помощью метода наименьших квадратов
            k, *_ = linalg.lstsq(A, b)

            # Переводим коэффициенты в список и сохраняем в словарь
            model_coefficients = k.reshape((1, encoded_df.shape[1])).tolist()[0]
            model_coefficients_dict[column] = model_coefficients

            # Прогнозируем значения на обученных данных
            y_pred = [
                sum(np.multiply(encoded_df.iloc[i, :].tolist(), model_coefficients))
                for i in range(encoded_df.shape[0])
            ]

            # Добавляем прогнозируемые значения в новый DataFrame
            predicted_df[f'{column}'] = y_pred

        # Прогнозирование на N месяцев вперед
        # Определяем последний месяц в данных
        last_date = data[self.column_name_with_date].max()

        # Создаем новые даты для следующего года
        future_dates = pd.date_range(last_date + pd.DateOffset(months = 1), periods = self.forecast_periods, freq = 'MS')

        # Создаем DataFrame для будущих дат
        future_df = pd.DataFrame({self.column_name_with_date: future_dates})
        future_df['month_name'] = future_df[self.column_name_with_date].dt.strftime("%B")

        # Преобразуем названия месяцев в бинарные признаки (One-Hot Encoding)
        encoded_future_months = encoder.transform(future_df[['month_name']])
        encoded_future_df_0 = pd.DataFrame(encoded_future_months, columns = encoder.get_feature_names_out(['month_name']))

        # Тренд для новых дат
        if param == 'linear trend':
            encoded_future_df_trend = pd.DataFrame({'Linear_Trend': np.arange(len(data) + 1, len(data) + (self.forecast_periods + 1))})
        elif param == 'logistic trend':
            encoded_future_df_trend = pd.DataFrame({'Log_Trend': np.log(np.arange(len(data) + 1, len(data) + (self.forecast_periods + 1)))}) 

        # Свободный член (интерсепт)
        encoded_future_df_free_var = pd.DataFrame({'free_variable': np.ones(self.forecast_periods)})

        # Итоговая матрица признаков для будущих дат
        encoded_future_df = pd.concat([encoded_future_df_0, encoded_future_df_trend, encoded_future_df_free_var], axis = 1)

        # Новый DataFrame для хранения прогнозируемых значений на следующий год
        future_predictions = pd.DataFrame({self.column_name_with_date: future_df[self.column_name_with_date]})

        # Прогнозирование для каждого столбца на следующий год
        for column in data.columns[1:-1]:  # Пропускаем столбцы column_name_with_date и "month_name"
            model_coefficients = model_coefficients_dict[column]

            y_future_pred = [
                sum(np.multiply(encoded_future_df.iloc[i, :].tolist(), model_coefficients))
                for i in range(encoded_future_df.shape[0])
            ]

            # Добавляем прогнозируемые значения в DataFrame
            future_predictions[f'{column}'] = y_future_pred

        return future_predictions, predicted_df


