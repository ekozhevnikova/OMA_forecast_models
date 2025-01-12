import sys
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

    def naive_forecast(self, past_values: int = 3, weigh_error: float = 0):
        # TODO documentation
        """
            Наивный метод прогнозирования с учетом ошибки (опционально).

                Args:
                    past_values: Количество предыдущих месяцев, по которым выполняется прогноз (по умолчанию 3).
                    weigh_error: Вес, с которым учитывается ошибка [0:1].
                                        Если 0, то используется простой наивный метод (по умолчанию 0).

                Returns:
                    Новый DataFrame с прогнозом.
        """

        # Проверка, что past_values корректен
        if past_values <= 0:
            raise ValueError("Количество предыдущих месяцев (past_values) должно быть больше нуля.")

        if len(self.df) < past_values:
            raise ValueError(f"Недостаточно данных для анализа. Требуется минимум {past_values} месяцев данных.")

        if not (0 <= weigh_error <= 1):
            raise ValueError("Вес ошибки (weigh_error) должен быть в диапазоне [0, 1].")

        forecast_results = {}

        # Прогнозирование наивным методом
        for column in self.df.columns:
            st_data = self.df[column].copy()

            # Получаем значения за последние `past_values` месяцев
            last_values = st_data.tail(past_values)

            # Находим среднее значение
            average = last_values.mean()

            # Если включен учет ошибки
            if weigh_error > 0:
                for i in range(past_values, 0, -1):
                    error = (st_data.iloc[-i] - average)  # Рассчитываем ошибку
                    average += error * weigh_error

            # Создаем новую серию с прогнозируемым значением
            forecast_values = [average] * self.forecast_periods
            forecast_df = pd.DataFrame(
                {column: forecast_values},
                index=pd.date_range(
                    start=self.df.index[-1] + pd.DateOffset(months=1),
                    periods=self.forecast_periods,
                    freq='ME'
                )
            )

            forecast_results[column] = forecast_df
        # Объединение всех столбцов в один DataFrame
        result_df = pd.concat(forecast_results.values(), axis=1)

        return result_df


    # def seasonal_decomposition_fixed_periods(self, past_values: int):
    #
    #     past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years=past_values)]
    #
    #     # Удаляем тренд с помощью дифференцирования
    #     detrended = past_years.diff().dropna()
    #     detrended['month'] = detrended.index.month
    #     average_season_k = detrended.groupby('month').mean()
    #     average_season_val = pd.concat([average_season_k] * (len(past_years) // 12), ignore_index=True)
    #
    #     # Из исходного временного ряда вычитаем сезонность, и остается тренд
    #     trend = past_years.reset_index(drop=True) - average_season_val
    #     trend = trend.dropna()
    #
    #     # Прогнозируем тренд
    #     average_step = (detrended.mean())
    #     last_value = trend.iloc[-1]
    #     next_year_trend = [last_value + (i + 1) * average_step for i in range(self.forecast_periods)]
    #
    #     next_year_dates = pd.date_range(
    #         start=past_years.index[-1] + pd.DateOffset(months=1),
    #         periods=self.forecast_periods,
    #         freq='ME'
    #     )
    #     # Преобразуем результат в DataFrame
    #     next_year_trend_df = pd.DataFrame(next_year_trend, index=next_year_dates, columns=self.df.columns)
    #     season_forecast_df = pd.DataFrame(average_season_val.values[:self.forecast_periods], index=next_year_dates,
    #                                       columns=self.df.columns)
    #
    #     # Итоговый прогноз на следующий год (тренд + сезонность)
    #     forecast_df = season_forecast_df + next_year_trend_df
    #
    #     return forecast_df
    #
    #
    # def seasonal_decomposition_calendar_years(self, val: int):
    #     # TODO documentation and add check for number of years (do it later)
    #     last_date = self.df.index.max()
    #     end_year = last_date.year
    #     last_month = last_date.month
    #
    #     # Получаем последние val лет данных
    #     if last_month < 12:
    #         last_years = self.df[
    #             (self.df.index >= f'{end_year - val}-01-01') & (self.df.index < f'{end_year}-01-01')]
    #     else:
    #         last_years = self.df[
    #             (self.df.index >= f'{end_year - (val - 1)}-01-01') & (self.df.index < f'{end_year + 1}-01-01')]
    #
    #     # Удаляем тренд
    #     detrended = last_years.diff().dropna()
    #     detrended['month'] = detrended.index.month
    #     average_season_k = detrended.groupby('month').mean()
    #
    #     # Дублируем сезонность
    #     average_season_val = pd.concat([average_season_k] * (len(last_years) // 12), ignore_index=True)
    #
    #     # Из исходного ряда вычитаем сезонность
    #     trend = last_years.reset_index(drop=True) - average_season_val
    #
    #     # Прогнозируем тренд
    #     average_step = detrended.mean()
    #     last_value = trend.iloc[-1]
    #
    #     next_year_start = last_date + pd.DateOffset(months=1)
    #     months_to_forecast = self.forecast_periods + (last_month if last_month != 12 else 0)
    #
    #     next_year_trend = [last_value + (i + 1) * average_step for i in range(months_to_forecast)]
    #     next_year_dates = pd.date_range(start=next_year_start, periods=months_to_forecast, freq='ME')
    #
    #     next_year_trend_df = pd.DataFrame(next_year_trend, index=next_year_dates, columns=self.df.columns)
    #     season_forecast_df = pd.DataFrame(
    #         average_season_val.values[:months_to_forecast],
    #         index=next_year_dates,
    #         columns=self.df.columns
    #     )
    #     # Корректируем прогнозы для случаев, когда last_month < 12
    #     next_year_dates_fact = pd.date_range(start=next_year_start, periods=self.forecast_periods, freq='ME')
    #     if last_month < 12:
    #         next_year_trend_df = next_year_trend_df.iloc[last_month:].set_index(next_year_dates_fact)
    #         season_forecast_df = season_forecast_df.iloc[last_month:].set_index(next_year_dates_fact)
    #     # Финальный прогноз
    #     final_forecast = next_year_trend_df + season_forecast_df
    #
    #     return final_forecast

    def seasonal_decomposition(self, method: str, past_values: int = 3):
        # TODO documentation

        """
            Декомпозиция временного ряда с выделением тренда и сезонности.
                Args:
                    method: 'fixed_periods' или 'calendar_years', определяет способ обработки данных.
                    past_values: Количество предыдущих фикс.периодов/календарных лет, по которым выполняется прогноз (по умолчанию 3)
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        # Проверка на корректность метода
        if method not in ['fixed_periods', 'calendar_years']:
            raise ValueError("Метод должен быть 'fixed_periods' или 'calendar_years'.")

        # Проверка, достаточно ли данных для анализа
        if len(self.df) < past_values * 12:
            raise ValueError(
                "Недостаточно данных для анализа. Требуется минимум {} месяцев данных.".format(past_values * 12))

        last_date = self.df.index.max()
        end_year = last_date.year
        last_month = last_date.month

        if method == 'fixed_periods':
            past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years=past_values)]

        elif method == 'calendar_years':
            # Получаем последние val лет данных
            if last_month < 12:
                past_years = self.df[
                    (self.df.index >= f'{end_year - past_values}-01-01') & (self.df.index < f'{end_year}-01-01')]
            else:
                past_years = self.df[
                    (self.df.index >= f'{end_year - (past_values - 1)}-01-01') & (
                            self.df.index < f'{end_year + 1}-01-01')]

        # Удаляем тренд с помощью дифференцирования
        detrended = past_years.diff().dropna()
        detrended['month'] = detrended.index.month
        average_season_k = detrended.groupby('month').mean()
        average_season_val = pd.concat([average_season_k] * (len(past_years) // 12), ignore_index=True)

        # Из исходного временного ряда вычитаем сезонность, и остается тренд
        trend = past_years.reset_index(drop=True) - average_season_val
        trend = trend.dropna()

        # Прогнозируем тренд
        average_step = (detrended.mean())
        last_value = trend.iloc[-1]
        next_year_start = last_date + pd.DateOffset(months=1)
        months_to_forecast = self.forecast_periods + (
            last_month if (last_month != 12) & (method == 'calendar_years') else 0)
        next_year_trend = [last_value + (i + 1) * average_step for i in range(months_to_forecast)]
        next_year_dates = pd.date_range(start=next_year_start, periods=months_to_forecast, freq='ME')
        next_year_trend_df = pd.DataFrame(next_year_trend, index=next_year_dates, columns=self.df.columns)
        season_forecast_df = pd.DataFrame(
            average_season_val.values[:months_to_forecast],
            index=next_year_dates,
            columns=self.df.columns
        )
        forecast_df = next_year_trend_df + season_forecast_df

        if method == 'calendar_years':
            # Корректируем прогнозы для случаев, когда last_month < 12
            next_year_dates_fact = pd.date_range(start=next_year_start, periods=self.forecast_periods, freq='ME')
            if last_month < 12:
                next_year_trend_df = next_year_trend_df.iloc[last_month:].set_index(next_year_dates_fact)
                season_forecast_df = season_forecast_df.iloc[last_month:].set_index(next_year_dates_fact)
                # Финальный прогноз
                forecast_df = next_year_trend_df + season_forecast_df

        return forecast_df

    def rolling_mean(self, method: str, past_values: int = 3):
        # TODO documentation
        """
            Декомпозиция временного ряда с применением скользящего среднего.
                Args:
                    method: 'fixed_periods' или 'calendar_years', определяет способ обработки данных.
                    past_values: Количество предыдущих фикс.периодов, по которым выполняется прогноз (по умолчанию 3).
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        # Проверка на корректность метода
        if method not in ['fixed_periods', 'calendar_years']:
            raise ValueError("Метод должен быть 'fixed_periods' или 'calendar_years'.")

        # Проверка, достаточно ли данных для анализа
        if len(self.df) < past_values * 12:
            raise ValueError(
                "Недостаточно данных для анализа. Требуется минимум {} месяцев данных.".format(past_values * 12))

        last_date = self.df.index.max()
        end_year = last_date.year
        last_month = last_date.month

        if method == 'fixed_periods':
            past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years=past_values)]

        elif method == 'calendar_years':

            # Формируем последние три года данных
            if last_month < 12:
                past_years = self.df[(self.df.index >= f'{end_year - past_values}-01-01') &
                                     (self.df.index < f'{end_year}-01-01')]

            else:
                past_years = self.df[(self.df.index >= f'{end_year - (past_values - 1)}-01-01') &
                                     (self.df.index < f'{end_year + 1}-01-01')]

        months_to_forecast = self.forecast_periods + (
            last_month if (last_month != 12) & (method == 'calendar_years') else 0)

        # Удаляем тренд с помощью дифференцирования
        rolling_mean = past_years.rolling(window=13).mean()
        detrended = (past_years - rolling_mean).dropna()
        detrended['month'] = detrended.index.month

        # Прогнозируем сезонность
        aver_season = detrended.groupby('month').mean()
        seasonatily_pred = pd.concat([aver_season] * (len(past_years) // 12), ignore_index=True)
        seasonatily = seasonatily_pred.iloc[:months_to_forecast]

        steps = rolling_mean.diff().dropna()
        average_step = np.abs(steps.mean())
        last_value = rolling_mean.iloc[-1]

        # Прогнозируем скользящее среднее
        next_year_rolling_mean = [last_value + (i + 1) * average_step for i in range(months_to_forecast)]
        next_year_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                        periods=months_to_forecast,
                                        freq='ME'
                                        )
        next_year_rolling_mean_df = pd.DataFrame(
            next_year_rolling_mean,
            index=next_year_dates,
            columns=self.df.columns
        )

        seasonal_forecast = np.tile(seasonatily.values, (1, 1)).T
        # Финальный прогноз: сложение тренда и сезонности
        final_forecast = next_year_rolling_mean_df + seasonal_forecast.T
        forecast_df = pd.DataFrame(final_forecast, index=next_year_dates, columns=self.df.columns)

        if method == 'calendar_years':
            next_year_dates_fact = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                                 periods=self.forecast_periods,
                                                 freq='ME')
            if last_month < 12:
                forecast_df = forecast_df.iloc[last_month:].set_index(next_year_dates_fact)

        return forecast_df

    def decomposition_with_trend_fixed_periods(self, past_values: int = 3, w_1: float = 0.2, w_2: float = 0.8):
        # TODO documentation; с трендом мб только для 3+ лет
        # Проверяем, достаточно ли данных для анализа
        min_required_months = past_values * 12
        if len(self.df) < min_required_months:
            raise ValueError(f"Недостаточно данных для анализа. Требуется минимум {min_required_months} месяцев.")

        # Инициализация переменных
        yearly_means = []
        yearly_normalized = []
        means_df = pd.DataFrame()

        # Формирование данных для каждого периода
        for i in range(past_values):
            start_idx = -(i + 1) * 12
            end_idx = -i * 12 if i > 0 else None
            period_data = self.df.iloc[start_idx:end_idx]

            # Рассчитываем среднее и нормализованные значения
            period_mean = period_data.mean()
            yearly_means.append(period_mean)
            yearly_normalized.append((period_data / period_mean).reset_index(drop=True))

            # Сохраняем средние значения для анализа тренда
            means_df[f'Period_{i + 1}'] = period_mean

        # Среднее нормированных значений для прогнозирования сезонности
        average_normalized = sum(yearly_normalized) / past_values
        average_normalized = pd.concat(
            [average_normalized] * self.forecast_periods,
            ignore_index=True
        ).iloc[:self.forecast_periods]

        # Среднее всех значений для прогнозирования тренда
        overall_mean = sum(yearly_means) / past_values

        # Прогнозируем тренд в зависимости от количества периодов
        if past_values in [1, 2]:
            # Прогноз на основе среднего по всем периодам
            forecast_average = overall_mean
        else:
            # Рассчитываем шаги для прогнозирования тренда
            total_step = sum(
                (means_df[f'Period_{i - 1}'] - means_df[f'Period_{i}']) * (w_2 if i == 2 else w_1)
                for i in range(2, past_values + 1)
            )
            forecast_average = total_step + means_df['Period_1'].values

        # Итоговый прогноз: тренд * сезонность
        result_df = average_normalized * forecast_average

        # Формируем временной индекс для прогноза
        new_index = pd.date_range(start=self.df.index.max() + pd.DateOffset(months=1),
                                  periods=self.forecast_periods,
                                  freq='ME')
        result_df.index = new_index

        return result_df

    def decomposition_with_trend_calendar_years(self, past_values: int = 3, w_1: float = 0.2, w_2: float = 0.8):
        #TODO documentation; с трендом мб только для 3+ лет

        if self.df.index.year.max() - past_values + 1 not in self.df.index.year.unique():
            raise ValueError(f"Недостаточно данных для анализа {past_values} периодов.")

        # Инициализация переменных
        last_date = self.df.index.max()
        last_month = last_date.month
        yearly_means, yearly_normalized = [], []
        means_df = pd.DataFrame()

        current_year = self.df.index.year.max()
        start_year = current_year
        range_start = past_values - 1 if last_month == 12 else past_values
        range_end = -1 if last_month == 12 else 0

        for i in range(range_start, range_end, -1):
            year = start_year - i
            period_data = self.df[self.df.index.year == year]

            # Среднее и нормализация
            period_mean = period_data.mean()
            yearly_means.append(period_mean)
            yearly_normalized.append((period_data / period_mean).reset_index(drop=True))
            if last_month == 12:
                means_df[f'Period_{i + 1}'] = period_mean
            else:
                means_df[f'Period_{i}'] = period_mean
        months_to_forecast = self.forecast_periods + (last_month if (last_month != 12) else 0)
        # Среднее нормализованное значение
        average_normalized = sum(yearly_normalized) / past_values
        average_normalized = pd.concat([average_normalized] * self.forecast_periods, ignore_index=True).iloc[
                             :months_to_forecast]

        # Общее среднее значение
        overall_mean = sum(yearly_means) / past_values

        # Расчёт прогноза тренда
        if past_values <= 2:
            forecast_average = overall_mean
            result_df = average_normalized * forecast_average
        else:
            steps = [
                (means_df[f'Period_{i}'] - means_df[f'Period_{i + 1}']) * (w_2 if i == 1 else w_1)
                for i in range(1, past_values)
            ]
            forecast_average = sum(steps) + means_df['Period_1']
            result_dff = average_normalized * forecast_average.values

            # Дополнительная корректировка
            adjustment = (
                    (means_df['Period_1'] - means_df['Period_2']) * w_1 +
                    (result_dff.mean() - means_df['Period_1']) * w_2
            )
            forecast_average_adjusted = adjustment + result_dff.mean()
            result_df_dop = forecast_average_adjusted * average_normalized
            result_df = pd.concat([result_dff, result_df_dop], axis=0)

        if last_month != 12:
            # Прогноз для оставшихся месяцев текущего года
            result_df_forecast = result_df.iloc[last_month:months_to_forecast]
            new_index_current_year = pd.date_range(
                start=last_date + pd.DateOffset(months=1), periods=self.forecast_periods, freq='ME'
            )
            result_df_forecast.index = new_index_current_year

        else:
            # Прогноз на 12 месяцев, начиная с января
            result_df_forecast = result_df.iloc[:months_to_forecast]
            new_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_periods,
                freq='ME'
            )
            result_df_forecast.index = new_index

        return result_df_forecast

    def decomposition_without_trend_fixed_periods(self, past_values: int = 3):
        #TODO documentation
        '''
            Создает новый DataFrame с спрогнозированными значениями.
            Прогноз значений выполняется на основе указанного количества предыдущих периодов (один период равен 12 месяцев),
            путем выделения из этих периодов сезонности, считая, что тренд отсутствует.

            Сезонность рассчитывается как среднее нормированных значений за указанные периоды.

            :param df: Исходный DataFrame
            :param years_count: Количество предыдущих периодов для анализа (1, 2 или 3)
            :return:  Новый DataFrame, список рассатриваемых исторических значений
            '''

        # Проверка, что данных достаточно
        if len(self.df) < 12 * past_values:
            raise ValueError("В DataFrame недостаточно данных для указанного количества периодов.")

        yearly_means = []
        yearly_normalized = []

        # Периоды для анализа
        for i in range(past_values):
            start_idx = -(i + 1) * 12
            end_idx = -i * 12 if i > 0 else None
            period_data = self.df.iloc[start_idx:end_idx]

            # Рассчитываем среднее и нормализованные значения
            period_mean = period_data.mean()
            yearly_means.append(period_mean)
            yearly_normalized.append((period_data / period_mean).reset_index(drop=True))

        average_normalized = sum(yearly_normalized) / past_values
        average_normalized = pd.concat(
            [average_normalized] * self.forecast_periods,
            ignore_index=True
        ).iloc[:self.forecast_periods]

        # Среднее всех значений для прогнозирования тренда
        overall_mean = sum(yearly_means) / past_values

        # Создание DataFrame из умноженных значений
        result_df = average_normalized * overall_mean

        # Установка правильного индекса для новых данных
        new_index = pd.date_range(start=self.df.index.max() + pd.DateOffset(months=1),
                                  periods=self.forecast_periods,
                                  freq='ME')
        result_df.index = new_index
        return result_df

    def decomposition_without_trend_calendar_years(self, past_values: int = 3):
        # TODO documentation
        if self.df.index.year.max() - past_values + 1 not in self.df.index.year.unique():
            raise ValueError(f"Недостаточно данных для анализа {past_values} периодов.")

        # Получение текущего года и последнего месяца исходного ряда
        last_date = self.df.index.max()
        last_month = last_date.month
        yearly_means, yearly_normalized = [], []

        current_year = self.df.index.year.max()
        start_year = current_year
        range_start = past_values - 1 if last_month == 12 else past_values
        range_end = -1 if last_month == 12 else 0

        for i in range(range_start, range_end, -1):
            year = start_year - i
            period_data = self.df[self.df.index.year == year]

            # Среднее и нормализация
            period_mean = period_data.mean()
            yearly_means.append(period_mean)
            yearly_normalized.append((period_data / period_mean).reset_index(drop=True))

        # Расчёт нормализованного среднего для прогнозного периода
        average_normalized = pd.concat(yearly_normalized).groupby(level=0).mean()
        average_normalized = pd.concat([average_normalized] * self.forecast_periods, ignore_index=True)

        # Расчёт общего среднего
        overall_mean = sum(yearly_means) / past_values

        # Расчёт прогноза
        result_df = average_normalized * overall_mean
        months_to_forecast = self.forecast_periods + (last_month if (last_month != 12) else 0)

        if last_month != 12:
            # Прогноз для оставшихся месяцев текущего года
            result_df_forecast = result_df.iloc[last_month:months_to_forecast]
            new_index_current_year = pd.date_range(
                start=last_date + pd.DateOffset(months=1), periods=self.forecast_periods, freq='ME'
            )
            result_df_forecast.index = new_index_current_year

        else:
            # Прогноз на 12 месяцев, начиная с января
            result_df_forecast = result_df.iloc[:months_to_forecast]
            new_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_periods,
                freq='ME'
            )
            result_df_forecast.index = new_index


        return result_df_forecast





    # def auto_arima_forecast(self, plot=False):
    #
    # def prophet_forecast(self):