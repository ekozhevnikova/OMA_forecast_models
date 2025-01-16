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
from preprocessing import *



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
        """
            Наивный метод прогнозирования с учетом ошибки (опционально).
            Используется для прогноза ВР без тренда и сезонности.
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

    def seasonal_decomposition(self, method: str, past_values: int = 3):
        """
            Декомпозиция временного ряда с выделением тренда и сезонности.
            Используется для прогноза ВР с сезонностью и неявно выраженным трендом.

                Из временного ряда выделяются данные за последние past_values фиксированных периода(период = 12 мес)/
            календарных года. Удаляется тренд с помощью дифференцирования временного ряда. Сезонность рассчитывается как
            среднее значение разностей за последние past_values. Из исходного временного ряда вычитается рассчитанная
            сезонность и остается тренд.
                Прогнозируются тренд и сезонность. Финальный прогноз на следующий год строится как сумма прогнозов
            тренда и сезонности.

                Args:
                    method: 'fixed_periods' или 'calendar_years', определяет способ обработки данных.
                    past_values: Количество предыдущих фикс.периодов (период = 12 мес)/календарных лет, по которым
                    выполняется прогноз (по умолчанию 3).
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

        preprocessing = Preprocessing(self.df)
        end_year, last_month, last_date = preprocessing.search_last_fact_data()

        if method == 'fixed_periods':
            past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years=past_values)]

        elif method == 'calendar_years':
            # Получаем последние past_values лет данных
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
        """
            Декомпозиция временного ряда с применением скользящего среднего.
            Используется для прогноза ВР с неявно выраженным трендом и сезонностью.

                Из временного ряда выделяются данные за последние past_values фиксированных периода(период = 12 мес)/
            календарных года. Применяется скользящее среднее с окном в 12 месяцев для устранения тренда, после чего
            рассчитывается остаток(detrended). Сезонность рассчитывается на основе данных за последние (past_values-1).
                Прогнозируется сезонность и тренд. Финальный прогноз строится как сумма прогнозируемого
            тренда и сезонности.
                Args:
                    method: 'fixed_periods' или 'calendar_years', определяет способ обработки данных.
                    past_values: Количество предыдущих фикс.периодов (период = 12 мес)/календарных лет, по которым
                    выполняется прогноз (по умолчанию 3).
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

        preprocessing = Preprocessing(self.df)
        end_year, last_month, last_date = preprocessing.search_last_fact_data()

        if method == 'fixed_periods':
            past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years=past_values)]

        elif method == 'calendar_years':

            # Получаем последние past_values лет данных
            if last_month < 12:
                past_years = self.df[(self.df.index >= f'{end_year - past_values}-01-01') &
                                     (self.df.index < f'{end_year}-01-01')]

            else:
                past_years = self.df[(self.df.index >= f'{end_year - (past_values - 1)}-01-01') &
                                     (self.df.index < f'{end_year + 1}-01-01')]

        months_to_forecast = self.forecast_periods + (
            last_month if (last_month != 12) & (method == 'calendar_years') else 0)

        # Удаляем тренд с помощью дифференцирования
        rolling_mean = past_years.rolling(window=12).mean()
        detrended = (past_years - rolling_mean).dropna()
        detrended['month'] = detrended.index.month

        # Прогнозируем сезонность
        aver_season = detrended.groupby('month').mean()
        seasonatily_pred = pd.concat([aver_season] * (len(past_years) // 12), ignore_index=True)
        seasonatily = seasonatily_pred.iloc[:months_to_forecast]

        steps = rolling_mean.diff().dropna()
        average_step = steps.mean()
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

        seasonal_forecast = np.tile(seasonatily.values, (1, 1))
        # Финальный прогноз: сложение тренда и сезонности
        final_forecast = next_year_rolling_mean_df + seasonal_forecast
        forecast_df = pd.DataFrame(final_forecast, index=next_year_dates, columns=self.df.columns)

        if method == 'calendar_years':
            next_year_dates_fact = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                                 periods=self.forecast_periods,
                                                 freq='ME')
            if last_month < 12:
                forecast_df = forecast_df.iloc[last_month:].set_index(next_year_dates_fact)

        return forecast_df

    def decomposition_fixed_periods(self, method: str, past_values: int = 3, w_1: float = 0.2, w_2: float = 0.8):
        """
            Ручная декомпозиция временного ряда по фиксированным периодам.
            Используется для прогноза ВР с трендом (если указан method: 'with_trend') и сезонностью.

                Прогноз значений выполняется на основе указанного количества предыдущих периодов (период=12 мес),
            путем выделения из этих периодов тренда (при method: 'with_trend') и сезонности.

                Будущее значение тренда на основе 3-х и более прошлых периодов рассчитывается, как сумма предыдущего
            значения и средней разницы между средними значениями периодов (где разность между средними значениями
            для разных периодов имеет разные весовые коэффициенты).
                Для прогноза на основе одного и двух прошлых периодов, считается, что тренда нет.

                Сезонность рассчитывается как среднее нормированных значений за указанные периоды.
                При method: 'without_trend' весовые коэффициенты не используются.

                Args:
                    method: 'with_trend' или 'without_trend', выбирается в зависимости от наличия/отсутствия тренда в
                    исходном ВР.
                    past_values: Количество предыдущих фиксированных периодов (период = 12 мес), по которым выполняется
                    прогноз (по умолчанию 3).
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        if method not in ['with_trend', 'without_trend']:
            raise ValueError("Метод должен быть 'with_trend' или 'without_trend'.")

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
        if method == 'with_trend':
            # Прогнозируем тренд в зависимости от количества периодов
            if past_values in [1, 2]:
                # Прогноз на основе среднего по всем периодам
                forecast_average = overall_mean
            else:
                # Рассчитываем шаги для прогнозирования тренда
                total_step = sum(
                    (means_df[f'Period_{i - 1}'] - means_df[f'Period_{i}']) * (w_2 if i == 2 else w_1/(past_values - 2))
                    for i in range(2, past_values + 1)
                )
                forecast_average = total_step + means_df['Period_1'].values

        # Итоговый прогноз: тренд * сезонность
            result_df = average_normalized * forecast_average

        elif method == 'without_trend':
            result_df = average_normalized * overall_mean

        # Формируем временной индекс для прогноза
        new_index = pd.date_range(start=self.df.index.max() + pd.DateOffset(months=1),
                                  periods=self.forecast_periods,
                                  freq='ME')
        result_df.index = new_index

        return result_df


    def decomposition_calendar_years(self, method: str, past_values: int = 3, w_1: float = 0.2, w_2: float = 0.8):
        """
            Ручная декомпозиция временного ряда по календарным годам.
            Используется для прогноза ВР с трендом (если указан method: 'with_trend') и сезонностью.

                Прогноз значений выполняется на основе указанного количества предыдущих календарных лет, путем выделения
            тренда (при method: 'with_trend') и сезонности.

                Будущее значение тренда на основе 3-х и более прошлых лет рассчитывается, как сумма предыдущего
            значения и средней разницы между средними значениями лет (где разность между средними значениями
            для разных лет имеет разные весовые коэффициенты).
                Для прогноза на основе одного и двух прошлых лет, считается, что тренда нет.

                Сезонность рассчитывается как среднее нормированных значений за указанные годы.
                При method: 'without_trend' весовые коэффициенты не используются.

                Args:
                    method: 'with_trend' или 'without_trend', выбирается в зависимости от наличия/отсутствия тренда в
                    исходном ВР.
                    past_values: Количество предыдущих фиксированных периодов (период = 12 мес), по которым выполняется
                    прогноз (по умолчанию 3).
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        if self.df.index.year.max() - past_values + 1 not in self.df.index.year.unique():
            raise ValueError(f"Недостаточно данных для анализа {past_values} периодов.")

        # Инициализация переменных
        preprocessing = Preprocessing(self.df)
        end_year, last_month, last_date = preprocessing.search_last_fact_data()
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
        if method == 'with_trend':
            # Расчёт прогноза тренда
            if past_values <= 2:
                forecast_average = overall_mean
                result_df = average_normalized * forecast_average
            else:
                steps = [
                    (means_df[f'Period_{i}'] - means_df[f'Period_{i + 1}']) * (w_2 if i == 1 else w_1/(past_values - 2))
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

        elif method == 'without_trend':
            result_df = average_normalized * overall_mean

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

    def prophet_forecast(self):
        """
            Метод PROPHET.
            Универсальный для всех ВР.
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        df = self.df.copy()
        forecast_periods = self.forecast_periods

        # Параметры для перебора
        param_grid = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [12, 18, 24],
            'changepoint_prior_scale': [0.01, 0.05, 0.1]
        }

        # Сортировка и подготовка данных
        df.reset_index(inplace=True)
        df[self.column_name_with_date] = pd.to_datetime(df[self.column_name_with_date])
        df = df.sort_values(by=self.column_name_with_date)

        # Разделение на обучающую и тестовую выборки
        train = df.iloc[:-forecast_periods]
        test = df.iloc[-forecast_periods:]

        # Список временных рядов для прогнозирования
        series_list = [col for col in train.columns if col != self.column_name_with_date]

        results = []  # Результаты параметров для каждого временного ряда

        for series in series_list:
            # Подготовка данных для Prophet
            series_df = train[[self.column_name_with_date, series]].rename(
                columns={self.column_name_with_date: 'ds', series: 'y'})

            best_mape = float('inf')
            best_params = None

            for params in ParameterGrid(param_grid):
                model = Prophet(seasonality_mode=params['seasonality_mode'],
                                n_changepoints=params['n_changepoints'],
                                changepoint_prior_scale=params['changepoint_prior_scale'])
                model.fit(series_df)

                # Прогнозирование
                future = model.make_future_dataframe(periods=forecast_periods, freq='M')
                forecast = model.predict(future)

                # Вычисление MAPE
                test_series = test[[self.column_name_with_date, series]].rename(
                    columns={self.column_name_with_date: 'ds', series: 'y'})
                forecast_test_period = forecast[-forecast_periods:]

                # Сравнение только по датам тестового набора
                aligned_forecast = forecast_test_period.set_index('ds').reindex(test_series['ds']).dropna()
                aligned_test = test_series.set_index('ds').reindex(aligned_forecast.index).dropna()

                mape = np.mean(np.abs((aligned_forecast['yhat'] - aligned_test['y']) / aligned_test['y'])) * 100

                if mape < best_mape:
                    best_mape = mape
                    best_params = params

            results.append((series, best_params, best_mape))
            print(f"Лучшие параметры для {series}: {best_params}, MAPE: {best_mape:.2f}")

        # Прогнозирование для полного набора данных
        forecast_df = pd.DataFrame()
        for series, best_params, _ in results:
            series_df = df[[self.column_name_with_date, series]].rename(
                columns={self.column_name_with_date: 'ds', series: 'y'})

            model = Prophet(seasonality_mode=best_params['seasonality_mode'],
                            n_changepoints=best_params['n_changepoints'],
                            changepoint_prior_scale=best_params['changepoint_prior_scale'])

            model.fit(series_df)
            future = model.make_future_dataframe(periods=forecast_periods, freq='M')
            forecast = model.predict(future)

            if forecast_df.empty:
                forecast_df['ds'] = forecast['ds']
            forecast_df[series] = forecast['yhat']

        forecast_df.set_index('ds', inplace=True)
        forecast_df = forecast_df.tail(forecast_periods)
        return forecast_df

    def auto_arima_forecast(self):
        """
            Метод auto_arima.
            Универсальный для всех ВР.

                Параметры модели ARIMA:
                    - seasonal (bool): Определяет, учитывать ли сезонность временного ряда при построении модели.
                    - D (int): Порядок сезонного дифференцирования. Это количество раз, которое сезонные данные
                    дифференцируются, чтобы устранить сезонность и сделать временной ряд стационарным.
                    - m (int): Частота сезонности - период, через который повторяются сезонные колебания.
                    - stationary (bool): Указывает, является ли временной ряд стационарным.
                    - test (str): Определяет тест для проверки стационарности и выбора порядка интеграции (d):
                        adf: Тест Дики-Фуллера.
                        pp: Тест Филлипса-Перрона.
                        kpss: Тест KPSS.
                    - information_criterion (str): Критерий для выбора лучшей модели ARIMA. Основные варианты:
                        aic (информационный критерий Акаике): Хорош для моделей с большим числом параметров.
                        bic (байесовский информационный критерий): Предпочтителен, если данных мало.
                        hqic (критерий Ханна-Куинна): Балансирует между AIC и BIC, рекомендуется для сложных моделей.
                    - stepwise (bool): Включает пошаговый (итеративный) поиск наилучшей комбинации параметров (p, d, q).
                    - suppress_warnings (bool): Подавление предупреждений во время обучения модели.
                    - max_p (int): Максимальный порядок авторегрессии (p). Этот параметр определяет, сколько прошлых
                    значений временного ряда используется для предсказания текущего значения.
                    - max_q (int): Максимальный порядок скользящей средней (q). Это количество прошлых ошибок прогноза,
                    используемых для корректировки текущего предсказания.
                    - max_d (int): Максимальное количество дифференцирований для достижения стационарности.
                    - max_P (int): Максимальный сезонный порядок авторегрессии (P).
                    - max_Q (int): Максимальный сезонный порядок скользящей средней (Q).
                    - max_D (int): Максимальный порядок сезонного дифференцирования (D).
                    Используется для устранения сезонных трендов.
                    - trace (bool): Показ выводов в процессе подбора параметров.

                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        forecasts = {}

        for channel in self.df.columns:
            ts = self.df[channel].dropna()
            original_series = ts.copy()
            was_non_stationary = False

            # Проверка на стационарность
            if not Preprocessing.check_stationarity(ts):
                ts = Preprocessing.make_stationary(ts)
                was_non_stationary = True

            try:
                # Построение модели
                model = auto_arima(
                    ts,
                    seasonal=True,                      # Использовать сезонность
                    D=1,                                # Порядок сезонного дифференцирования
                    m=12,                               # Частота сезонности (12 месяцев)
                    stationary=False,                   # Определение стационарности
                    test='pp',                          # Тест на стационарность
                    information_criterion='hqic',       # Критерий для выбора лучшей модели
                    stepwise=True,                      # Пошаговый подбор параметров
                    suppress_warnings=False,            # Подавление предупреждений
                    max_p=5, max_q=5, max_d=1,          # Максимальные значения p, q, d
                    max_P=2, max_Q=2, max_D=1,          # Максимальные значения P, Q, D
                    trace=False                         # Вывод логов
                )

                # Обучение модели
                model.fit(ts)
                forecast = model.predict(n_periods=self.forecast_periods)

                # Преобразование прогноза в исходный масштаб
                if was_non_stationary:
                    last_observation = original_series.iloc[-1]
                    forecast = Preprocessing.inverse_difference(pd.Series(forecast), last_observation)

                forecasts[channel] = forecast

            except Exception as e:
                print(f"Error processing channel {channel}: {e}")
                forecasts[channel] = np.nan

        # Формирование DataFrame с прогнозом
        forecast_df = pd.DataFrame(
            forecasts,
            index=pd.date_range(start=self.df.index[-1] + pd.DateOffset(months=1), periods=self.forecast_periods,
                                freq='ME')
        )

        return forecast_df







