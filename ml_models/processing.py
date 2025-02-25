import sys
import numpy as np
import pandas as pd
import threading
from threadpoolctl import ThreadpoolController
from pmdarima import auto_arima
import pymannkendall as mk
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from scipy import linalg
from contextlib import contextmanager
import logging
import os
from OMA_tools.ml_models.preprocessing import Preprocessing
from OMA_tools.ml_models.postprocessing import Postprocessing



import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("cmdstanpy").disabled = True


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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
    
    
    def regression_model(self, method: str):
        """
            Модель Регрессии

                Args:
                    method: Линейный тренд (linear_trend) или Логистический тренд (logistic_trend)
                    plots: Построение графиков (по умолчанию False)
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        # Проверка на корректность метода
        if method not in ['linear_trend', 'logistic_trend']:
            raise ValueError("Метод должен быть 'linear_trend' или 'logistic_trend'.")

        data_copy = self.df.copy()
        # Сброс индекса и добавление названий месяцев
        data_copy.reset_index(inplace = True)
        #Предварительная кодировка в признак
        data_copy['month_name'] = data_copy[self.column_name_with_date].dt.strftime("%B") #преобразование даты ('2020-01-01') в текстовый формат типа 'Январь'

        #С текстовыми данными работать не можем => применяется OneHotEncoder кодировщик для преобразования категориальных или текстовых данных в числа
        #Числа заменяются на единицы и нули, в зависимости от того, какому столбцу какое значение присуще.
        encoder = OneHotEncoder(categories = 'auto', drop = 'first', sparse_output = False)
        encoded_months = encoder.fit_transform(data_copy[['month_name']])#конвертация в массив закодированного столбца

        # Матрица сезонных признаков
        encoded_df_0 = pd.DataFrame(encoded_months, columns = encoder.get_feature_names_out(['month_name']))

        # Колонка с трендом (наклон)
        if method == 'linear_trend':
            encoded_df_trend = pd.DataFrame({'Linear_Trend': np.arange(1, data_copy.shape[0] + 1)})
        elif method == 'logistic_trend':
            encoded_df_trend = pd.DataFrame({'Log_Trend': np.log(np.arange(1, data_copy.shape[0] + 1))})
        else:
            raise ValueError('Неверно выбран тип тренда.')

        # Свободный член в модели регресии (интерсепт)
        encoded_df_free_var = pd.DataFrame({'free_variable': np.ones(data_copy.shape[0])})

        # Итоговая матрица признаков (сезонность, тренд, интерсепт)
        encoded_df = pd.concat([encoded_df_0, encoded_df_trend, encoded_df_free_var], axis = 1)

        # Словарь для хранения коэффициентов модели для каждого столбца
        model_coefficients_dict = {}

        # Прогнозирование для каждого столбца
        for column in data_copy.columns[1:-1]:  # Пропускаем столбцы column_name_with_date и "month_name"
            A = encoded_df.values
            b = data_copy[column].values.reshape((int(encoded_df.shape[0]), 1))

            # Решаем систему уравнений с помощью метода наименьших квадратов
            k, *_ = linalg.lstsq(A, b)

            # Переводим коэффициенты в список и сохраняем в словарь
            model_coefficients = k.reshape((1, encoded_df.shape[1])).tolist()[0]
            model_coefficients_dict[column] = model_coefficients

        # Прогнозирование на N месяцев вперед
        # Определяем последний месяц в данных
        last_date = data_copy[self.column_name_with_date].max()

        # Создаем новые даты для следующего года
        future_dates = pd.date_range(last_date + pd.DateOffset(months = 1), periods = self.forecast_periods, freq='MS')

        # Создаем DataFrame для будущих дат
        future_df = pd.DataFrame({self.column_name_with_date: future_dates})
        future_df['month_name'] = future_df[self.column_name_with_date].dt.strftime("%B")

        # Преобразуем названия месяцев в бинарные признаки (One-Hot Encoding)
        encoded_future_months = encoder.transform(future_df[['month_name']])
        encoded_future_df_0 = pd.DataFrame(encoded_future_months, columns = encoder.get_feature_names_out(['month_name']))

        # Тренд для новых дат
        if method == 'linear trend':
            encoded_future_df_trend = pd.DataFrame(
                {'Linear_Trend': np.arange(len(data_copy) + 1, len(data_copy) + (self.forecast_periods + 1))})
        else:
            encoded_future_df_trend = pd.DataFrame(
                {'Log_Trend': np.log(np.arange(len(data_copy) + 1, len(data_copy) + (self.forecast_periods + 1)))})

        # Свободный член (интерсепт)
        encoded_future_df_free_var = pd.DataFrame({'free_variable': np.ones(self.forecast_periods)})

        # Итоговая матрица признаков для будущих дат
        encoded_future_df = pd.concat([encoded_future_df_0, encoded_future_df_trend, encoded_future_df_free_var], axis = 1)

        # Новый DataFrame для хранения прогнозируемых значений на следующий год
        future_predictions = pd.DataFrame({self.column_name_with_date: future_df[self.column_name_with_date]})

        # Прогнозирование для каждого столбца на следующий год
        for column in data_copy.columns[1:-1]:  # Пропускаем столбцы column_name_with_date и "month_name"
            model_coefficients = model_coefficients_dict[column]

            y_future_pred = [
                sum(np.multiply(encoded_future_df.iloc[i, :].tolist(), model_coefficients))
                for i in range(encoded_future_df.shape[0])
            ]

            # Добавляем прогнозируемые значения в DataFrame
            future_predictions[f'{column}'] = y_future_pred
        future_predictions.set_index(self.column_name_with_date, inplace = True)

        #Замена отрицательных значений на минимально возможное в колонке
        for column in future_predictions.columns:
            for i in range(len(list(future_predictions[column]))):
                if list(future_predictions[column])[i] < 0:
                    values = list(self.df[column])
                    min_value = min(values)
                    list(future_predictions[column])[i] = min_value
        return future_predictions


    def naive_forecast(self, past_values: int = 3, weigh_error: float = 0):
        """
            Наивный метод прогнозирования с учетом ошибки (опционально).
            Используется для прогноза ВР без тренда и сезонности.

                Из исходного ВР выделяются past_values последних месяцев, заданных пользователем, и рассчитывается их
            среднее значение. При weigh_error > 0, последовательно для каждого месяца, вычисляется ошибка, т.е. отклонение
            фактического значения от среднего и корректируется рассчитанное среднее, путем прибавления к нему полученной
            ошибки с заданным весом (weigh_error).

                Args:
                    past_values: Количество предыдущих месяцев, по которым выполняется прогноз (по умолчанию 3).
                    weigh_error: Вес, с которым учитывается ошибка [0:1].
                                        Если 0, то используется простой наивный метод (по умолчанию 0).
                    plots: Построение графиков (по умолчанию False)

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
                index = pd.date_range(
                    start = self.df.index[-1] + pd.DateOffset(months = 1),
                    periods = self.forecast_periods,
                    freq = 'MS'
                )
            )

            forecast_results[column] = forecast_df
        # Объединение всех столбцов в один DataFrame
        result_df = pd.concat(forecast_results.values(), axis = 1)
        # Введение названия столбца с индексом для столбца с датами
        result_df.reset_index(inplace=True)
        result_df.rename(columns = {result_df.columns[0]: self.column_name_with_date}, inplace=True)
        result_df.set_index(self.column_name_with_date, inplace = True)

        return result_df


    def seasonal_decomposition(self, method: str, past_values: int = 3):
        """
            Декомпозиция временного ряда с выделением тренда и сезонности.
            Используется для прогноза ВР с сезонностью и неявно выраженным трендом.

                Из временного ряда выделяются данные за последние past_values фиксированных периодов (период=12 мес.)/
            календарных лет. Путем дифференцирования ВР удаляется тренд и остается сезонная составляющая.
                Сезонность рассчитывается как среднее между сезонными составляющими ВР за каждый период. Из исходного
            временного ряда вычитается рассчитанная сезонность и остается тренд.
                Для прогноза тренда вычисляется шаг изменения исторических значений тренда на каждом месяце и
            усредняется. Далее это усредненное значение шага прибавляется к последнему известному значению тренда и
            итерация повторяется столько раз, сколько месяцев необходимо спрогнозировать.
                Финальный прогноз строится как сумма прогноза тренда и рассчитанной сезонности.

                Args:
                    method: 'fixed_periods' или 'calendar_years', определяет способ обработки данных.
                    past_values: Количество предыдущих фикс.периодов (период = 12 мес)/календарных лет, по которым
                    выполняется прогноз (по умолчанию 3).
                    plots: Построение графиков (по умолчанию False)

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
            past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years = past_values)]

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
        average_season_val = pd.concat([average_season_k] * (len(past_years) // 12), ignore_index = True)

        # Из исходного временного ряда вычитаем сезонность, и остается тренд
        trend = past_years.reset_index(drop = True) - average_season_val
        trend = trend.dropna()

        # Прогнозируем тренд
        average_step = (detrended.mean())
        last_value = trend.iloc[-1]
        next_year_start = last_date + pd.DateOffset(months = 1)
        months_to_forecast = self.forecast_periods + (
            last_month if (last_month != 12) & (method == 'calendar_years') else 0)
        next_year_trend = [last_value + (i + 1) * average_step for i in range(months_to_forecast)]
        next_year_dates = pd.date_range(start = next_year_start, periods = months_to_forecast, freq = 'MS')
        next_year_trend_df = pd.DataFrame(next_year_trend, index = next_year_dates, columns = self.df.columns)
        season_forecast_df = pd.DataFrame(
            average_season_val.values[:months_to_forecast],
            index = next_year_dates,
            columns = self.df.columns
        )
        forecast_df = next_year_trend_df + season_forecast_df

        # Введение названия столбца с индексом для столбца с датами
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns = {forecast_df.columns[0]: self.column_name_with_date}, inplace=True)
        forecast_df.set_index(self.column_name_with_date, inplace = True)

        #Замена отрицательных значений в столбце на минимально возможное
        for column in forecast_df.columns:
            for i in range(len(list(forecast_df[column]))):
                if list(forecast_df[column])[i] < 0:
                    values = list(self.df[column])
                    min_value = min(values)
                    list(forecast_df[column])[i] = min_value

        if method == 'calendar_years':
            # Корректируем прогнозы для случаев, когда last_month < 12
            next_year_dates_fact = pd.date_range(start = next_year_start, periods = self.forecast_periods, freq = 'MS')
            if last_month < 12:
                next_year_trend_df = next_year_trend_df.iloc[last_month:].set_index(next_year_dates_fact)
                season_forecast_df = season_forecast_df.iloc[last_month:].set_index(next_year_dates_fact)
                # Финальный прогноз
                forecast_df = next_year_trend_df + season_forecast_df
                # Введение названия столбца с индексом для столбца с датами
                forecast_df.reset_index(inplace=True)
                forecast_df.rename(columns = {forecast_df.columns[0]: self.column_name_with_date}, inplace=True)
                forecast_df.set_index(self.column_name_with_date, inplace = True)
                #Замена отрицательных значений в столбце на минимально возможное
                for column in forecast_df.columns:
                    for i in range(len(list(forecast_df[column]))):
                        if list(forecast_df[column])[i] < 0:
                            values = list(self.df[column])
                            min_value = min(values)
                            list(forecast_df[column])[i] = min_value
        return forecast_df


    def rolling_mean(self, method: str, past_values: int = 3):
        """
            Декомпозиция временного ряда с применением скользящего среднего.
            Используется для прогноза ВР с неявно выраженным трендом и сезонностью.

                Из временного ряда выделяются данные за последние past_values фиксированных периодов(период=12 мес.)/
            календарных лет. Применяется скользящее среднее с окном в 12 месяцев и выделяется сезонная составляющая,
            путем расчета разности между историческими значениями и скользящим средним.
                Прогноз сезонности рассчитывается как среднее между сезонными составляющими ВР за каждый период.
                Для прогноза тренда вычисляется шаг изменения значений скользящего среднего на каждом месяце и
            усредняется. Далее это усредненное значение шага прибавляется к последнему известному значению скользящего
            среднего и итерация повторяется столько раз, сколько месяцев необходимо спрогнозировать.
                Финальный прогноз строится как сумма прогноза тренда и сезонности.

                Args:
                    method: 'fixed_periods' или 'calendar_years', определяет способ обработки данных.
                    past_values: Количество предыдущих фикс.периодов (период = 12 мес)/календарных лет, по которым
                    выполняется прогноз (по умолчанию 3).
                    plots: Построение графиков (по умолчанию False)

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
            past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years = past_values)]

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
        #detrended[self.column_name_with_date] = detrended.index.month

        # Прогнозируем сезонность
        aver_season = detrended.groupby(self.column_name_with_date).mean()
        seasonatily_pred = pd.concat([aver_season] * (len(past_years) // 12), ignore_index = True)
        seasonatily = seasonatily_pred.iloc[:months_to_forecast]

        steps = rolling_mean.diff().dropna()
        average_step = steps.mean()
        last_value = rolling_mean.iloc[-1]

        # Прогнозируем скользящее среднее
        next_year_rolling_mean = [last_value + (i + 1) * average_step for i in range(months_to_forecast)]
        next_year_dates = pd.date_range(start = last_date + pd.DateOffset(months = 1),
                                        periods = months_to_forecast,
                                        freq = 'MS'
                                        )
        next_year_rolling_mean_df = pd.DataFrame(
            next_year_rolling_mean,
            index = next_year_dates,
            columns = self.df.columns
        )

        seasonal_forecast = np.tile(seasonatily.values, (1, 1))
        # Финальный прогноз: сложение тренда и сезонности
        final_forecast = next_year_rolling_mean_df + seasonal_forecast
        forecast_df = pd.DataFrame(final_forecast, index=next_year_dates, columns=self.df.columns)
        # Введение названия столбца с индексом для столбца с датами
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns = {forecast_df.columns[0]: self.column_name_with_date}, inplace=True)
        forecast_df.set_index(self.column_name_with_date, inplace = True)

        if method == 'calendar_years':
            next_year_dates_fact = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                                 periods=self.forecast_periods,
                                                 freq='MS')
            if last_month < 12:
                forecast_df.iloc[last_month:].set_index(next_year_dates_fact, inplace=True)
                # Введение названия столбца с индексом для столбца с датами
                forecast_df.reset_index(inplace=True)
                forecast_df.rename(columns = {forecast_df.columns[0]: self.column_name_with_date}, inplace=True)
                forecast_df.set_index(self.column_name_with_date, inplace = True)

        return forecast_df


    def decomposition_fixed_periods(self, method: str, past_values: int = 3, w_1: float = 0.2, w_2: float = 0.8):
        """
            Ручная декомпозиция временного ряда по фиксированным периодам.
            Используется для прогноза ВР с трендом (если указан method: 'with_trend') и сезонностью.

                Из временного ряда выделяются данные за последние past_values фиксированных периодов (период=12 мес.).
            При прогнозировании ВР без тренда:
                Для каждого из выделенных периодов считается среднее значение и данный период делится на это значение,
            т.е. выполняется нормировка. Полученные нормированные периоды усредняются и умножаются на среднее значение
            рассматриваемой части ВР.

            При прогнозировании ВР без тренда:
                Сезонность прогнозируется путем расчета среднего значения для каждого из выделенных периодов и данный
            период делится на это значение, т.е. выполняется нормировка. Тренд прогнозируется путем прибавления к
            последнему историческому периоду шага тренда, рассчитанного как изменение среднего значения за прошлые
            периоды с учетом весов w_1 и w_2 для уменьшения влияния на прогноз дальних значений и усиления влияния
            последних известных значений, соответственно.
                Финальный прогноз строится как произведение прогноза тренда и сезонности.

                При method: 'without_trend' весовые коэффициенты не используются.

                Args:
                    method: 'with_trend' или 'without_trend', выбирается в зависимости от наличия/отсутствия тренда в
                    исходном ВР.
                    past_values: Количество предыдущих фиксированных периодов (период = 12 мес), по которым выполняется
                    прогноз (по умолчанию 3).
                    plots: Построение графиков (по умолчанию False)

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
            yearly_normalized.append((period_data / period_mean).reset_index(drop = True))

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
                    (means_df[f'Period_{i - 1}'] - means_df[f'Period_{i}']) * (w_2 if i == 2 else w_1 / (past_values - 2))
                    for i in range(2, past_values + 1)
                )
                forecast_average = total_step + means_df['Period_1'].values

            # Итоговый прогноз: тренд * нормированная сезонность
            result_df = average_normalized * forecast_average
            # МОЙ КУСОК
            result_df.reset_index(inplace=True)
            result_df.rename(columns = {result_df.columns[0]: self.column_name_with_date}, inplace=True)
            result_df.set_index(self.column_name_with_date, inplace = True)

            #Замена отрицательных значений в столбце на минимально возможное
            for column in result_df.columns:
                for i in range(len(list(result_df[column]))):
                    if list(result_df[column])[i] < 0:
                        values = list(self.df[column])
                        min_value = min(values)
                        list(result_df[column])[i] = min_value

        elif method == 'without_trend':
            result_df = average_normalized * overall_mean
            # Введение названия столбца с индексом для столбца с датами
            result_df.reset_index(inplace=True)
            result_df.rename(columns = {result_df.columns[0]: self.column_name_with_date}, inplace=True)
            result_df.set_index(self.column_name_with_date, inplace = True)

        # Формируем временной индекс для прогноза
        new_index = pd.date_range(start = self.df.index.max() + pd.DateOffset(months=1),
                                  periods = self.forecast_periods,
                                  freq = 'MS')
        result_df.index = new_index
        result_df.reset_index(inplace=True)
        result_df.rename(columns = {result_df.columns[0]: self.column_name_with_date}, inplace=True)
        result_df.set_index(self.column_name_with_date, inplace = True)

        #Замена отрицательных значений в столбце на минимально возможное
        for column in result_df.columns:
            for i in range(len(list(result_df[column]))):
                if list(result_df[column])[i] < 0:
                    values = list(self.df[column])
                    min_value = min(values)
                    list(result_df[column])[i] = min_value

        return result_df


    def decomposition_calendar_years(self, method: str, past_values: int = 3, w_1: float = 0.2, w_2: float = 0.8):
        """
            Ручная декомпозиция временного ряда по календарным годам.
            Используется для прогноза ВР с трендом (если указан method: 'with_trend') и сезонностью.

                Из временного ряда выделяются данные за последние past_values календарных лет.
            При прогнозировании ВР без тренда:
                Для каждого из выделенных периодов считается среднее значение и данный период делится на это значение,
            т.е. выполняется нормировка. Полученные нормированные периоды усредняются и умножаются на среднее значение
            рассматриваемой части ВР.

            При прогнозировании ВР без тренда:
                Сезонность прогнозируется путем расчета среднего значения для каждого из выделенных периодов и данный
            период делится на это значение, т.е. выполняется нормировка. Тренд прогнозируется путем прибавления к
            последнему историческому периоду шага тренда, рассчитанного как изменение среднего значения за прошлые
            периоды с учетом весов w_1 и w_2 для уменьшения влияния на прогноз дальних значений и усиления влияния
            последних известных значений, соответственно.
                Финальный прогноз строится как произведение прогноза тренда и сезонности.

                При method: 'without_trend' весовые коэффициенты не используются.

                Args:
                    method: 'with_trend' или 'without_trend', выбирается в зависимости от наличия/отсутствия тренда в
                    исходном ВР.
                    past_values: Количество предыдущих фиксированных периодов (период = 12 мес), по которым выполняется
                    прогноз (по умолчанию 3).
                    plots: Построение графиков (по умолчанию False)

                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        if method not in ['with_trend', 'without_trend']:
            raise ValueError("Метод должен быть 'with_trend' или 'without_trend'.")

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
            yearly_normalized.append((period_data / period_mean).reset_index(drop = True))
            if last_month == 12:
                means_df[f'Period_{i + 1}'] = period_mean
            else:
                means_df[f'Period_{i}'] = period_mean
        months_to_forecast = self.forecast_periods + (last_month if (last_month != 12) else 0)
        # Среднее нормализованное значение
        average_normalized = sum(yearly_normalized) / past_values
        average_normalized = pd.concat([average_normalized] * self.forecast_periods, ignore_index = True).iloc[
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
                    (means_df[f'Period_{i}'] - means_df[f'Period_{i + 1}']) * (w_2 if i == 1 else w_1 / (past_values - 2))
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
                result_df = pd.concat([result_dff, result_df_dop], axis = 0)


        elif method == 'without_trend':
            result_df = average_normalized * overall_mean

        if last_month != 12:
            # Прогноз для оставшихся месяцев текущего года
            result_df_forecast = result_df.iloc[last_month:months_to_forecast]
            new_index_current_year = pd.date_range(
                start = last_date + pd.DateOffset(months = 1), periods = self.forecast_periods, freq = 'MS'
            )
            result_df_forecast.index = new_index_current_year
            # Введение названия столбца с индексом для столбца с датами
            result_df_forecast.reset_index(inplace=True)
            result_df_forecast.rename(columns = {result_df_forecast.columns[0]: self.column_name_with_date}, inplace=True)
            result_df_forecast.set_index(self.column_name_with_date, inplace = True)

            #Замена отрицательных значений в столбце на минимально возможное
            for column in result_df_forecast.columns:
                for i in range(len(list(result_df_forecast[column]))):
                    if list(result_df_forecast[column])[i] < 0:
                        values = list(self.df[column])
                        min_value = min(values)
                        list(result_df_forecast[column])[i] = min_value

        else:
            # Прогноз на `months_to_forecast` месяцев, начиная с января
            result_df_forecast = result_df.iloc[:months_to_forecast]
            new_index = pd.date_range(
                start = last_date + pd.DateOffset(months = 1),
                periods = self.forecast_periods,
                freq = 'MS'
            )
            result_df_forecast.index = new_index
            # Введение названия столбца с индексом для столбца с датами
            result_df_forecast.reset_index(inplace=True)
            result_df_forecast.rename(columns = {result_df_forecast.columns[0]: self.column_name_with_date}, inplace=True)
            result_df_forecast.set_index(self.column_name_with_date, inplace = True)

            #Замена отрицательных значений в столбце на минимально возможное
            for column in result_df_forecast.columns:
                for i in range(len(list(result_df_forecast[column]))):
                    if list(result_df_forecast[column])[i] < 0:
                        values = list(self.df[column])
                        min_value = min(values)
                        list(result_df_forecast[column])[i] = min_value

        return result_df_forecast


    def prophet_forecast_sequential(self):
        """
            ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ Метод PROPHET.
            Универсальный для всех ВР.

                Функция выполняет прогнозирование ВР с использованием модели Prophet. Она автоматически оптимизирует
            параметры модели для каждого ряда при помощи Grid Search, минимизируя ошибку MAPE.

                Параметры модели Prophet:
                    seasonality_mode: управляет характером сезонности (аддитивный или мультипликативный).
                    n_changepoints: задает количество точек, где модель может изменять направление тренда.
                    changepoint_prior_scale: регулирует гибкость модели в этих точках, чтобы учесть изменения или сгладить тренд.

                Args:
                    plots: Построение графиков (по умолчанию False)

                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        # Отключение логов INFO, возможно отключаются еще какие-то логи дополнительно
        df = self.df.copy()
        #forecast_periods = self.forecast_periods

        # Параметры для перебора
        param_grid = {
            'seasonality_mode': ['multiplicative'],
            'n_changepoints': [12, 18, 24, 36],
            'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.2, 0.5],
        }

        # Сортировка и подготовка данных
        df.reset_index(inplace = True)
        df[self.column_name_with_date] = pd.to_datetime(df[self.column_name_with_date])
        df.sort_values(by = self.column_name_with_date, inplace=True)

        # Разделение на обучающую и тестовую выборки
        train = df.iloc[:-self.forecast_periods]
        test = df.iloc[-self.forecast_periods:]

        # Список временных рядов для прогнозирования
        series_list = [col for col in train.columns if col != self.column_name_with_date]

        results = []  # Результаты параметров для каждого временного ряда

        for series in series_list:
            # Подготовка данных для Prophet
            series_df = train[[self.column_name_with_date, series]].rename(
                columns = {self.column_name_with_date: 'ds', series: 'y'})

            best_mape = float('inf')
            best_params = None

            for params in ParameterGrid(param_grid):
                model = Prophet(
                                seasonality_mode = params['seasonality_mode'],
                                n_changepoints = params['n_changepoints'],
                                changepoint_prior_scale = params['changepoint_prior_scale'])

                model.fit(series_df)

                # Прогнозирование
                future = model.make_future_dataframe(periods = self.forecast_periods, freq = 'MS')
                forecast = model.predict(future)

                # Вычисление MAPE
                test_series = test[[self.column_name_with_date, series]].rename(
                    columns = {self.column_name_with_date: 'ds', series: 'y'})
                forecast_test_period = forecast[-self.forecast_periods:]

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
                columns = {self.column_name_with_date: 'ds', series: 'y'})
            values = list(series_df['y'])
            min_value = min(values)

            model = Prophet(seasonality_mode = best_params['seasonality_mode'],
                            n_changepoints = best_params['n_changepoints'],
                            changepoint_prior_scale = best_params['changepoint_prior_scale'])

            model.fit(series_df)
            future = model.make_future_dataframe(periods = self.forecast_periods, freq = 'MS')
            forecast = model.predict(future)

            for i in range(len(forecast)):
                if forecast[i] < 0:
                    forecast[i] = min_value

            if forecast_df.empty:
                forecast_df['ds'] = forecast['ds']
            forecast_df[series] = forecast['yhat']

        # Введение названия столбца с индексом для столбца с датами
        forecast_df.rename(columns = {forecast_df.columns[0]: self.column_name_with_date}, inplace=True)
        forecast_df.set_index(self.column_name_with_date, inplace = True)

        return forecast_df.tail(self.forecast_periods)

    

    def prophet_forecast_per_thread(self, df, train, test, series, forecast_df):
        """
            ВСПОМОГАТЕЛЬНЫЙ МЕТОД ДЛЯ ПАРАЛЛЕЛЬНОЙ ВЕРСИИ МЕТОДА PROPHET.
            Args:
                df: исходный ВР для анализа
                train: тренировочная выборка
                test: тестовая выборка
                series: ВР для прогнозирования
                forecast_df: Датафрейм с прогнозом
        """
        param_grid = {
                'seasonality_mode': ['multiplicative'],
                'n_changepoints': [12, 18, 24, 36],
                'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.2, 0.5]
            }
        data = df[[self.column_name_with_date, series]].rename(
            columns = {self.column_name_with_date: 'ds', series: 'y'})
        
        values = list(data['y'])
        min_value = min(values)

        series_df = train[[self.column_name_with_date, series]].rename(
            columns = {self.column_name_with_date: 'ds', series: 'y'})
        #print(data)

        best_mape = float('inf')
        best_params = None
        for params in ParameterGrid(param_grid):
            model = Prophet(
                            seasonality_mode = params['seasonality_mode'],
                            n_changepoints = params['n_changepoints'],
                            changepoint_prior_scale = params['changepoint_prior_scale'])

            model.fit(series_df)

            # Прогнозирование
            forecast = model.predict(
                model.make_future_dataframe(periods = self.forecast_periods, freq = 'MS')
                )

            # Вычисление MAPE
            test_series = test[[self.column_name_with_date, series]].rename(
                columns = {self.column_name_with_date: 'ds', series: 'y'})
            forecast_test_period = forecast[-self.forecast_periods:]

            # Сравнение только по датам тестового набора
            aligned_forecast = forecast_test_period.set_index('ds').reindex(test_series['ds']).dropna()
            aligned_test = test_series.set_index('ds').reindex(aligned_forecast.index).dropna()

            mape = np.mean(np.abs((aligned_forecast['yhat'] - aligned_test['y']) / aligned_test['y'])) * 100

            if mape < best_mape:
                best_mape = mape
                best_params = params

        #print(f"Лучшие параметры для {series}: {best_params}, MAPE: {best_mape:.2f}")
        model = Prophet(seasonality_mode = best_params['seasonality_mode'],
                        n_changepoints = best_params['n_changepoints'],
                        changepoint_prior_scale = best_params['changepoint_prior_scale'])

        model.fit(data)
        forecast = model.predict(model.make_future_dataframe(periods = self.forecast_periods, freq = 'MS'))

        if forecast_df.empty:
            forecast_df['ds'] = forecast['ds']
        forecast_df[series] = forecast['yhat']

        for i in range(len(list(forecast_df[series]))):
            if list(forecast_df[series])[i] < 0:
                list(forecast_df[series])[i] = min_value


    def prophet_forecast_PARALLEL(self):
        """
            ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ МЕТОДА PROPHET
            Для корректной рабооты данной функции используется вспомогательная функция prophet_forecast_per_thread.
            Здесь ниже представлено непосредственно распараллеливание последовательной версии prophet_forecast_sequential
            !!!ВНИМАНИЕ!!! Если при использовании данной функции у Вас не хватает оперативной памяти компьютера, то воспользуйтесь
            последовательной версии данного метода prophet_forecast_sequential.
        """
        df = self.df.copy()
        # Сортировка и подготовка данных
        df.reset_index(inplace = True)
        df[self.column_name_with_date] = pd.to_datetime(df[self.column_name_with_date])
        df.sort_values(by = self.column_name_with_date, inplace=True)

        # Разделение на обучающую и тестовую выборки
        train = df.iloc[:-self.forecast_periods]
        test = df.iloc[-self.forecast_periods:]

        # Список временных рядов для прогнозирования
        series_list = [col for col in train.columns if col != self.column_name_with_date]

        threads = []
        forecast_df = pd.DataFrame()
        for series in series_list:
            t = threading.Thread(target = self.prophet_forecast_per_thread,
                                    args = (df, train, test, series, forecast_df))

            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        
        # Введение названия столбца с индексом для столбца с датами
        forecast_df.rename(columns = {'ds': self.column_name_with_date}, inplace = True)
        forecast_df.set_index(self.column_name_with_date, inplace = True)

        return forecast_df.tail(self.forecast_periods)


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
                Args:
                    plots: Построение графиков (по умолчанию False)

                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        forecasts = {}

        for channel in self.df.columns:
            #Поиск минимального значения в колонке
            values = list(self.df[channel])
            min_value = min(values)

            #Логарифмирование данных
            log_transformer = FunctionTransformer(np.log)
            self.df[channel] = log_transformer.transform(self.df[channel])
            ts = self.df[channel].dropna()
            original_series = ts.copy()
            was_non_stationary = False

            # Проверка на стационарность
            if not Preprocessing(ts).check_stationarity():
                ts = Preprocessing(ts).make_stationary()
                was_non_stationary = True

            try:
                # Построение модели
                model = auto_arima(
                    ts,
                    seasonal = True,                      # Использовать сезонность
                    D = 1,                                # Порядок сезонного дифференцирования
                    m = 12,                               # Частота сезонности (12 месяцев)
                    stationary = False,                   # Определение стационарности
                    test = 'pp',                          # Тест на стационарность
                    information_criterion = 'hqic',       # Критерий для выбора лучшей модели
                    stepwise = True,                      # Пошаговый подбор параметров
                    suppress_warnings = False,            # Подавление предупреждений
                    max_p = 5, max_q = 5, max_d = 1,          # Максимальные значения p, q, d
                    max_P = 2, max_Q = 2, max_D = 1,          # Максимальные значения P, Q, D
                    trace = False                         # Вывод логов
                )

                # Обучение модели
                model.fit(ts)
                forecast = model.predict(n_periods = self.forecast_periods)

                # Преобразование прогноза в исходный масштаб
                if was_non_stationary:
                    last_observation = original_series.iloc[-1]
                    forecast = Preprocessing(pd.Series(forecast)).inverse_difference(last_observation)

                forecast = np.exp(forecast)

                #Замена отрицательных значений на минимально возможное в колонке
                for i in range(len(forecast)):
                    if forecast[i] < 0:
                        forecast[i] = min_value
                        
                forecasts[channel] = forecast

            except Exception as e:
                print(f"Error processing channel {channel}: {e}")
                forecasts[channel] = np.nan

        # Формирование DataFrame с прогнозом
        forecast_df = pd.DataFrame(
            forecasts,
            index = pd.date_range(start = self.df.index[-1] + pd.DateOffset(months = 1), periods = self.forecast_periods,
                                freq = 'MS')
        )
        # Введение названия столбца с индексом для столбца с датами
        forecast_df = forecast_df.reset_index()
        forecast_df = forecast_df.rename(columns = {forecast_df.columns[0]: self.column_name_with_date})
        forecast_df.set_index(self.column_name_with_date, inplace = True)

        return forecast_df


    def XGBRegressor_model(self, n_splits, features, target_value, test_size = 12, gap = 1):
        """
            Модель Градиентного Бустинга на основе библиотеки XGBoost.
            Настоящий метод основан на кросс-валидации. Вся выборка разделяется на определенное число фолдов (n_splits) и путем метода кросс-валидации
            ищется наиболее оптимальное число решающих деревьев, которое потом используется для основного прогноза. На каждом шаге кросс-валидации 
            выборка разделяется на тренирововчную и тестовую. Для каждой тренировочной и тестовой выборок создаются столбцы с категориальными признаками и
            запаздывающими компонентами рассматриваемого ВР.
            Для прогноза в будущее также создаются столбцы с категориальными признаками.
            Args:
                n_splits: число фолдов (разбиений)
                features: Список из столбцов DataFrame с категориальными признаками.
                target_value: Целевая переменная (Переменная, которую предсказываем)
                test_size: размер тестовой выборки
                gap: 
            Returns:
        """
        if ( max(self.df.index.year) - min(self.df.index.year) ) < n_splits:
            raise ValueError('Выберите меньшее целое число фолдов (разбиений)')

        df_full = self.df.copy()
        FEATURES = features
        TARGET = target_value
    
        date_range = pd.date_range(start = self.df.index[-1] + pd.DateOffset(months = 1), periods = self.forecast_periods,
                                freq = 'MS')
        forecasts = {}
        for column in df_full.columns:
            tss = TimeSeriesSplit(n_splits = n_splits, test_size = test_size, gap = gap)
            df = df_full[[column]]
            df.rename(columns = {column: 'Share'}, inplace = True)
            df = df.sort_index()

            #Кросс-валидация
            fold = 0
            iterations = []
            for train_idx, val_idx in tss.split(df):
                train = df.iloc[train_idx]
                test = df.iloc[val_idx]
            
                train = Preprocessing(train).create_features()
                test = Preprocessing(test).create_features()
                
                train = Preprocessing(train).add_lags()
                test = Preprocessing(test).add_lags()
            
                X_train = train[FEATURES]
                y_train = train[TARGET]
            
                X_test = test[FEATURES]
                y_test = test[TARGET]
            
                reg = xgb.XGBRegressor(base_score = 0.5,           # первоначальная оценка прогноза (глобальная погрешность).
                                       booster = 'gbtree',         # тип бустера.
                                       n_estimators = 1000,        # число деревьев.
                                       early_stopping_rounds = 50, # параметр для предотвращения переобучения.
                                       objective = 'reg:linear',   # задача машинного обучения.
                                       learning_rate = 0.1,        # контроль величины шага, с которой модель обучается.
                                       max_depth = 8,              # максимальная глубина дерева.
                                       min_child_weight = 7,       # минимальная сумма весов наблюдений, требующаяся для дочернего объекта.
                                       gamma = 0.4,                # параметр регуляризации, принимает значения от 0 до inf.
                                       subsample = 0.7,            # доля объектов обучающей выборки.
                                       colsample_bytree = 0.6,     # семейство параметров для подвыборки столбцов.
                                       reg_alpha = 0.05,           # L1 регуляризация
                                       reg_lambda = 1.0,
                                       n_jobs = 16)           # L2 регуляризация
                reg.fit(X_train, y_train,
                        eval_set = [(X_train, y_train), (X_test, y_test)],
                        verbose = False)
            
                reg.predict(X_test)
                iterations.append(reg.best_iteration)
                
            #Считаем нужное количество деревьев для корректного прогноза
            n_estimators_new = int(np.round((np.average(iterations)), 0))
            
            #Прогноз в будущее
            df = Preprocessing(df).create_features()
            df = Preprocessing(df).add_lags()

            X_all = df[FEATURES]
            y_all = df[TARGET]

            reg = xgb.XGBRegressor(base_score = 0.5,
                        booster = 'gbtree', 
                        n_estimators = n_estimators_new,
                        early_stopping_rounds = 50,
                        objective = 'reg:linear',
                    learning_rate = 0.1,
                        max_depth = 8, 
                        min_child_weight = 7, 
                        gamma = 0.4, 
                        subsample = 0.7, 
                        colsample_bytree = 0.6, 
                        reg_alpha = 0.05,
                        reg_lambda = 1.0,
                        n_jobs = 16)
            reg.fit(X_all, y_all,
                    eval_set = [(X_all, y_all)],
                    verbose = False)

            future_df = pd.DataFrame(index = date_range)
            future_df['isFuture'] = True
            df['isFuture'] = False
            df_and_future = pd.concat([df, future_df])

            df_and_future = Preprocessing(df_and_future).create_features()
            df_and_future = Preprocessing(df_and_future).add_lags()
            future_w_features = df_and_future.query('isFuture').copy()
        
            forecast = reg.predict(future_w_features[FEATURES])
            forecasts[column] = forecast

        #Формирование итогового DataFrame с прогнозом
        forecast_df = pd.DataFrame(
            forecasts,
            index = pd.date_range(start = self.df.index[-1] + pd.DateOffset(months = 1), periods = self.forecast_periods,
                                freq = 'MS')
        )
        # Введение названия столбца с индексом для столбца с датами
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns = {forecast_df.columns[0]: self.column_name_with_date}, inplace=True)
        forecast_df.set_index(self.column_name_with_date, inplace = True)
        return forecast_df
    

    def random_forest_forecast(self):
        """
            Модель Случайного леса.
            #TODO
        """
        df = self.df.copy()

        df.reset_index(inplace = True)
        df[self.column_name_with_date] = pd.to_datetime(df[self.column_name_with_date])
        df = df.sort_values(by=self.column_name_with_date)

        series_list = [col for col in df.columns if col != self.column_name_with_date]
        forecast_df = pd.DataFrame()

        for series in series_list:
            series_df = df[[self.column_name_with_date, series]].rename(
                columns = {self.column_name_with_date: 'ds', series: 'y'})

            for lag in range(1, 13):
                series_df[f"lag_{lag}"] = series_df["y"].shift(lag)

            series_df.dropna(inplace=True)

            X = series_df.drop(columns=["ds", "y"])
            y = series_df["y"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.forecast_periods, shuffle = False
            )

            params = {
                "n_estimators": 400,    # число деревьев.
                "random_state": 42,    
                "min_samples_split": 2, # минимальное количество образцов, необходимых для разделения узла.
                "min_samples_leaf": 2,  # минимальное количество образцов в листовом узле.
                "max_depth": 10         # максимальная глубина дерева
            }

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            future_X = pd.DataFrame([X_test.iloc[-1]])  # Последнее наблюдение для прогнозирования

            preds = []
            for _ in range(self.forecast_periods):
                pred = model.predict(future_X)[0]
                preds.append(pred)

                future_X = future_X.shift(-1, axis = 1)
                future_X.iloc[0, -1] = pred

            if forecast_df.empty:
                forecast_df["ds"] = pd.date_range(
                    start=df[self.column_name_with_date].max(), periods = self.forecast_periods + 1, freq = "MS")[1:]

            forecast_df[series] = preds

        forecast_df.set_index("ds", inplace = True)
        forecast_df.reset_index(inplace=True).rename(columns = {"ds": self.column_name_with_date}, inplace=True)
        forecast_df.set_index(self.column_name_with_date, inplace = True)
        return forecast_df


    def process_model(self,
                      model_name: str,
                      error_dir: str = None,
                      plots_dir: str = None,
                      plots: bool = False,
                      test: bool = False):
        """
            Функция для применения интересующей ML-модели.
            Args:
                model_name: Сокращённое имя ML_модели
                model_method: Метод используемой модели (fixed periods, calendar_years, linear_trend, logistic_trend)
                type_of_group: Тип группы (GROUP 1, GROUP 2, GROUP 3, GROUP 4)
                weight: вес используемой модели
                plots_dir: путь к директории, куда будут сохраняться графики, если параметр plots = True
                plots: переменная типа bool. Если True, то графики с прогнозам строятся. В противном случае нет.
                test: переменная типа bool. Если True, то тестинг модели проводится. В противном случае нет.
            Returns:
                forecast_df: Прогнозный DataFrame.
        """
        method_map = {
                'ARIMA': self.auto_arima_forecast,
                'Prophet': self.prophet_forecast_PARALLEL,
                'XGB_Regressor': lambda: self.XGBRegressor_model(n_splits = 2, 
                                                              features = ['year', 'year start', 'month', 'quarter start', 'season'], 
                                                              target_value = 'Share'),
                'Random_Forest': self.random_forest_forecast,

                'Regr_lin': lambda: self.regression_model(method = 'linear_trend'),
                'Regr_log': lambda: self.regression_model(method = 'logistic_trend'),

                'Naive': self.naive_forecast, #по умолчанию past_values = 3
                'Naive_last_6_months': lambda: self.naive_forecast(past_values = 6),

                'Naive_with_error': lambda: self.naive_forecast(past_values = 3, weigh_error = 0.8), #по умолчанию past_values = 3
                'Naive_with_error_last_6_months': lambda: self.naive_forecast(past_values = 6, weigh_error = 0.8),

                'RollMean_periods': lambda: self.rolling_mean(method = 'fixed_periods'),
                'RollMean_years': lambda: self.rolling_mean(method = 'calendar_years'),

                'SeasonDec_periods': lambda: self.seasonal_decomposition(method = 'fixed_periods'),
                'SeasonDec_years': lambda: self.seasonal_decomposition(method = 'calendar_years'),

                'Dec_with_trend_periods': lambda: self.decomposition_fixed_periods(method = 'with_trend'),
                'Dec_with_trend_years': lambda: self.decomposition_calendar_years(method = 'with_trend'),

                'Dec_without_trend_periods': lambda: self.decomposition_fixed_periods(method = 'without_trend'), #по умолчанию past_values = 3
                'Dec_without_trend_last_2_periods': lambda: self.decomposition_fixed_periods(method = 'without_trend', past_values = 2),

                'Dec_without_trend_years': lambda: self.decomposition_calendar_years(method = 'without_trend'), #по умолчанию past_values = 3
                'Dec_without_trend_last_2_years': lambda: self.decomposition_calendar_years(method = 'with_trend', past_values = 2)
            }
        if model_name not in method_map:
                raise ValueError(f"Модель '{model_name}' не найдена в списке! Выберите другую модель.")

        # Тестинг проводится только если число предсказываемых периодов (месяцев) <= 12. В противном случае тестинг не проводится
        if test and self.forecast_periods <= 12:
            train_data = self.df.iloc[:-self.forecast_periods]
            test_data = self.df.iloc[-self.forecast_periods:]
            self.df = train_data
        else:
            test = False

        forecast_df = method_map[model_name]()
        print(f"РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ {model_name.upper()}", forecast_df.round(4), sep = '\n', end = '\n\n')
        
        # Если задан параметр plots == True
        if plots and plots_dir is not None:
            Postprocessing(self.df, forecast_df).get_plot(column_name_with_date = self.column_name_with_date,
                                                            save_dir = f'{plots_dir}/{model_name}', test_data = test_data)
        # Если задан параметр test == True    
        if test:
            error_df = Postprocessing.calculate_forecast_error(forecast_df = forecast_df, test_data = test_data)
            error_df.to_excel(f'{error_dir}/{model_name}_MAPE(%)_{self.forecast_periods}.xlsx')
        return forecast_df
    

    #Перегрузка функций
    def process_model_PARALLEL(self,
                      forecasts,
                      tests,
                      trains,
                      #coeff,
                      model_name: str,
                      error_dir: str = None,
                      plots_dir: str = None,
                      plots: bool = False,
                      test: bool = False):
        """
            Функция для применения интересующей ML-модели.
            Args:
                model_name: Сокращённое имя ML_модели
                model_method: Метод используемой модели (fixed periods, calendar_years, linear_trend, logistic_trend)
                type_of_group: Тип группы (GROUP 1, GROUP 2, GROUP 3, GROUP 4)
                weight: вес используемой модели
                plots_dir: путь к директории, куда будут сохраняться графики, если параметр plots = True
                plots: переменная типа bool. Если True, то графики с прогнозам строятся. В противном случае нет.
                test: переменная типа bool. Если True, то тестинг модели проводится. В противном случае нет.
            Returns:
                forecast_df: Прогнозный DataFrame.
        """
        method_map = {
                'ARIMA': self.auto_arima_forecast,
                'Prophet': self.prophet_forecast_PARALLEL,
                'XGB_Regressor': lambda: self.XGBRegressor_model(n_splits = 2, 
                                                              features = ['year', 'year start', 'month', 'quarter start', 'season'], 
                                                              target_value = 'Share'),
                'Random_Forest': self.random_forest_forecast,

                'Regr_lin': lambda: self.regression_model(method = 'linear_trend'),
                'Regr_log': lambda: self.regression_model(method = 'logistic_trend'),

                'Naive': self.naive_forecast, #по умолчанию past_values = 3
                'Naive_last_6_months': lambda: self.naive_forecast(past_values = 6),

                'Naive_with_error': lambda: self.naive_forecast(past_values = 3, weigh_error = 0.8), #по умолчанию past_values = 3
                'Naive_with_error_last_6_months': lambda: self.naive_forecast(past_values = 6, weigh_error = 0.8),

                'RollMean_periods': lambda: self.rolling_mean(method = 'fixed_periods'),
                'RollMean_years': lambda: self.rolling_mean(method = 'calendar_years'),

                'SeasonDec_periods': lambda: self.seasonal_decomposition(method = 'fixed_periods'),
                'SeasonDec_years': lambda: self.seasonal_decomposition(method = 'calendar_years'),

                'Dec_with_trend_periods': lambda: self.decomposition_fixed_periods(method = 'with_trend'),
                'Dec_with_trend_years': lambda: self.decomposition_calendar_years(method = 'with_trend'),

                'Dec_without_trend_periods': lambda: self.decomposition_fixed_periods(method = 'without_trend'), #по умолчанию past_values = 3
                'Dec_without_trend_last_2_periods': lambda: self.decomposition_fixed_periods(method = 'without_trend', past_values = 2),

                'Dec_without_trend_years': lambda: self.decomposition_calendar_years(method = 'without_trend'), #по умолчанию past_values = 3
                'Dec_without_trend_last_2_years': lambda: self.decomposition_calendar_years(method = 'with_trend', past_values = 2)
            }
        if model_name not in method_map:
                raise ValueError(f"Модель '{model_name}' не найдена в списке! Выберите другую модель.")

        # Тестинг проводится только если число предсказываемых периодов (месяцев) <= 12. В противном случае тестинг не проводится
        if test and self.forecast_periods <= 12:
            train_data = self.df.iloc[:-self.forecast_periods]
            test_data = self.df.iloc[-self.forecast_periods:]
            self.df = train_data
            trains.append({ model_name: train_data })
            tests.append ({ model_name: test_data })
        else:
            test = False

        forecast_df = method_map[model_name]()
        print(f"РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ {model_name.upper()}",
            forecast_df.round(4), sep = '\n', end = '\n\n')
        
        # # Если задан параметр plots == True
        # if plots and plots_dir is not None:
        #     Postprocessing(self.df, forecast_df).get_plot(column_name_with_date = self.column_name_with_date,
        #                                                     save_dir = f'{plots_dir}/{model_name}', test_data = test_data)
        # # Если задан параметр test == True    
        #if test:
        #    error_df = Postprocessing.calculate_forecast_error(
        #                        forecast_df = forecast_df,
        #                        test_data = test_data
        #                    )
        #    error_df.to_excel(f'{error_dir}/{model_name}_MAPE(%).xlsx')

        forecasts.append({ model_name: forecast_df})
