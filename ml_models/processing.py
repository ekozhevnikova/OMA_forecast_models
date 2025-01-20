import sys
import json
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.ml_models.postprocessing import Postprocessing
from pmdarima import auto_arima
import pymannkendall as mk
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from scipy import linalg
from contextlib import contextmanager
import os
#это я для себя:
from OMA_tools.ml_models.preprocessing import Preprocessing
from OMA_tools.ml_models.postprocessing import Postprocessing


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
        future_dates = pd.date_range(last_date + pd.DateOffset(months = 1), periods = self.forecast_periods, freq='ME')

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
        future_predictions.set_index(self.column_name_with_date, inplace=True)
        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots and method == 'linear_trend':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=future_predictions,
                save_dir='plots_for_ML_models/regression_model_linear_trend',
            )
        if plots and method == 'logistic_trend':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=future_predictions,
                save_dir='plots_for_ML_models/regression_model_logistic_trend',
            )
        '''

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
                    freq = 'ME'
                )
            )

            forecast_results[column] = forecast_df
        # Объединение всех столбцов в один DataFrame
        result_df = pd.concat(forecast_results.values(), axis = 1)

        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots and weigh_error > 0:
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=result_df,
                save_dir='plots_for_ML_models/naive_forecast_weigh_error',
            )
        if plots and weigh_error == 0:
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=result_df,
                save_dir='plots_for_ML_models/naive_forecast',
            )
        '''
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
        next_year_dates = pd.date_range(start = next_year_start, periods = months_to_forecast, freq = 'ME')
        next_year_trend_df = pd.DataFrame(next_year_trend, index = next_year_dates, columns = self.df.columns)
        season_forecast_df = pd.DataFrame(
            average_season_val.values[:months_to_forecast],
            index = next_year_dates,
            columns = self.df.columns
        )
        forecast_df = next_year_trend_df + season_forecast_df

        if method == 'calendar_years':
            # Корректируем прогнозы для случаев, когда last_month < 12
            next_year_dates_fact = pd.date_range(start = next_year_start, periods = self.forecast_periods, freq = 'ME')
            if last_month < 12:
                next_year_trend_df = next_year_trend_df.iloc[last_month:].set_index(next_year_dates_fact)
                season_forecast_df = season_forecast_df.iloc[last_month:].set_index(next_year_dates_fact)
                # Финальный прогноз
                forecast_df = next_year_trend_df + season_forecast_df
        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots and method == 'calendar_years':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=forecast_df,
                save_dir='plots_for_ML_models/seasonal_decomposition_calendar_years',
            )
        if plots and method == 'fixed_periods':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=forecast_df,
                save_dir='plots_for_ML_models/seasonal_decomposition_fixed_periods',
            )
        '''
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
        rolling_mean = past_years.rolling(window = self.forecast_periods).mean()
        rolling_mean = past_years.rolling(window = 12).mean()
        detrended = (past_years - rolling_mean).dropna()
        detrended[self.column_name_with_date] = detrended.index.month

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
                                        freq = 'ME'
                                        )
        next_year_rolling_mean_df = pd.DataFrame(
            next_year_rolling_mean,
            index = next_year_dates,
            columns = self.df.columns
        )

        seasonal_forecast = np.tile(seasonatily.values, (1, 1))
        # Финальный прогноз: сложение тренда и сезонности
        final_forecast = next_year_rolling_mean_df + seasonal_forecast
        forecast_df = pd.DataFrame(final_forecast, index = next_year_dates, columns = self.df.columns)

        if method == 'calendar_years':
            next_year_dates_fact = pd.date_range(start = last_date + pd.DateOffset(months=1),
                                                 periods = self.forecast_periods,
                                                 freq = 'ME')
            if last_month < 12:
                forecast_df = forecast_df.iloc[last_month:].set_index(next_year_dates_fact)
        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots and method == 'calendar_years':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=forecast_df,
                save_dir='plots_for_ML_models/rolling_mean_calendar_years',
            )
        if plots and method == 'fixed_periods':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=forecast_df,
                save_dir='plots_for_ML_models/rolling_mean_fixed_periods',
            )
        '''
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

        elif method == 'without_trend':
            result_df = average_normalized * overall_mean

        # Формируем временной индекс для прогноза
        new_index = pd.date_range(start=self.df.index.max() + pd.DateOffset(months=1),
                                  periods=self.forecast_periods,
                                  freq='ME')
        result_df.index = new_index

        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots and method == 'without_trend':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=result_df,
                save_dir='plots_for_ML_models/decomposition_fixed_periods_without_trend',
            )
        if plots and method == 'with_trend':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=result_df,
                save_dir='plots_for_ML_models/decomposition_fixed_periods_with_trend',
            )
        '''
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
                start = last_date + pd.DateOffset(months=1), periods=self.forecast_periods, freq = 'ME'
            )
            result_df_forecast.index = new_index_current_year

        else:
            # Прогноз на `months_to_forecast` месяцев, начиная с января
            result_df_forecast = result_df.iloc[:months_to_forecast]
            new_index = pd.date_range(
                start = last_date + pd.DateOffset(months = 1),
                periods = self.forecast_periods,
                freq = 'ME'
            )
            result_df_forecast.index = new_index
        
        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots and method == 'without_trend':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=result_df_forecast,
                save_dir='plots_for_ML_models/decomposition_calendar_years_without_trend',
            )
        if plots and method == 'with_trend':
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=result_df_forecast,
                save_dir='plots_for_ML_models/decomposition_calendar_years_with_trend',
            )
        '''
        return result_df_forecast

    def prophet_forecast(self):
        """
            Метод PROPHET.
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
        df = self.df.copy()
        #forecast_periods = self.forecast_periods

        # Параметры для перебора
        param_grid = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [12, 18, 24, 36],
            'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.2, 0.5]
        }

        # Сортировка и подготовка данных
        df.reset_index(inplace = True)
        df[self.column_name_with_date] = pd.to_datetime(df[self.column_name_with_date])
        df = df.sort_values(by = self.column_name_with_date)

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
                model = Prophet(seasonality_mode = params['seasonality_mode'],
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

            model = Prophet(seasonality_mode = best_params['seasonality_mode'],
                            n_changepoints = best_params['n_changepoints'],
                            changepoint_prior_scale = best_params['changepoint_prior_scale'])

            model.fit(series_df)
            future = model.make_future_dataframe(periods = self.forecast_periods, freq = 'MS')
            forecast = model.predict(future)

            if forecast_df.empty:
                forecast_df['ds'] = forecast['ds']
            forecast_df[series] = forecast['yhat']

        forecast_df.set_index('ds', inplace = True)
        forecast_df = forecast_df.tail(self.forecast_periods)

        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots:
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=forecast_df,
                save_dir='plots_for_ML_models/prophet_forecast',
            )
        '''

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
                Args:
                    plots: Построение графиков (по умолчанию False)

                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        forecasts = {}

        for channel in self.df.columns:
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
        '''
        # Вызов графиков
        postprocessor = Postprocessing(self.df)
        if plots:
            postprocessor.get_plot(
                column_name_with_date=self.column_name_with_date,
                forecast_df=forecast_df,
                save_dir='plots_for_ML_models/auto_arima_forecast',
            )
        '''
        return forecast_df


class GROUPS(Forecast_Models):
    
    def __init__(self, forecast_periods: int, column_name_with_date: str):
        super().__init__(None, forecast_periods, column_name_with_date)
        
        
    def make_forecast_for_group(self,
                                forecasts,
                                model_name: str, 
                                type_of_group: str,
                                weights_filepath: str,
                                filepath: str,
                                plots = False):
        """
            Функция для прогноза по четырём различным группам.
            
                Args:
                    model_name: Имя модели.
                        Введены следующие сокращения к рассматриваемым моделям:
                        - ARIMA: ARIMA;
                        - Prophet: Prophet;
                        - Наивный метод/Наивный метод с ошибкой (naive_forecast): Naive/Naive_with_error;
                            Наивный метод последние 3 месяца/Наивный метод последние 6 месяце: Naive_last_3_months/Naive_last_6_months
                            Наивный метод с учетом ошибки последние 3 месяца/Наивный метод с учетом ошибки последние 6 месяце: Naive_with_error_last_3_months/Naive_with_error_last_6_months
                        - Скользящее среднее фиксированные периоды/календарные годы (rolling_mean): RollMean_periods/RollMean_years;
                        - Сезонная декомпозиция фиксированные периоды/календарные годы (seasonal_decomposition): SeasonDec_periods/SeasonDec_years;
                        - Декомпозиция с/без тренда фиксированные периоды (decomposition_fixed_periods): Dec_with_trend_periods/Dec_without_trend_periods;
                        - Декомпозиция с/без тренда календарные годы (decomposition_calendar_years): Dec_with_trend_years/Dec_without_trend_years;
                            Декомпозиция с/без тренда последние 3 года/Декомпозиция с/без тренда последние 2 года: Dec_without_trend_2_years/Dec_without_trend_3_years
                        - Регрессия с линейным/логарифмическим трендом (regression_model): Regr_lin/Regr_log.
                        
                    type_of_group: Тип рассматриваемой группы.
                        - GROUP_1: ВР с сезонной и трендовой компонентами;
                        - GROUP_2: ВР с трендовой, но без сезонной компоненты;
                        - GROUP_3: ВР с сезонной, но без трендовой компоненты;
                        - GROUP_4: ВР без сезонной и без трендовой компонент.
                        
                    weights_filepath: Полный путь к конфиг-файлу с весами каждой модели и группы.
                    
                    filepath: Путь, куда будут сохраняться графики.
                    
                    plots: По умолчанию False (т.е не строятся).
                    
                Returns:
                    avg_forecast: усредненный прогноз по всем моделям для каждый из групп.
        """
        last_month_in_df = Preprocessing(self.df).search_last_fact_data()[1]

        #Чтение config.json для корректного указания веса каждой из моделей
        with open(f'{weights_filepath}') as f:
            file_content = f.read()
            groups = json.loads(file_content)

        ############################################################ GROUP 1 (Сезонность и тренд) #########################################################   
        if type_of_group == 'GROUP_1':
            path_to_save = f'{filepath}/Сезонность и тренд' #ВР с трендом и сезонностью
            #Случай, если последний месяц в исходном DataFrame не равен 12 (не декабрь)
            if last_month_in_df != 12:
                match model_name:
                    case 'ARIMA':
                        forecast_df = self.auto_arima_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ ARIMA', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/ARIMA')
                        forecasts.append(forecast_df * groups['GROUP_1_not_december'][0]['ARIMA'])
                        

                    case 'Prophet':
                        forecast_df = self.prophet_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ PROPHET', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/PROPHET')
                        forecasts.append(forecast_df * groups['GROUP_1_not_december'][1]['Prophet'])


                    case 'Dec_with_trend_periods':
                        forecast_df = self.decomposition_fixed_periods(method = 'with_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Декомпозиция с трендом 3 последних НЕ КАЛЕНДАРНЫХ года (seasonality_with_trend_periods)',
                                forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Декомпозиция с трендом (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_1_not_december'][2]['Dec_with_trend_periods'])


                    case 'RollMean_periods':
                        forecast_df = self.rolling_mean(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ скользящее среднее последние 3 ПЕРИОДА (rolling_mean)', 
                                forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Скользящее среднее (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_1_not_december'][3]['RollMean_periods'])


                    case 'SeasonDec_periods':
                        forecast_df = self.seasonal_decomposition(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (тренд - дифференцированием), 3 периода (season_dec_periods)',
                                forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_1_not_december'][4]['SeasonDec_periods'])


                    case 'RollMean_years':
                        forecast_df = self.rolling_mean(method = 'calendar_years')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ скользящее среднее последние 3 ПЕРИОДА (rolling_mean)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Скользящее среднее (календарные годы)')
                        forecasts.append(forecast_df * groups['GROUP_1_not_december'][5]['RollMean_years'])


                    case 'Regr_log':
                        forecast_df = self.regression_model(method = 'logistic_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Логистическая регрессия (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (логарифм)')
                        forecasts.append(forecast_df * groups['GROUP_1_not_december'][6]['Regr_log'])
            #Случай, если последний месяц в исходном DataFrame равен 12 (декабрь)               
            elif last_month_in_df == 12:
                match model_name:
                    case 'ARIMA':
                        forecast_df = self.auto_arima_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ ARIMA', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/ARIMA')
                        forecasts.append(forecast_df * groups['GROUP_1_december'][0]['ARIMA'])
                        

                    case 'Prophet':
                        forecast_df = self.prophet_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ PROPHET', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/PROPHET')
                        forecasts.append(forecast_df * groups['GROUP_1_december'][1]['Prophet'])


                    case 'Dec_with_trend_years':
                        forecast_df = self.decomposition_calendar_years(method = 'with_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Декомпозиция с трендом (фиксированные периоды) (decomposition_fixed_periods)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Декомпозиция с трендом (календарные годы)')
                        forecasts.append(forecast_df * groups['GROUP_1_december'][2]['Dec_with_trend_years'])


                    case 'RollMean_periods':
                        forecast_df = self.rolling_mean(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ скользящее среднее последние 3 ПЕРИОДА (rolling_mean)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Скользящее среднее (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_1_december'][3]['RollMean_periods'])


                    case 'SeasonDec_periods':
                        forecast_df = self.seasonal_decomposition(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (тренд - дифференцированием), фиксированные периоды (seasonal_decomposition)',
                                forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_1_december'][4]['SeasonDec_periods'])


                    case 'Regr_log':
                        forecast_df = self.regression_model(method = 'logistic_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Логистическая регрессия (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (логарифмический тренд)')
                        forecasts.append(forecast_df * groups['GROUP_1_december'][5]['Regr_log'])
        ############################################################ GROUP 2 (Тренд без сезонности) #########################################################               
        elif type_of_group == 'GROUP_2':
            path_to_save = f'{filepath}/Тренд без сезонности' #ВР с трендом и сезонностью
            #Случай, если последний месяц в исходном DataFrame не равен 12 (не декабрь)
            if last_month_in_df != 12:
                match model_name:
                    case 'ARIMA':
                        forecast_df = self.auto_arima_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ ARIMA', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/ARIMA')
                        forecasts.append(forecast_df * groups['GROUP_2_not_december'][0]['ARIMA'])
                        

                    case 'Prophet':
                        forecast_df = self.prophet_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ PROPHET', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/PROPHET')
                        forecasts.append(forecast_df * groups['GROUP_2_not_december'][1]['Prophet'])


                    case 'Regr_lin':
                        forecast_df = self.regression_model(method = 'linear_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Линейная регрессия (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (линейный тренд)')
                        forecasts.append(forecast_df * groups['GROUP_2_not_december'][2]['Regr_lin'])


                    case 'Regr_log':
                        forecast_df = self.regression_model(method = 'logistic_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Логистическая регрессия (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (логарифмический тренд)')
                        forecasts.append(forecast_df * groups['GROUP_2_not_december'][3]['Regr_log'])


                    case 'SeasonDec_periods':
                        forecast_df = self.seasonal_decomposition(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (фиксированные периоды) (seasonal_decomposition_fixed_periods)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_2_not_december'][4]['SeasonDec_periods'])


                    case 'Naive_with_error':
                        forecast_df = self.naive_forecast(past_values = 6, weigh_error = 0.8)
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз с учетом ошибки (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз с учетом ошибки')
                        forecasts.append(forecast_df * groups['GROUP_2_not_december'][5]['Naive_with_error'])
                            
            #Случай, если последний месяц в исходном DataFrame равен 12 (декабрь)               
            elif last_month_in_df == 12:
                match model_name:
                    case 'ARIMA':
                        forecast_df = self.auto_arima_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ ARIMA', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/ARIMA')
                        forecasts.append(forecast_df * groups['GROUP_2_december'][0]['ARIMA'])
                        

                    case 'Prophet':
                        forecast_df = self.prophet_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ PROPHET', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/PROPHET')
                        forecasts.append(forecast_df * groups['GROUP_2_december'][1]['Prophet'])


                    case 'Regr_lin':
                        forecast_df = self.regression_model(method = 'linear_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Регрессия с линейным трендом (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (линейный тренд)')
                        forecasts.append(forecast_df * groups['GROUP_2_december'][2]['Regr_lin'])


                    case 'SeasonDec_periods':
                        forecast_df = self.seasonal_decomposition(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (тренд - дифференцированием)',
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_2_december'][3]['SeasonDec_periods'])


                    case 'Naive_with_error_last_3_months':
                        forecast_df = self.naive_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз с учетом ошибки (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз с учетом ошибки')
                        forecasts.append(forecast_df * groups['GROUP_2_december'][4]['Naive_with_error'])


                    case 'Naive_with_error_last_6_months':
                        forecast_df = self.naive_forecast(past_values = 6)
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз с учетом ошибки (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз с учетом ошибки')
                        forecasts.append(forecast_df * groups['GROUP_2_december'][4]['Naive_with_error'])
                            
        ############################################################ GROUP 3 (Сезонность без тренда) #########################################################
        elif type_of_group == 'GROUP_3':
            path_to_save = f'{filepath}/Сезонность без тренда' #ВР с трендом и сезонностью
            match model_name:
                case 'ARIMA':
                    forecast_df = self.auto_arima_forecast()
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ ARIMA', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/ARIMA')
                    forecasts.append(forecast_df * groups['GROUP_3'][0]['ARIMA'])
                    

                case 'Prophet':
                    forecast_df = self.prophet_forecast()
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ PROPHET', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/PROPHET')
                    forecasts.append(forecast_df * groups['GROUP_3'][1]['Prophet'])


                case 'Dec_without_trend_3_years':
                    forecast_df = self.decomposition_calendar_years(method = 'without_trend')
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ decomposition_calendar_years', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Декомпозиция без тренда (календарные годы)')
                    forecasts.append(forecast_df * groups['GROUP_3'][2]['Dec_without_trend_years'][0])


                case 'Dec_without_trend_2_years':
                    forecast_df = self.decomposition_calendar_years(method = 'without_trend', past_values = 2)
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ decomposition_calendar_years', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Декомпозиция без тренда (календарные годы) последние 2 года')
                    forecasts.append(forecast_df * groups['GROUP_3'][2]['Dec_without_trend_years'][1])


                case 'Dec_without_trend_3_periods':
                    forecast_df = self.decomposition_fixed_periods(method = 'without_trend')
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ decomposition_fixed_periods', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Декомпозиция без тренда (фиксированные периоды) последние 3 года')
                    forecasts.append(forecast_df * groups['GROUP_3'][3]['Dec_without_trend_periods'])


                case 'Dec_without_trend_2_periods':
                    forecast_df = self.decomposition_fixed_periods(method = 'without_trend', past_values = 2)
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ decomposition_fixed_periods', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Декомпозиция без тренда (фиксированные периоды) последние 2 года')
                    forecasts.append(forecast_df * groups['GROUP_3'][3]['Dec_without_trend_periods'])


                case 'RollMean_periods':
                    forecast_df = self.rolling_mean(method = 'fixed_periods')
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Скользящее среднее (фиксированные периоды) rolling_mean', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Скользящее среднее (фиксированные периоды)')
                    forecasts.append(forecast_df * groups['GROUP_3'][4]['RollMean_periods'])


                case 'SeasonDec_periods':
                    forecast_df = self.seasonal_decomposition(method = 'fixed_periods')
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (фиксированные периоды) seasonal_decomposition', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (фиксированные периоды)')
                    forecasts.append(forecast_df * groups['GROUP_3'][5]['SeasonDec_periods'])


                case 'RollMean_years':
                    forecast_df = self.rolling_mean(method = 'calendar_years')
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Скользящее среднее (календарные годы) rolling_mean', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Скользящее среднее (календарные годы)')
                    forecasts.append(forecast_df * groups['GROUP_3'][6]['RollMean_years'])


                case 'SeasonDec_years':
                    forecast_df = self.seasonal_decomposition(method = 'calendar_years')
                    print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (календарные годы) seasonal_decomposition', 
                          forecast_df.round(4), sep = '\n', end = '\n\n')
                    if plots:
                        Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (календарные годы)')
                    forecasts.append(forecast_df * groups['GROUP_3'][6]['SeasonDec_years'])


                case 'Regr_lin':
                        forecast_df = self.regression_model(method = 'linear_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Регрессия с линейным трендом (regr_lin)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (линейный тренд)')
                        forecasts.append(forecast_df * groups['GROUP_3'][7]['Regr_lin'])

        ############################################################ GROUP 4 (Без сезонности и без тренда) #########################################################   
        if type_of_group == 'GROUP_4':
            path_to_save = f'{filepath}/Без сезонности и без тренда' #ВР с трендом и сезонностью
            #Случай, если последний месяц в исходном DataFrame не равен 12 (не декабрь)
            if last_month_in_df != 12:
                match model_name:
                    case 'ARIMA':
                        forecast_df = self.auto_arima_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ ARIMA', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/ARIMA')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][0]['ARIMA'])
                        

                    case 'Prophet':
                        forecast_df = self.prophet_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ PROPHET', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/PROPHET')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][1]['Prophet'])


                    case 'Naive':
                        forecast_df = self.naive_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][2]['Naive'])


                    case 'Dec_without_trend_periods':
                        forecast_df = self.decomposition_fixed_periods(method = 'without_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Декомпозиция без тренда (фиксированные периоды) (decomposition_fixed_periods)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Декомпозиция без тренда (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][3]['Dec_without_trend_periods'])


                    case 'Regr_lin':
                        forecast_df = self.regression_model(method = 'linear_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Регрессия с линейныйм трендом (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (линейный тренд)')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][4]['Regr_lin'])


                    case 'Regr_log':
                        forecast_df = self.regression_model(method = 'logistic_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Регрессия с логарифмическим трендом (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (логарифмический тренд)')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][5]['Regr_log'])


                    case 'SeasonDec_periods':
                        forecast_df = self.seasonal_decomposition(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (фиксированные периоды) seasonal_decomposition', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][6]['SeasonDec_periods'])


                    case 'Naive_with_error_last_6_months':
                        forecast_df = self.naive_forecast(past_values = 6)
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз с учетом ошибки (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз с учетом ошибки последние 6 месяцев')
                        forecasts.append(forecast_df * groups['GROUP_4_not_december'][7]['Naive_with_error'])
            #Случай, если последний месяц в исходном DataFrame равен 12 (декабрь)               
            elif last_month_in_df == 12:
                match model_name:
                    case 'ARIMA':
                        forecast_df = self.auto_arima_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ ARIMA', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/ARIMA')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][0]['ARIMA'])
                        

                    case 'Prophet':
                        forecast_df = self.prophet_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ PROPHET', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/PROPHET')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][1]['Prophet'])


                    case 'Naive_last_3_months':
                        forecast_df = self.naive_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз последние 3 месяца')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][2]['Naive'])


                    case 'Naive_last_6_months':
                        forecast_df = self.naive_forecast(past_values = 6)
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз последние 6 месяцев')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][2]['Naive'])


                    case 'Regr_log':
                        forecast_df = self.regression_model(method = 'logistic_trend')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Регрессия с логарифмическим трендом (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Регрессия (логарифмический тренд)')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][3]['Regr_log'])


                    case 'SeasonDec_periods':
                        forecast_df = self.seasonal_decomposition(method = 'fixed_periods')
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Сезонная декомпозиция (фиксированные периоды) (regression_model)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Сезонная декомпозиция, тренд дифференцированием (фиксированные периоды)')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][4]['SeasonDec_periods'])    


                    case 'Naive_with_error_last_3_months':
                        forecast_df = self.naive_forecast()
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз с учетом ошибки (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз с учетом ошибки последние 3 месяца')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][5]['Naive_with_error'])  


                    case 'Naive_with_error_last_6_months':
                        forecast_df = self.naive_forecast(past_values = 6)
                        print('РЕЗУЛЬТАТ РАБОТЫ ФУНКЦИИ Наивный прогноз с учетом ошибки (naive_forecast)', 
                              forecast_df.round(4), sep = '\n', end = '\n\n')
                        if plots:
                            Postprocessing(self.df, forecast_df).get_plot(self.column_name_with_date, f'{path_to_save}/Наивный прогноз с учетом ошибки последние 6 месяцев')
                        forecasts.append(forecast_df * groups['GROUP_4_december'][5]['Naive_with_error'])                
    
    def main(self,
             filename, 
             list_of_replacements: list, 
             model_name_list_group_1: list, 
             model_name_list_group_2: list, 
             model_name_list_group_3: list,
             model_name_list_group_4: list,
             weights_filepath: str,
             filepath_for_graphs: str,
             filepath_for_avg_graphs: str,
             plots = False):
        """
            Функция для запуска ансамбля ML-моделей для прогнозирования ВР.

                Args:
                    filename: файл с фактическими данными по месячным значениям долей;
                    list_of_replacements: Список листов, взодящих в состав файла с фактическими значениями;
                    model_name_list_group_1: Список из моделей, которые используются в GROUP_1;
                    model_name_list_group_2: Список из моделей, которые используются в GROUP_2;
                    model_name_list_group_3: Список из моделей, которые используются в GROUP_3;
                    model_name_list_group_4: Список из моделей, которые используются в GROUP_4;
                    weights_filepath: Полный путь к config-файлу с весами для каждой из моделей;
                    filepath_for_graphs: Путь к директории, куда будут сохраняться графики с прогнозами.
                    filepath_for_avg_graphs: Путь к директории, куда будут сохраняться графики ансамблей из моделей с результирующим прогнозом.
                Returns:
                    Прогнозный DataFrame, полученный в результате работы ансамбля ML-моделей.
        """
        self.df = Preprocessing.get_data_for_forecast(filename, list_of_replacements, self.column_name_with_date)

        df_list_1 = []
        df_list_2 = []
        df_list_3 = []
        df_list_4 = []
        for column in self.df.columns:
            time_series = pd.Series(self.df[column])
            lagged_series = time_series.shift(12)
            correlation = time_series.corr(lagged_series)
            trend_test_result = mk.original_test(time_series)
            # Есть сезонность, есть тренд
            if ((correlation >= 0.7) and trend_test_result.h):
                df_list_1.append(self.df[column])

            # Нет сезонности, есть тренд
            if ((correlation < 0.7) and trend_test_result.h):
                df_list_2.append(self.df[column])

            # Есть сезонность, нет тренда
            if ((correlation >= 0.7) and not trend_test_result.h):
                df_list_3.append(self.df[column])

            # Нет сезонности, нет тренда
            if ((correlation < 0.7) and not trend_test_result.h):
                df_list_4.append(self.df[column])

        group_1 = pd.DataFrame(df_list_1).T
        group_2 = pd.DataFrame(df_list_2).T
        group_3 = pd.DataFrame(df_list_3).T
        group_4 = pd.DataFrame(df_list_4).T

        avg_forecasts = []
        if not group_1.empty:
            print('', 'Результаты работы различных методов для ТВ-каналов с сезонностью и трендом', 
                sep = '\n', end = '\n\n')
            forecasts = []
            for i in range(len(model_name_list_group_1)):
                self.make_forecast_for_group(forecasts = forecasts,
                                            model_name = model_name_list_group_1[i], 
                                            type_of_group = 'GROUP_1', 
                                            weights_filepath = weights_filepath, 
                                            filepath = filepath_for_graphs,
                                            plots = plots)
            avg_forecast_1 = Postprocessing.calculate_average_forecast(forecasts)
            
            if plots:
                Postprocessing(group_1, avg_forecast_1).get_plot(column_name_with_date = self.column_name_with_date,
                                                                save_dir = f'{filepath_for_avg_graphs}Сезонность и тренд')
            avg_forecasts.append(avg_forecast_1)

        if not group_2.empty:
            print('', 'Результаты работы различных методов для ТВ-каналов с трендом без сезонности', sep='\n', end='\n\n')
            forecasts = []
            for i in range(len(model_name_list_group_2)):
                self.make_forecast_for_group(forecasts = forecasts,
                                            model_name = model_name_list_group_2[i], 
                                            type_of_group = 'GROUP_2', 
                                            weights_filepath = weights_filepath, 
                                            filepath = filepath_for_graphs,
                                            plots = plots)
            avg_forecast_2 = Postprocessing.calculate_average_forecast(forecasts)

            if plots:
                Postprocessing(group_2, avg_forecast_2).get_plot(column_name_with_date = self.column_name_with_date,
                                                                save_dir = f'{filepath_for_avg_graphs}Тренд без сезонности')
            avg_forecasts.append(avg_forecast_2)

        if not group_3.empty:
            print('', 'Результаты работы различных методов для ТВ-каналов с сезонностью и без тренда', 
                  sep = '\n', end = '\n\n')
            forecasts = []
            for i in range(len(model_name_list_group_3)):
                self.make_forecast_for_group(forecasts = forecasts,
                                            model_name = model_name_list_group_3[i],
                                            type_of_group = 'GROUP_3', 
                                            weights_filepath = weights_filepath, 
                                            filepath = filepath_for_graphs,
                                            plots = plots)
            avg_forecast_3 = Postprocessing.calculate_average_forecast(forecasts)

            if plots:
                Postprocessing(group_3, avg_forecast_3).get_plot(column_name_with_date = self.column_name_with_date,
                                                                save_dir = f'{filepath_for_avg_graphs}Сезонность без тренда')
            avg_forecasts.append(avg_forecast_3)

        if not group_4.empty:
            print('','Результаты работы различных методов для ТВ-каналов без сезонности и без тренда', 
                  sep = '\n', end = '\n\n')
            forecasts = []
            for i in range(len(model_name_list_group_4)):
                self.make_forecast_for_group(forecasts = forecasts,
                                            model_name = model_name_list_group_4[i], 
                                            type_of_group = 'GROUP_4', 
                                            weights_filepath = weights_filepath, 
                                            filepath = filepath_for_graphs,
                                            plots = plots)
            avg_forecast_4 = Postprocessing.calculate_average_forecast(forecasts)

            if plots:
                Postprocessing(group_4, avg_forecast_4).get_plot(column_name_with_date = self.column_name_with_date,
                                                                save_dir = f'{filepath_for_avg_graphs}Без сезонности и без тренда')
            avg_forecasts.append(avg_forecast_4) 
            
        general_df = Postprocessing.testing(self.df, *avg_forecasts, *avg_forecasts)
        #TODO Добавить testing
        return general_df