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

    def seasonal_decomposition(self, past_values: int = 3, method: str = 'calendar_years'):
        """
        Декомпозиция временного ряда с выделением тренда и сезонности.
                Args:
                    past_values: Количество предыдущих фикс.периодов/календарных лет, по которым выполняется прогноз (по умолчанию 3)
                    method: Метод декомпозиции ('fixed_periods' или 'calendar_years').
                Returns:
                    Новый ДатаФрейм с прогнозом
        """

        # Проверка количества доступных данных
        if len(self.df) < past_values * 12:
            raise ValueError(
                f"Недостаточно данных для анализа. Требуется как минимум {past_values * 12} месяцев данных.")

        if method not in ['fixed_periods', 'calendar_years']:
            raise ValueError("Недопустимый метод. Используйте 'fixed_periods' или 'calendar_years'.")

        if method == 'fixed_periods':
            # Данные за последние past_values лет
            start_date = self.df.index[-1] - pd.DateOffset(years=past_values)
            past_years = self.df[self.df.index >= start_date]

            if len(past_years) < past_values * 12:
                raise ValueError(
                    "Недостаточно данных для метода 'fixed_periods'. Убедитесь, что данных хватает на указанное количество лет.")

        elif method == 'calendar_years':
            last_date = self.df.index.max()
            end_year = last_date.year
            last_month = last_date.month

            # Данные за последние past_values календарных лет
            if last_month < 12:
                past_years = self.df[(self.df.index >= f'{end_year - past_values}-01-01') &
                                     (self.df.index < f'{end_year}-01-01')]
            else:
                past_years = self.df[(self.df.index >= f'{end_year - (past_values - 1)}-01-01') &
                                     (self.df.index < f'{end_year + 1}-01-01')]

            if len(past_years) < past_values * 12:
                raise ValueError(
                    "Недостаточно данных для метода 'calendar_years'. Убедитесь, что данных хватает на указанное количество календарных лет.")

        # Удаляем тренд с помощью дифференцирования
        detrended = past_years.diff().dropna()
        detrended['month'] = detrended.index.month
        average_season_k = detrended.groupby('month').mean()
        average_season_val = pd.concat([average_season_k] * (len(past_years) // 12), ignore_index=True)

        # Выделяем тренд
        trend = past_years.reset_index(drop=True) - average_season_val
        trend = trend.dropna()

        # Прогноз тренда
        average_step = detrended.mean()
        last_value = trend.iloc[-1]
        next_year_trend = [last_value + (i + 1) * average_step for i in range(self.forecast_periods)]

        # Даты для прогноза
        next_year_start = past_years.index[-1] + pd.DateOffset(months=1)
        next_year_dates = pd.date_range(
            start=next_year_start,
            periods=self.forecast_periods,
            freq='MS'
        )

        # Преобразуем результат в DataFrame
        next_year_trend_df = pd.DataFrame(next_year_trend, index=next_year_dates, columns=self.df.columns)
        season_forecast_df = pd.DataFrame(
            average_season_val.values[:self.forecast_periods],
            index=next_year_dates,
            columns=self.df.columns
        )

        # Финальный прогноз: тренд + сезонность
        final_forecast = next_year_trend_df + season_forecast_df

        return final_forecast

    def rolling_mean_fixed_periods(self, past_values: int = 3):
        """
            Декомпозиция временного ряда с применением скользящего среднего.
                Args:
                    past_values: Количество предыдущих фикс.периодов, по которым выполняется прогноз (по умолчанию 3).
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        # Выделение данных за последние годы
        past_years = self.df[self.df.index >= self.df.index[-1] - pd.DateOffset(years=past_values)]
        # Скользящее среднее за 12 месяцев
        rolling_mean = past_years.rolling(window=13).mean()
        detrended = (past_years - rolling_mean).dropna()
        detrended['month'] = detrended.index.month

        # Прогнозируем сезонность
        aver_season = detrended.groupby('month').mean()
        seasonatily_pred = pd.concat([aver_season] * (len(past_years) // 12), ignore_index=True)
        seasonatily = seasonatily_pred.iloc[:self.forecast_periods]

        # Прогнозируем скользящее среднее (тренд)
        steps = rolling_mean.diff().dropna()
        # Усредняем шаги
        average_step = np.abs(steps.mean())

        # Прогнозируем скользящее среднее на следующий год, добавляя усреднённый шаг к последнему значению скользящего среднего
        last_value = rolling_mean.iloc[-1]
        next_year_rolling_mean = [last_value + (i + 1) * average_step for i in range(self.forecast_periods)]
        next_year_dates = pd.date_range(
            start=past_years.index[-1],
            periods=self.forecast_periods,
            freq='MS'
        )
        # Преобразуем результат в DataFrame
        next_year_rolling_mean_df = pd.DataFrame(next_year_rolling_mean, index=next_year_dates, columns=self.df.columns)
        # Прогноз сезонности
        seasonal_forecast = np.tile(seasonatily.values, (1, 1)).T
        # Финальный прогноз: сложение скользящего среднего с прогнозом сезонности
        final_forecast = next_year_rolling_mean_df + seasonal_forecast.T
        # Создание DataFrame с прогнозом
        forecast_df = pd.DataFrame(final_forecast, index=next_year_dates, columns=self.df.columns)

        return forecast_df

    def rolling_mean_calendar_years(self, past_values: int = 3):

        """
            Декомпозиция временного ряда с применением скользящего среднего.
                Args:
                    past_values: Количество предыдущих календарных лет, по которым выполняется прогноз (по умолчанию 3).
                Returns:
                    Новый ДатаФрейм с прогнозом
        """
        # Находим последние три полных календарных года
        last_date = self.df.index.max()
        end_year = last_date.year
        last_month = last_date.month

        # Формируем последние три года данных
        if last_month < 12:
            past_years = self.df[(self.df.index >= f'{end_year - past_values}-01-01') &
                                 (self.df.index < f'{end_year}-01-01')]

        else:
            past_years = self.df[(self.df.index >= f'{end_year - (past_values - 1)}-01-01') &
                                 (self.df.index < f'{end_year + 1}-01-01')]

        months_to_forecast = self.forecast_periods + (last_month if last_month != 12 else 0)
        # Скользящее среднее за 12 месяцев и детрендирование
        rolling_mean = past_years.rolling(window=13).mean()
        detrended = (past_years - rolling_mean).dropna()
        detrended['month'] = detrended.index.month

        # Прогнозируем сезонность
        aver_season = detrended.groupby('month').mean()
        seasonatily_pred = pd.concat([aver_season] * (len(past_years) // 12), ignore_index=True)
        seasonatily = seasonatily_pred.iloc[:months_to_forecast]

        # Прогнозируем тренд (скользящее среднее)
        steps = rolling_mean.diff().dropna()
        average_step = np.abs(steps.mean())
        last_value = rolling_mean.iloc[-1]

        # Прогнозируем скользящее среднее на следующие 2 года (24 месяца)
        next_year_rolling_mean = [last_value + (i + 1) * average_step for i in range(months_to_forecast)]
        next_year_dates = pd.date_range(start=last_date, periods=months_to_forecast, freq='MS')
        next_year_rolling_mean_df = pd.DataFrame(next_year_rolling_mean, index=next_year_dates, columns=self.df.columns)

        seasonal_forecast = np.tile(seasonatily.values, (1, 1)).T
        # Финальный прогноз: сложение тренда и сезонности
        final_forecast = next_year_rolling_mean_df + seasonal_forecast.T
        forecast_df = pd.DataFrame(final_forecast, index=next_year_dates, columns=self.df.columns)

        next_year_dates_fact = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=self.forecast_periods,
                                             freq='ME')
        if last_month < 12:
            forecast_df = forecast_df.iloc[last_month:].set_index(next_year_dates_fact)

        return forecast_df

    # def decomposition_with_trend_fixed_periods(self, years_count, w_1=0.2, w_2=0.8):
    #     #TODO выбирать количество исторических значений; оставить вес тут или дать пользователю?
    #     # С трендом мб только по трем годам
    #     '''
    #         Создает новый DataFrame с спрогнозированными значениями.
    #         Прогноз значений выполняется на основе указанного количества предыдущих
    #         периодов (один период равен 12 месяцев), путем выделения из этих периодов сезонности и тренда.
    #
    #         Сезонность рассчитывается как среднее нормированных значений за указанные периоды.
    #
    #         Для прогноза на основе одного и двух прошлых периодов, считаем, что тренда нет.
    #         Будущее значение тренда на основе трёх прошлых периодов рассчитывается как сумма предыдущего значения и средней
    #         разницы между средними значениями прериодов (где разность между средними значениями для разных периодов имеет
    #         разные весовые коэффициенты).
    #
    #
    #         :param df: Исходный DataFrame
    #         :param years_count: Количество предыдущих периодов для анализа (1, 2 или 3)
    #         :param w_1: Вес для разности между 1 и 2 периодом (по умолчанию 0.2)
    #         :param w_2: Вес для разности между 2 и 3 периодом (по умолчанию 0.8)
    #         :return:  Новый DataFrame, список рассатриваемых исторических значений
    #         '''
    #
    #     if len(self.df) < 12 * years_count:
    #         raise ValueError("В DataFrame недостаточно данных для указанного количества периодов.")
    #
    #     yearly_means = []
    #     last_n_list = []
    #     yearly_normalized = []
    #     means_df = pd.DataFrame()
    #
    #     # Периоды для анализа
    #     for i in range(years_count):
    #         if i == 0:
    #             last_n = self.df.iloc[-12:]
    #         elif i == 1:
    #             last_n = self.df.iloc[-24:-12]
    #         elif i == 2:
    #             last_n = self.df.iloc[-36:-24]
    #
    #         # Добавляем значения прошлых периодов в список и для каждого периода раасчитываем среднее
    #         last_n_list.append(last_n)
    #         last_n_mean = last_n.mean()
    #
    #         # Добавляем средние значения за кждый период в список
    #         yearly_means.append(last_n_mean)
    #         means_df[f'Period_{i + 1}'] = last_n_mean
    #
    #         # Рассчитываем нормированные значения и добавляем в список, сбрасывая индексы
    #         last_n_normalized = last_n / last_n_mean
    #         yearly_normalized.append(last_n_normalized.reset_index(drop=True))
    #
    #     # Рассчитываем среднее для нормированных значений для прогноза сезонности
    #     average_normalized = sum(yearly_normalized) / years_count
    #     # Рассчитываем среднее для всех значений для прогноза тренда
    #     overall_mean = sum(yearly_means) / years_count
    #
    #     # Прогноз по одному или двум прошлым периодам
    #     if (years_count == 1) or (years_count == 2):
    #         lst_overall_mean = overall_mean.tolist()
    #         result_df = average_normalized * lst_overall_mean
    #     # Прогноз по трем прошлым периодам
    #     if years_count == 3:
    #         step = (means_df['Period_2'] - means_df['Period_3']) * w_1 + (
    #                 means_df['Period_1'] - means_df['Period_2']) * w_2
    #         forecast_average = step + means_df['Period_1']
    #         result_df = average_normalized * forecast_average
    #
    #     # Обновляем индекс
    #     new_index = pd.date_range(start=self.df.index.max() + pd.DateOffset(months=1), periods=12, freq='MS')
    #     result_df.index = new_index
    #
    #     return result_df, last_n_list
    #
    # def decomposition_with_trend_calendar_years(self, years_count, w_1=0.2, w_2=0.8):
    #     #TODO выбирать количество исторических значений; оставить вес тут или дать пользователю?
    #     # С трендом мб только по трем годам
    #     '''
    #         Создает новый DataFrame с спрогнозированными значениями.
    #         Прогноз значений выполняется на основе указанного количества предыдущих лет, путем выделения из них сезонности и
    #         тренда.
    #
    #         Прогноз выполняется на 12 месяцев, даже если исходный ряд не заканчивается концом календарного года.
    #
    #         :param df: Исходный DataFrame
    #         :param years_count: Количество предыдущих периодов для анализа (1, 2 или 3)
    #         :param w_1: Вес для разности между 1 и 2 периодом (по умолчанию 0.2)
    #         :param w_2: Вес для разности между 2 и 3 периодом (по умолчанию 0.8)
    #         :return:  Новый DataFrame, список рассматриваемых исторических значений
    #         '''
    #
    #     # Получение текущего года и последнего месяца
    #     last_date = self.df.index.max()
    #     last_year = last_date.year
    #     last_month = last_date.month
    #
    #     current_year = self.df.index.year.max()
    #
    #     # Проверка, что данных достаточно
    #     if (current_year - years_count + 1) not in self.df.index.year.unique():
    #         raise ValueError("В DataFrame недостаточно данных для указанного количества периодов.")
    #
    #     yearly_means = []
    #     yearly_normalized = []
    #     means_df = pd.DataFrame()
    #     last_n_list = []
    #
    #     # Периоды для анализа
    #     for i in range(years_count):
    #         if last_month < 12:
    #             year = current_year - (i + 1)
    #             last_n = self.df[self.df.index.year == year]
    #         else:
    #             year = current_year - (i)
    #             last_n = self.df[self.df.index.year == year]
    #
    #         # Расчет среднего значения по каждому периоду для каждого столбца
    #         last_n_list.append(last_n)
    #         last_n_mean = last_n.mean()
    #         yearly_means.append(last_n_mean)
    #         means_df[f'Period_{i + 1}'] = last_n_mean
    #
    #         # Нормализация данных для каждого периода для каждого столбца
    #         last_n_normalized = last_n / last_n_mean
    #         yearly_normalized.append(last_n_normalized.reset_index(drop=True))
    #
    #     # Расчет средних нормированных значений по всем периодам
    #     average_normalized = sum(yearly_normalized) / years_count
    #     # Расчет общего среднего значения
    #     overall_mean = sum(yearly_means) / years_count
    #
    #     # Вычисление тренда в зависимости от количества рассматриваемых лет
    #     if years_count == 1 or years_count == 2:
    #         result_df = average_normalized * overall_mean
    #     elif years_count == 3:
    #         r = (means_df['Period_2'] - means_df['Period_3']) * w_1 + (
    #                     means_df['Period_1'] - means_df['Period_2']) * w_2
    #         res_df = r + means_df['Period_1']
    #         result_df = average_normalized * res_df
    #         b = (means_df['Period_1'] - means_df['Period_2']) * w_1 + (result_df.mean() - means_df['Period_1']) * w_2
    #         r_2 = (b + result_df.mean()) * average_normalized
    #
    #     # Определение количества месяцев, которые нужно спрогнозировать
    #     months_to_forecast = 12
    #     months_remaining_current_year = 12 - last_month
    #
    #     # Если последний месяц не декабрь, сначала прогнозируем на оставшиеся месяцы текущего года
    #     if months_remaining_current_year > 0:
    #         # Прогноз на оставшиеся месяцы текущего года с использованием result_df
    #         forecast_current_year = result_df.tail(months_remaining_current_year)
    #         new_index_current_year = pd.date_range(start=last_date + pd.DateOffset(months=1),
    #                                                periods=months_remaining_current_year, freq='MS')
    #         forecast_current_year.index = new_index_current_year
    #
    #         # Прогноз на месяцы следующего года с использованием r_2
    #         months_next_year = months_to_forecast - months_remaining_current_year
    #         forecast_next_year = r_2.head(months_next_year)
    #         new_index_next_year = pd.date_range(start=new_index_current_year[-1] + pd.DateOffset(months=1),
    #                                             periods=months_next_year, freq='MS')
    #         forecast_next_year.index = new_index_next_year
    #
    #         result_df_forecast = pd.concat([forecast_current_year, forecast_next_year])
    #
    #     # Если последний месяц — декабрь, прогнозируем на следующие 12 месяцев
    #     else:
    #         result_df_forecast = result_df
    #         new_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_to_forecast, freq='MS')
    #         result_df_forecast.index = new_index
    #
    #     return result_df_forecast, last_n_list
    #
    # def decomposition_without_trend_fixed_periods(self, years_count, w_1=0.2, w_2=0.8):
    #     #TODO выбирать количество исторических значений; оставить вес тут или дать пользователю?
    #     '''
    #         Создает новый DataFrame с спрогнозированными значениями.
    #         Прогноз значений выполняется на основе указанного количества предыдущих периодов (один период равен 12 месяцев),
    #         путем выделения из этих периодов сезонности, считая, что тренд отсутствует.
    #
    #         Сезонность рассчитывается как среднее нормированных значений за указанные периоды.
    #
    #         :param df: Исходный DataFrame
    #         :param years_count: Количество предыдущих периодов для анализа (1, 2 или 3)
    #         :return:  Новый DataFrame, список рассатриваемых исторических значений
    #         '''
    #
    #     # Проверка, что данных достаточно
    #     if len(self.df) < 12 * years_count:
    #         raise ValueError("В DataFrame недостаточно данных для указанного количества периодов.")
    #
    #     yearly_means = []
    #     yearly_normalized = []
    #     last_n_list = []
    #
    #     # Периоды для анализа
    #     for i in range(years_count):
    #         # Выбор предыдущих периодов в зависимости от количества periods_count
    #         if i == 0:
    #             last_n = self.df.iloc[-12:]
    #         elif i == 1:
    #             last_n = self.df.iloc[-24:-12]
    #         elif i == 2:
    #             last_n = self.df.iloc[-36:-24]
    #
    #         # Расчет среднего значения по каждому периоду для каждого столбца
    #         last_n_list.append(last_n)
    #         last_n_mean = last_n.mean()
    #         # Добавление в список
    #         yearly_means.append(last_n_mean)
    #
    #         # Нормализация данных для каждого периода для кадого столбца
    #         last_n_normalized = last_n / last_n_mean
    #         # Добавление в список
    #         yearly_normalized.append(last_n_normalized.reset_index(drop=True))
    #
    #     # Расчет средних нормированных значений по всем периодам
    #     average_normalized = sum(yearly_normalized) / years_count
    #     # Расчет общего среднего значения
    #     overall_mean = sum(yearly_means) / years_count
    #
    #     # Преобразование общего среднего значения в список
    #     ls = overall_mean.tolist()
    #
    #     # Создание DataFrame из умноженных значений
    #     result_df = average_normalized * ls
    #
    #     # Установка правильного индекса для новых данных
    #     new_index = pd.date_range(start=self.df.index.max() + pd.DateOffset(months=1), periods=12, freq='MS')
    #     result_df.index = new_index
    #
    #     return result_df, last_n_list
    #
    # def decomposition_without_trend_calendar_years(self, years_count, w_1=0.2, w_2=0.8):
    #     #TODO выбирать количество исторических значений; оставить вес тут или дать пользователю?
    #     '''
    #     Создает новый DataFrame с спрогнозированными значениями на 12 месяцев вперед.
    #     Если последний месяц в df не совпадает с концом года, то прогноз продолжается в следующий год.
    #
    #     Сезонность рассчитывается как среднее нормированных значений за указанные периоды, тренд отсутствует.
    #
    #     :param df: Исходный DataFrame
    #     :param years_count: Количество предыдущих периодов для анализа (1, 2 или 3)
    #     :return: Новый DataFrame с прогнозом и список исторических значений
    #     '''
    #
    #     # Получение текущего года и последнего месяца исходного ряда
    #     last_date = self.df.index.max()
    #     last_year = last_date.year
    #     last_month = last_date.month
    #     current_year = self.df.index.year.max()
    #
    #     # Проверка, что данных достаточно
    #     current_year = self.df.index.year.max()
    #     if (current_year - years_count + 1) not in self.df.index.year.unique():
    #         raise ValueError("В DataFrame недостаточно данных для указанного количества периодов.")
    #
    #     yearly_means = []
    #     yearly_normalized = []
    #     last_n_list = []
    #
    #     # Периоды для анализа
    #     if last_month < 12:
    #         for i in range(years_count):
    #             year = current_year - (i + 1)
    #             last_n = self.df[self.df.index.year == year]
    #
    #             # Расчет среднего значения по каждому периоду для каждого столбца
    #             last_n_list.append(last_n)
    #             last_n_mean = last_n.mean()
    #             yearly_means.append(last_n_mean)
    #
    #             # Нормализация данных для каждого периода для каждого столбца
    #             last_n_normalized = last_n / last_n_mean
    #             yearly_normalized.append(last_n_normalized.reset_index(drop=True))
    #
    #         # Расчет средних нормированных значений по всем периодам
    #         average_normalized = sum(yearly_normalized) / years_count
    #         overall_mean = sum(yearly_means) / years_count
    #
    #         # Преобразование общего среднего значения в список
    #         ls = overall_mean.tolist()
    #
    #         # Создание DataFrame из умноженных значений
    #         result_df = average_normalized * ls
    #
    #         # Определение месяца, с которого начинается прогноз
    #         months_remaining_current_year = 12 - last_month
    #
    #         if months_remaining_current_year > 0:
    #             # Прогнозируем оставшиеся месяцы текущего года
    #             forecast_current_year = result_df.tail(months_remaining_current_year)
    #             new_index_current_year = pd.date_range(start=last_date + pd.DateOffset(months=1),
    #                                                    periods=months_remaining_current_year, freq='MS')
    #             forecast_current_year.index = new_index_current_year
    #
    #             # Прогноз на оставшиеся месяцы следующего года
    #             months_next_year = 12 - months_remaining_current_year
    #             forecast_next_year = result_df.head(months_next_year)
    #             new_index_next_year = pd.date_range(start=new_index_current_year[-1] + pd.DateOffset(months=1),
    #                                                 periods=months_next_year, freq='MS')
    #             forecast_next_year.index = new_index_next_year
    #
    #             # Объединение прогноза на текущий и следующий год
    #             result_df_forecast = pd.concat([forecast_current_year, forecast_next_year])
    #
    #
    #     else:
    #         for i in range(years_count):
    #             year = current_year - (i)
    #             last_n = self.df[self.df.index.year == year]
    #
    #             # Расчет среднего значения по каждому периоду для каждого столбца
    #             last_n_list.append(last_n)
    #             last_n_mean = last_n.mean()
    #             yearly_means.append(last_n_mean)
    #
    #             # Нормализация данных для каждого периода для каждого столбца
    #             last_n_normalized = last_n / last_n_mean
    #             yearly_normalized.append(last_n_normalized.reset_index(drop=True))
    #
    #         # Расчет средних нормированных значений по всем периодам
    #         average_normalized = sum(yearly_normalized) / years_count
    #         overall_mean = sum(yearly_means) / years_count
    #
    #         # Преобразование общего среднего значения в список
    #         ls = overall_mean.tolist()
    #
    #         # Создание DataFrame из умноженных значений
    #         result_df = average_normalized * ls
    #
    #         # Определение месяца, с которого начинается прогноз
    #         months_remaining_current_year = 12 - last_month
    #
    #         if months_remaining_current_year > 0:
    #             # Прогнозируем оставшиеся месяцы текущего года
    #             forecast_current_year = result_df.tail(months_remaining_current_year)
    #             new_index_current_year = pd.date_range(start=last_date + pd.DateOffset(months=1),
    #                                                    periods=months_remaining_current_year, freq='MS')
    #             forecast_current_year.index = new_index_current_year
    #
    #             # Прогноз на оставшиеся месяцы следующего года
    #             months_next_year = 12 - months_remaining_current_year
    #             forecast_next_year = result_df.head(months_next_year)
    #             new_index_next_year = pd.date_range(start=new_index_current_year[-1] + pd.DateOffset(months=1),
    #                                                 periods=months_next_year, freq='MS')
    #             forecast_next_year.index = new_index_next_year
    #
    #             # Объединение прогноза на текущий и следующий год
    #             result_df_forecast = pd.concat([forecast_current_year, forecast_next_year])
    #
    #         else:
    #             # Если последний месяц — декабрь, прогнозируем на следующие 12 месяцев
    #             new_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    #             result_df_forecast = result_df
    #             result_df_forecast.index = new_index
    #
    #     return result_df_forecast, last_n_list
    #
    #
    # def auto_arima_forecast(self, plot=False):
    #
    # def prophet_forecast(self):