import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima


class Preprocessing:
    """
        Класс для предобратки Временных Рядов (ВР).
    """
    def __init__(self, ts):
        self.ts = ts

    @staticmethod
    def check_stationarity(ts):
        """
            Проверка ВР на стационарность.
            p-value < 0.05 свидетельствует о стационарности ВР.
            p-value > 0.05 свидетельствует о плохой стационарности ВР.
        """
        result = adfuller(ts)
        return result[1] < 0.05

    @staticmethod
    def make_stationary(ts):
        """
            Приведение нестационарного ряда к стационарному виду путём дифференцирования.
            Удаление тренда из ВР.
        """
        return ts.diff().dropna()

    @staticmethod
    def inverse_difference(forecast, last_observation):
        """
            Приведение ВР к размерности исходного вида с использованием кумулятивной суммы.
            Args:
                last_observation: последние фактические данные в DataFrame.
        """
        return forecast.cumsum() + last_observation

    # def check_stationarity(self):
    #     """
    #         Проверка ВР на стационарность.
    #         p-value < 0.05 свидетельствует о стационарности ВР.
    #         p-value > 0.05 свидетельствует о плохой стационарности ВР.
    #     """
    #     result = adfuller(self.ts)
    #     return result[1] < 0.05
    #
    # def make_stationary(self):
    #     """
    #         Приведение нестационарного ряда к стационарному виду путём дифференцирования.
    #         Удаление тренда из ВР.
    #     """
    #     return self.ts.diff().dropna()
    #
    # def inverse_difference(self, last_observation):
    #     """
    #         Приведение ВР к размерности исходного вида с использованием кумулятивной суммы.
    #         Args:
    #             last_observation: последние фактические данные в DataFrame.
    #     """
    #     return self.ts.cumsum() + last_observation


    def search_last_fact_data(self):
        """
            Нахождение последнего фактического месяца и года в DataFrame.
        """
        last_date = self.ts.index.max()
        end_year = last_date.year
        last_month = last_date.month
        return end_year, last_month, last_date