import numpy as np
import pandas as pd

class Postprocessing:
    """
        Класс для постобработки ДатаФреймов Временных Рядов, полученных в результате ML-моделей.
    """
    def __init__(self, ts):
        self.ts = ts

    @staticmethod
    def calculate_average_forecast(list_of_forecasts: list):
        """
        Объединяет несколько DataFrame с прогнозами, суммируя столбцы с одинаковыми названиями.
            Args: 
                forecasts: Список DataFrame с прогнозами.
            Returns: 
                avg_forecast: DataFrame с суммированными прогнозами.
        """
        combined_forecasts = pd.concat(list_of_forecasts, axis = 1)
        avg_forecast = combined_forecasts.groupby(combined_forecasts.columns, axis = 1).sum()
        return avg_forecast
    

    def testing(self, *avg_forecasts):
        if not avg_forecasts:  # Проверка, есть ли хотя бы один прогноз
            raise ValueError('Не передано ни одного прогноза для тестирования.')

        general_df = pd.concat(avg_forecasts, axis = 1) # Объединяем доступные прогнозы
        general_df = general_df[self.ts.columns]  # Упорядочиваем колонки в соответствии с исходными данными
        return 'Объединенный DataFrame прогнозов:\n', general_df