import sys
import numpy as np
import pandas as pd
import pymorphy3
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, 'ru_RU')
from statsmodels.tsa.stattools import adfuller
from OMA_tools.io_data.operations import File, Table, Dict_Operations
#from io_data.operations import File, Table, Dict_Operations


class Preprocessing:
    """
        Класс для предобратки Временных Рядов (ВР).
    """
    def __init__(self, ts):
        self.ts = ts


    @staticmethod
    def get_data_for_forecast(filename, list_of_replacements: list, column_name_with_date: str):
        """
            Функция для чтения данных в формате DataFrame из файла формата .xlsx с несколькими листами
            Args:
                filename: название файла с данными, для которых хотим сделать прогноз в формате .xlsx
                list_of_replacements: список названий листов, находящихся в filename
                column_name_with_date: название столбца с датой
            Returns:
                Совокупный merged_df из всех листов, находящихся в файле формата .xlsx.
        """
        dict_data = File(filename).from_file(0, 0)
        dict_data_new = Dict_Operations(dict_data).replace_keys_in_dict(list_of_replacements)
        merged_df = Dict_Operations(dict_data_new).combine_dict_into_one_dataframe(column_name_with_date)
        # Замена формата дат типа Январь 2021 на 01.01.2021
        dates = list(merged_df[column_name_with_date])
        dates_converted = []
        for i in dates:
            # Если архитектура ядра процессора Darvin (MacOS)
            if sys.platform == 'darwin':
                morph = pymorphy3.MorphAnalyzer(lang='ru')
                month, year = i.split(' ')
                p = morph.parse(month)[0]
                inflect_month_name = p.inflect({'gent'}).word
                dates_converted.append(datetime.strptime(f'{inflect_month_name} {year}', '%B %Y').strftime('%d.%m.%Y'))
            else:
                dates_converted.append(datetime.strptime(i, '%B %Y').strftime('%d.%m.%Y'))
        # Замена столбца дат на новый конвертированный столбец
        merged_df[column_name_with_date] = merged_df[column_name_with_date].replace(dates, dates_converted)
        # Изменение типа данных в столбцах
        for column in merged_df.columns[1:]:
            merged_df[column] = merged_df[column].astype(float)
        merged_df[column_name_with_date] = pd.to_datetime(merged_df[column_name_with_date], format='%d.%m.%Y')
        merged_df = merged_df.sort_values(by=column_name_with_date)
        merged_df.set_index(column_name_with_date, inplace=True)
        return merged_df


    def check_stationarity(self):
        """
            Проверка ВР на стационарность.
            p-value < 0.05 свидетельствует о стационарности ВР.
            p-value > 0.05 свидетельствует о плохой стационарности ВР.
        """
        result = adfuller(self.ts)
        return result[1] < 0.05


    def make_stationary(self):
        """
            Приведение нестационарного ряда к стационарному виду путём дифференцирования.
            Удаление тренда из ВР.
        """
        return self.ts.diff().dropna()


    def inverse_difference(self, last_observation):
        """
            Приведение ВР к размерности исходного вида с использованием кумулятивной суммы.
            Args:
                last_observation: последние фактические данные в DataFrame.
        """
        return self.ts.cumsum() + last_observation
    

    def get_season(self):
        """
            Функция для определения типа сезона по номеру месяца.
        """
        month = self.ts.index.month
        season = []

        for i in range(len(month)):
            if month[i] in range(3, 5):
                season.append(2) #spring
            elif month[i] in range(6, 8):
                season.append(3) #summer
            elif month[i] in range(9, 11):
                season.append(4) #autumn
            else:
                season.append(1) #winter
        return season

 
    def create_features(self):
        """
            Функция для создания столбцов в DataFrame с категориальными признаками.
            Преимущественно используется в модели XGBoost.
            Returns:
                DataFrame с новыми столбцами - категориальными признаками.
        """
        self.ts['year'] = self.ts.index.year
        self.ts['year start'] = self.ts.index.is_year_start
        self.ts['month'] = self.ts.index.month
        self.ts['quarter start'] = self.ts.index.is_quarter_start
        self.ts['season'] = self.get_season()
        return self.ts


    def add_lags(self):
        """
            Функция для добавления запаздывающей компоненты в ВР.
            Returns:
                Обновленный DataFrame с новыми столбцами в виде запаздывающих компонент.
        """
        target_map = self.ts['Share'].to_dict()
        self.ts['lag1'] = (self.ts.index - pd.DateOffset(months = 12)).map(target_map)
        self.ts['lag2'] = (self.ts.index - pd.DateOffset(months = 24)).map(target_map)
        self.ts['lag3'] = (self.ts.index - pd.DateOffset(months = 36)).map(target_map)
        return self.ts


    def search_last_fact_data(self):
        """
            Нахождение последнего фактического месяца и года в DataFrame.
        """
        last_date = self.ts.index.max()
        end_year = last_date.year
        last_month = last_date.month
        return end_year, last_month, last_date