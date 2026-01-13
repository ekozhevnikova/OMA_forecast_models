import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Arial')
import warnings
warnings.filterwarnings('ignore')

from OMA_tools.federal.channel_forecast.grid_preprocessing import *


class ChannelAnalysisMaster:
    """
        Класс, в котором реализованы пайплайны для выгрузки данных из БД Mediascope, 
        а также обновление исторической сетки VIMB, освовываясь на отчете Размещение/Сводная таблица.

        !!! В А Ж Н О !!!
        Класс работает только для конкретного канала!
    """
    def __init__(
        self, date_filter: list,
        company_filter: str, basedemo_filter: str,
        auedience_file: str, web_file: str, 
        weighted_share_file: str,
        new_vimb_grids: str,
        hist_vimb_file: str
        ):

        """
            Атрибуты класса
                auedience_file: str: Полный путь к файлу с Total TV Auedience для какого-то конкретного канала.
                web_file: str: Полный путь к файлу с исторической сеткой Mediascope для какого-то конкретного канала.
                weighted_share_file: str: Полный путь к файлу со взвешенной долей и исторической сеткой Mediascope для какого-то конкретного канала.
                new_vimb_grids: Полный путь к файлам с новыми сетками для какого-то конкретного канала.
                hist_vimb_file: Полный путь к файлу с исторической сеткой VIMB для какого-то конкретного канала.
            
        """
        self.date_filter = date_filter
        self.company_filter = company_filter
        self.basedemo_filter = basedemo_filter
        self.auedience_file = auedience_file
        self.web_file = web_file
        self.weighted_share_file = weighted_share_file
        self.new_vimb_grids = new_vimb_grids
        self.hist_vimb_file = hist_vimb_file
    

    def auedience_pipeline(self):
        """
            Пайплайн для выгрузки и записи Auedience в файл для какого-то конкретного Федерального канала и конкретной БЦА.
        """
        A_parser = AuedienceParser(self.auedience_file)

        # 1. Выгрузка новых данных по Total TV Auedience
        auedience_new = A_parser.auedience_by_slots(self.date_filter, self.company_filter, self.basedemo_filter)

        # 2. Обновление таблицы
        print('Обновляю файл с Total TV Auedience. Пожалуйста, подождите ...')
        self.total_tv_auedience = A_parser.update_table_auedience(auedience_new)

        # 3. Сохранение в файл
        print('Сохраняю файл с Total TV Auedience. Пожалуйста, подождите ...')
        A_parser.make_style_of_auedience_table(self.total_tv_auedience, 'Sheet1')

        return self.total_tv_auedience
    

    def mediascope_web_pipeline(self):
        """
            Пайплайн для выгрузки и записи Auedience в файл для какого-то конкретного Федерального канала и конкретной БЦА.
        """
        plmrs_parser = MediascopeParser(self.web_file)
        # 1. Выгрузка новых исторических данных
        web_new = plmrs_parser.make_web(self.date_filter, self.company_filter, self.basedemo_filter)

        # 2. Обновление таблицы
        print('Обновляю файл с исторической сеткой Mediascope. Пожалуйста, подождите ...')
        self.web_df = plmrs_parser.update_web_table(web_new)

        # 3. Сохранение в файл
        print('Сохраняю файл с исторической сеткой Mediascope. Пожалуйста, подождите ...')
        plmrs_parser.make_style_of_web_table(self.web_df, 'Sheet1')

        # Если нужно вернуть в строковый формат
        web_new['Дата'] =  web_new['Дата'].dt.strftime('%Y-%m-%d')
        web_new['Время выхода'] =  web_new['Время выхода'].dt.strftime('%H:%M:%S')
        web_new['Время окончания'] =  web_new['Время окончания'].dt.strftime('%H:%M:%S')

        return web_new, self.web_df
    

    def plmrs_web_pipeline(
                self,
                web_new,
                start_time_col: str = 'Время выхода', 
                end_time_col: str = 'Время окончания',
                date_col: str = 'Дата'):
        """
            Пайплайн для обновления и записис в файл рассчитанных взвешенных долей для какого-то конкретного Федерального канала и конкретной БЦА.
        """
        parser = TVPreprocessing(web_new)

        print('Считаю взвешенную долю. Пожалуйста, подождите ...')
        # 1. Расчет взвешенной доли
        new_df, shares = parser.process_daily_weighted_shares(self.total_tv_auedience)

        # 2. Обновление таблицы
        new_df['Дата'] = pd.to_datetime(new_df['Дата'])
        updated = MediascopeParser(self.weighted_share_file).update_web_table(new_df)

        # 3. Сохранение в файл
        print('Обновляю файл со взвешенной долей и исторической сеткой Mediascope. Пожалуйста, подождите ...')
        parser.make_plmrs_style_of_table(self.weighted_share_file, updated, 'Sheet1')

        return updated
    

    def vimb_web_pipeline(self):
        """
            Пайплайн для обновления исторической сетки VIMB.
        """
        # 1. Составление таблицы с новой сеткой
        combined = VIMBGridProcessor(self.new_vimb_grids).parse_new_vimb_grids()

        # 2. Обновление файла с историческими данными
        VIMBGridProcessor(self.hist_vimb_file).update_vimb_file(combined)

    

    def unified_pipeline(self):
        """
            Объединенный пайплайн из всех выгрузок.
        """
        # 1. Выгрузка и обновление файла с Total TV Auedience.
        self.total_tv_auedience = self.auedience_pipeline()

        # 2. Выгрузка исторической сетки Mediascope и обновление файла с исторической сеткой.
        web_new, self.web_df = self.mediascope_web_pipeline()

        # 3. Расчет взвешенных долей через веса слотов и обновление файла с исторической сеткой.
        weighted_shares = self.plmrs_web_pipeline(web_new)

        # 4. Обновление сетки VIMB
        self.vimb_web_pipeline()

        return weighted_shares
