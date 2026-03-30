import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import glob
from pathlib import Path
import xlsxwriter
import calendar
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import locale
locale.setlocale(locale.LC_ALL, 'ru_RU')

from OMA_tools.io_data.operations import File, Table, Dict_Operations
#from OMA_tools.regions.data_extraction.task_builder import BaseDataService
from OMA_tools.federal.channel_forecast.calculator import *
from OMA_tools.federal.channel_forecast.core.content_matching import *

import warnings
warnings.filterwarnings('ignore')


######################################### КОНСТАНТЫ #########################################
TIME_FILTER = 'timeBand1 >= 50000 AND timeBand1 < 290000'
OPTIONS = {
            "kitId": 1, #TV Index Cities  
            "totalType": "TotalChannels" #база расчета Share: Total Channels. Возможны опции: TotalTVSet, TotalChannelsThem
                    }
WEEKDAY_FILTER = None
DAYTYPE_FILTER = None
TARGETDEMO_FILTER = None
LOCATION_FILTER = None
ADD_CITY_TO_BASEDEMO_FROM_REGION = False    # работаем в Федеральной Базе
ADD_CITY_TO_TARGETDEMO_FROM_REGION = False  # работаем в Федеральной Базе
BREAK_FILTER = None
AD_FILTER = None
PROGRAM_FILTER = 'programDuration >= 100'
#############################################################################################

class BaseParser:
    """
        Базовый класс с набором базовых операций для парсинга различных файлов.
        Все классы-парсеры должны наследоваться от этого класса.
    """
    
    def __init__(self, filepath: str):
        """
            Инициализация базового парсера.
            
            Args:
                filepath: Путь к файлу для работы
        """
        self.filepath = filepath

    
    def _ensure_file_exists(self, default_columns: list = None):
        """
            Проверяет существование файла. 
            Если файл не существует, создает его с базовой структурой.
            
            Args:
                default_columns: Список колонок для создания пустого файла
        """
        if not os.path.exists(self.filepath):
            if default_columns is None:
                print('⚠️ Список колонок не передан! Пожалуйста, исправьте!')
            
            # Создаем пустой DataFrame с указанными колонками
            empty_df = pd.DataFrame(columns = default_columns)
            
            # Создаем директорию, если она не существует
            directory = os.path.dirname(self.filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok = True)
                print(f'📥 Создана директория: {directory}')
            
            # Сохраняем пустой файл
            with pd.ExcelWriter(self.filepath, engine = 'xlsxwriter') as writer:
                empty_df.to_excel(writer, sheet_name = 'Sheet1', index = False)
            
            print(f'📁 Создан новый файл: {self.filepath}')

    
    def _get_column_letter(self, col_idx: int) -> str:
        """
            Преобразует индекс колонки в буквенное обозначение Excel.
            Например: 0 -> 'A', 1 -> 'B', 25 -> 'Z', 26 -> 'AA'
        """
        col_letter = ''
        while col_idx >= 0:
            col_letter = chr(col_idx % 26 + 65) + col_letter
            col_idx = col_idx // 26 - 1
        return col_letter

    
    @staticmethod
    def format_time(time_int: int) -> str:
        """
            Функция для преобразования временного слота.
            
            Args:
                time_int: Время в формате числа (например, 50000 для 05:00:00)
            
            Returns:
                Время в формате HH:MM:SS
        """
        time_str = str(time_int).zfill(6)
        return f'{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}'
    
    
    @staticmethod
    def get_sort_key(time_str: str) -> int:
        """
            Преобразует время в числовое значение для сортировки от 05:00.
            
            Args:
                time_str: время в формате 'HH:MM:SS'
            
            Returns:
                int: количество секунд для сортировки
        """
        try:
            # Парсим время
            h, m, s = map(int, time_str.split(':'))
            
            # Если время до 05:00, добавляем 24 часа
            if h < 5:
                h += 24
            
            return h * 3600 + m * 60 + s
        
        except (ValueError, AttributeError):
            # Если возникла ошибка, возвращаем 0
            return 0

    
    @staticmethod
    def convert_time(time_str: str):
        """
            Функция для конвертации времени из формата 25:00:00 в 01:00:00 или 5:00:00 в 05:00:00
            Args:
                time_str: время в формате строки
        """
        # Предполагаем стандартный формат HH:MM:SS или H:MM:SS
        if time_str[1] == ':':  # Формат H:MM:SS (одна цифра)
            hours = int(time_str[0])
            rest = time_str[1: ]  # :MM:SS
        else:  # Формат HH:MM:SS (две цифры)
            hours = int(time_str[: 2])
            rest = time_str[2: ]  # :MM:SS
        
        # Применяем преобразование часов
        if hours >= 24:
            hours = hours - 24
        # Форматируем с ведущим нулем
        return f'{hours:02d}{rest}'
    

    def make_style_of_table(self, df: pd.DataFrame, sheet_name: str, 
                          column_configs: list, date_columns: list = None,
                          use_filters: bool = True, freeze_panes: bool = True):
        """
        Универсальный метод для стилизации таблиц в Excel.
        
        Args:
            df: DataFrame для записи
            sheet_name: Имя листа
            column_configs: Список словарей с настройками колонок
                Пример: [
                    {'header': 'Дата', 'width': 13.0, 'format': 'date'},
                    {'header': 'TimeSlot', 'width': 9.0, 'format': 'general'},
                    ...
                ]
            date_columns: Список названий колонок, содержащих даты
            use_filters: Добавлять ли автофильтры
            freeze_panes: Замораживать ли верхнюю строку
        """
        if date_columns is None:
            date_columns = ['Дата']
        
        with pd.ExcelWriter(self.filepath, engine = 'xlsxwriter') as writer:
            # Записываем данные без заголовков
            df.to_excel(writer, 
                       sheet_name = sheet_name, 
                       index = False, 
                       header = False,
                       startrow = 0)
            
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Форматы
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'align': 'center',
                'valign': 'vcenter',
                'border': 0
            })
            
            table_fmt = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter',
                'border': 0
            })
            
            date_fmt = workbook.add_format({
                'num_format': 'yyyy-mm-dd',
                'align': 'center',
                'valign': 'vcenter',
                'border': 0
            })
            
            # Записываем заголовки
            headers = [config['header'] for config in column_configs]
            for col_num, header in enumerate(headers):
                worksheet.write(0, col_num, header, header_format)
            
            # Записываем данные с правильным форматом
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    cell_value = df.iat[row_idx, col_idx]
                    
                    # Для колонки A используем формат даты
                    if col_idx == 0:
                        worksheet.write(row_idx + 1, col_idx, cell_value, date_fmt)
                    else:
                        worksheet.write(row_idx + 1, col_idx, cell_value, table_fmt)
            
            # Устанавливаем ширину колонок
            for col_idx, config in enumerate(column_configs):
                width = config.get('width', 12.0)
                worksheet.set_column(col_idx, col_idx, width)
            
            # Добавляем автофильтры
            if use_filters and len(df) > 0:
                last_row = len(df)
                last_col = len(headers) - 1
                filter_range = f'A1:{self._get_column_letter(last_col)}{last_row + 1}'
                worksheet.autofilter(filter_range)
            
            # Замораживаем верхнюю строку
            if freeze_panes:
                worksheet.freeze_panes(1, 0)


class AuedienceParser(BaseParser):
    """
        Класс для предобработки и постобработки файлов с Total TV Auedience для ОДНОГО канала
    """
    
    def __init__(self, filepath: str):
        """
            Инициализация парсера аудитории.
            
            Args:
                filepath: Путь к файлу с данными аудитории
        """
        self.filepath = filepath

        super().__init__(filepath)

        # Вызываем ensure_file_exists с нужными колонками
        self._ensure_file_exists(['Дата', 'TimeSlot', 'Auedience', 'Slot_weight', 'hour_start'])
    

    def auedience_by_slots(
            self, date_filter, company_filter, basedemo_filter,
            targets = None,
            statistics = ['TTVRtg000'],
            time_filter = TIME_FILTER,
            slices = ['researchDate', 'tvCompanyName','timeBand60'],
            sortings = {'researchDate': 'ASC', 'tvCompanyName': 'ASC'},
            options = OPTIONS,
            weekday_filter = WEEKDAY_FILTER, daytype_filter = DAYTYPE_FILTER, 
            targetdemo_filter = TARGETDEMO_FILTER, location_filter = LOCATION_FILTER):
        """
            Метод для выгрузки Auedience из БД Mediscope API для одного канала
        """
        # Формируем задачи в формате json
        tasks = BaseDataService._build_timeband_common_params(
                                                        date_filter = date_filter, company_filter = company_filter, 
                                                        basedemo_filter = basedemo_filter, regions_id = None,          # работаем в Федеральной Базе
                                                        targets = targets, time_filter = time_filter, 
                                                        statistics = statistics, slices = slices, 
                                                        sortings = sortings, options = options,
                                                        location_filter = location_filter, weekday_filter = weekday_filter,
                                                        daytype_filter = daytype_filter, targetdemo_filter = targetdemo_filter,
                                                        add_city_to_basedemo_from_region = False,   # работаем в Федеральной Базе
                                                        add_city_to_targetdemo_from_region = False  # работаем в Федеральной Базе
                                                    )
        # Отправляем задачи на расчет
        df = BaseDataService._execute_tasks(tasks)

        df['tvCompanyName'] = df['tvCompanyName'].str.replace(' (СЕТЕВОЕ ВЕЩАНИЕ)', '', regex = False)
        df.rename(columns = {'researchDate': 'Date', 'tvCompanyName': 'Channel', 'timeBand60': 'TimeSlot'}, inplace = True)
        df['Date'] = pd.to_datetime(df['Date'])
        #Приведение слота к нормальному виду
        df['TimeSlot'] = df['TimeSlot'].apply(BaseParser.format_time)
        df['TimeSlot'] = df['TimeSlot'].apply(BaseParser.convert_time)
        
        df['TimeSlot_dt'] = pd.to_datetime(df['TimeSlot'], format='%H:%M:%S')

        # Сортируем по времени
        df_sorted = df.sort_values('TimeSlot_dt')

        # Удаляем временную колонку если нужно
        df_sorted = df_sorted.drop('TimeSlot_dt', axis = 1)
        df_sorted.reset_index(drop = True)
        data = df_sorted[['Channel', 'Date', 'TimeSlot', 'TTVRtg000']]
        data_by_slots = data.sort_values('Date').reset_index(drop = True)
        res = data_by_slots[data_by_slots['TTVRtg000'] != 0.0]
        final_data = res[['Date', 'TimeSlot', 'TTVRtg000']].reset_index(drop = True)

        sorted_df = final_data.sort_values(['Date', 'TimeSlot'], ascending = [True, True])
        sorted_df.rename(columns = {'TTVRtg000': 'Auedience', 'Date': 'Дата'}, inplace = True)

        sorted_df.reset_index(drop = True)

        sorted_df['Дата'] = pd.to_datetime(sorted_df['Дата'])
        sorted_df['Auedience'] = sorted_df['Auedience'].astype(float)

        # Расчет веса слотов
        A = TVShareCalculator.calculate_slot_weights(sorted_df)

        A['Auedience'] = A['Auedience'].round(5)
        A['Slot_weight'] = A['Slot_weight'].round(8)

        return A

    
    def update_table_auedience(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
            Метод для обновления таблицы с Auedience
        """
        new = pd.DataFrame()

        # Чтение данных из файла
        old_data = pd.read_excel(f'{self.filepath}')
        old_data['Дата'] = pd.to_datetime(old_data['Дата'])
        old_data['Auedience'] = old_data['Auedience'].astype(float)

        # Отбираем уникальные даты из старых и новых данных
        old_unique_dates = old_data['Дата'].unique()
        new_unique_dates = new_data['Дата'].unique()
        
        old_ones = []

        # Фильтруем даты, которые уже присутствуют в данных
        for new_date in new_unique_dates:
                
            if new_date in old_unique_dates:
                old_ones.append(pd.to_datetime(new_date))

        if len(old_ones) != 0:
            min_date_str = min(old_ones).strftime('%Y-%m-%d')

            # Оставляем только те даты, которые не встречаются в новых, если таковые нашлись
            filtered = old_data[old_data['Дата'] < min_date_str]

            if len(filtered) != 0:
        
                # Обновляем таблицу с фактическими данными
                new = pd.concat([filtered, new_data]).reset_index(drop = True)
        
        # В противном случае просто добавляем новые данные в конец старой таблицы
        else:
            new = pd.concat([old_data, new_data]).reset_index(drop = True)

        sorted_by_dates = new.sort_values('Дата').reset_index(drop = True)
        self.total_tv_auedience = sorted_by_dates.sort_values(['Дата', 'TimeSlot'], ascending = [True, True])

        # Если нужно вернуть в строковый формат
        self.total_tv_auedience['Дата'] = self.total_tv_auedience['Дата'].dt.strftime('%Y-%m-%d')
        
        return self.total_tv_auedience
    

    def make_style_of_auedience_table(self, df: pd.DataFrame, sheet_name: str):
        """
            Функция для генерации внешнего вида таблицы с аудиторией.
        """
        column_configs = [
            {'header': 'Дата', 'width': 13.0, 'format': 'date'},
            {'header': 'TimeSlot', 'width': 9.0, 'format': 'general'},
            {'header': 'Auedience', 'width': 11.0, 'format': 'general'},
            {'header': 'Slot_weight', 'width': 14.0, 'format': 'general'},
            {'header': 'hour_start', 'width': 12.0, 'format': 'general'}
        ]
        
        self.make_style_of_table(
            df = df,
            sheet_name = sheet_name,
            column_configs = column_configs,
            date_columns = ['Дата']
        )



class MediascopeParser(BaseParser):
    """
        Класс для работы с данными Mediascope
    """
    
    def __init__(self, channel: str, web_filepath: str):
        """
            Инициализация парсера Mediascope.
            
            Args:
                web_filepath: Путь к файлу с исторической сеткой Mediascope
        """
        super().__init__(web_filepath)

        # Список допустимых названий каналов
        allowed_channels = [
            'ТНТ4', '2X2', 'КАРУСЕЛЬ', 'СУББОТА', 
            'СТСЛав', 'ЗВЕЗДА', 'МИР', 'МатчТВ', 
            'МузТВ', 'СОЛНЦЕ', 'СПАС', 'ТВЦ', 'ЧЕ', 'Ю'
            ]
        
        # Проверка наличия канала в списке допустимых
        if channel not in allowed_channels:
            raise ValueError(
                f"Канал '{channel}' не существует. Выберите канал из списка: {', '.join(allowed_channels)}"
            )
        
        self.channel = channel

        self.web_filepath = web_filepath

        # Вызываем ensure_file_exists с нужными колонками
        self._ensure_file_exists([
            'Канал', 'Дата', 'Название программы', 'Время выхода',
            'Время окончания', 'Продолжительность', 'Share', 'Жанр', 'День недели'
        ])

    

    @staticmethod
    def sort_time(t):
        """
            Преобразует время в числовое значение для сортировки.
            Значения до 05:00 получают +24 часа, чтобы оказаться после 23:59
        """
        if t.hour < 5:
            total_seconds = (t.hour + 24) * 3600 + t.minute * 60 + t.second
        else:
            total_seconds = t.hour * 3600 + t.minute * 60 + t.second
        
        return total_seconds


    def make_web(self,
            date_filter, company_filter, basedemo_filter,
            weekday_filter = WEEKDAY_FILTER, daytype_filter = DAYTYPE_FILTER, 
            location_filter = LOCATION_FILTER, targetdemo_filter = TARGETDEMO_FILTER, 
            break_filter = BREAK_FILTER, ad_filter = AD_FILTER, 
            program_filter = PROGRAM_FILTER, 
            slices = ['programSpotId',                # Программа ID выхода, обязательный атрибут! 
                      'researchDate',                 # Дата, обязательный атрибут! 
                      'programName',                  # Название программы
                      'tvCompanyName',                # Телекомпания
                      'researchWeekDay',              # День недели
                      'programStartTime',             # Программа время начала
                      'programFinishTime',            # Программа время окончания
                      'programCategoryName',          # Программа категория
                      'programIssueDescriptionName',  # Программа описание выпуска
                      'programProducerYear'           # Программа дата создания
                        ], 
            statistics = ['Share'], 
            sortings = {'tvCompanyName': 'ASC', 'researchDate': 'ASC', 'programStartTime': 'ASC'},
            options = {
                       "kitId": 1 #TV Index Russia all
                   }
            ) -> pd.DataFrame:
            """
                Метод для выгрузки исторической сетки из БД Mediascope
            """
            # 1. Формируем задачи в формате json для отправки на сервер
            tasks = BaseDataService._build_simple_common_params(
                                        date_filter, company_filter, basedemo_filter, 
                                        weekday_filter, daytype_filter, location_filter,
                                        targetdemo_filter, break_filter, ad_filter, 
                                        program_filter, slices, statistics, sortings, options
                                        )
            
            # 2. Расчёт задач
            df = BaseDataService._execute_simple_tasks(tasks)

            # Приводим порядок столбцов в соответствие с условиями расчета
            df = df[slices + statistics]

            df.rename(columns = {'researchDate': 'Date'}, inplace = True)
            df = df[
                [
                    'tvCompanyName', 'Date', 'programName', 'programIssueDescriptionName',
                    'programStartTime', 'programFinishTime', 'Share', 
                    'programCategoryName', 'researchWeekDay'
                    ]
                    ]
            df['tvCompanyName'] = df['tvCompanyName'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))
            
            df['programStartTime'] = df['programStartTime'].astype(str).apply(BaseParser.convert_time)
            df['programStartTime'] = pd.to_datetime(df['programStartTime'], format = '%H:%M:%S', errors = 'coerce')
            
            df['programFinishTime'] = df['programFinishTime'].astype(str).apply(BaseParser.convert_time)
            df['programFinishTime'] = pd.to_datetime(df['programFinishTime'], format = '%H:%M:%S', errors = 'coerce')
            
            # Для канала МатчТВ объединяем столбцы 'Назване программы' и 'Описание программы'.
            if self.channel == 'МатчТВ':

                df.rename(columns = {
                    'tvCompanyName': 'Канал', 
                    'Date': 'Дата', 
                    'programName': 'Название программы', 
                    'programIssueDescriptionName': 'Описание программы',
                    'programStartTime': 'Время выхода', 
                    'programFinishTime': 'Время окончания', 
                    'programCategoryName': 'Жанр',
                    'researchWeekDay': 'День недели'}, inplace = True)
                
                # Сначала фильтруем строки
                mask = (df['Жанр'] == 'Трансляция спортивного мероприятия') & \
                    (df['Описание программы'].notna()) & \
                    (df['Описание программы'] != '')

                # Создаем новую колонку с объединенным названием
                df.loc[mask, 'Название программы'] = df.loc[mask, 'Название программы'] + '. ' + df.loc[mask, 'Описание программы']

                # Удаляем остальные строки (не спортивные)
                df_sport = df[mask].reset_index(drop = True)
                df_remained = df[~mask].reset_index(drop = True)

                need_columns = [
                                'Канал', 'Дата', 'Название программы', 
                                'Время выхода', 'Время окончания', 'Share', 
                                'Жанр', 'День недели'
                            ]
                # Оставляем нужные колонки
                df_sport = df_sport[need_columns]
                df_remained = df_remained[need_columns]

                df = pd.concat([df_sport, df_remained]).reset_index(drop = True)

                # Устанавливаем правильные сортировки для столбцов с датой и временем начала программы
                df['Дата'] = pd.to_datetime(df['Дата'], errors = 'coerce')
                df = df.sort_values('Дата').reset_index(drop = True)

            
            elif self.channel == 'МузТВ':

                df.rename(columns = {
                    'tvCompanyName': 'Канал', 
                    'Date': 'Дата', 
                    'programName': 'Название программы', 
                    'programIssueDescriptionName': 'Описание программы',
                    'programStartTime': 'Время выхода', 
                    'programFinishTime': 'Время окончания', 
                    'programCategoryName': 'Жанр',
                    'researchWeekDay': 'День недели'}, inplace = True)
                
                stop_words = ['засеки звезду', 'proклип', 'proновости. специальный выпуск']
                pattern = '|'.join(stop_words)
                df = df[~df['Название программы'].str.contains(pattern, case = False, na = False)]

                # Объединяем оба жанра в одну маску
                mask_doc = ((df['Жанр'] == 'Документальный сериал') | (df['Жанр'] == 'Документальный фильм')) & \
                    (df['Описание программы'].notna()) & \
                    (df['Описание программы'] != '')

                # Применяем изменения для обоих жанров (добавляем описание к названию)
                df.loc[mask_doc, 'Название программы'] = df.loc[mask_doc, 'Название программы'] + ' ' + df.loc[mask_doc, 'Описание программы']

                # Удаляем подстроку 'Сезон N/A / ' из названия программы (для всех строк)
                #df['Название программы'] = df['Название программы'].str.replace('Сезон N/A / ', '', regex = False)
                df['Название программы'] = df['Название программы'].str.replace('Личное дело', '', regex = False)

                # Оставляем нужные колонки
                need_columns = [
                    'Канал', 'Дата', 'Название программы', 
                    'Время выхода', 'Время окончания', 'Share', 
                    'Жанр', 'День недели'
                ]

                df = df[need_columns].reset_index(drop = True)

            
            elif self.channel == 'ТВЦ':
                df.rename(columns = {
                    'tvCompanyName': 'Канал', 
                    'Date': 'Дата', 
                    'programName': 'Название программы', 
                    'programIssueDescriptionName': 'Описание программы',
                    'programStartTime': 'Время выхода', 
                    'programFinishTime': 'Время окончания', 
                    'programCategoryName': 'Жанр',
                    'researchWeekDay': 'День недели'}, inplace = True)
                

                # Удаляем 'Документальное кино Леонида Млечина' из столбца "Название программы"
                # До 2025.12.31 в районе 02:10 вместо документого фильма стояла программа 'Документальное кино Леонида Млечина'.
                # Если появится какая-то другая программа, то надо будет настроить удаление аналогичным образом
                df['Название программы'] = df['Название программы'].str.replace('Документальное кино Леонида Млечина', 'Документальный сериал', regex = False)

                # Схлопываем столбцы с проверкой на пустые значения
                df['Название программы'] = np.where(
                    (df['Описание программы'].notna()) & (df['Описание программы'] != ''),
                    df['Название программы'] + ' ' + df['Описание программы'],
                    df['Название программы']
                )

                # Оставляем нужные колонки
                need_columns = [
                    'Канал', 'Дата', 'Название программы', 
                    'Время выхода', 'Время окончания', 'Share', 
                    'Жанр', 'День недели'
                ]

                df = df[need_columns].reset_index(drop = True)
            

            elif self.channel in ['СПАС', 'ЗВЕЗДА']:
                df.rename(columns = {
                    'tvCompanyName': 'Канал', 
                    'Date': 'Дата', 
                    'programName': 'Название программы', 
                    'programIssueDescriptionName': 'Описание программы',
                    'programStartTime': 'Время выхода', 
                    'programFinishTime': 'Время окончания', 
                    'programCategoryName': 'Жанр',
                    'researchWeekDay': 'День недели'}, inplace = True)
                

                # Схлопываем столбцы с проверкой на пустые значения
                df['Название программы'] = np.where(
                    (df['Описание программы'].notna()) & (df['Описание программы'] != ''),
                    df['Название программы'] + ' ' + df['Описание программы'],
                    df['Название программы']
                )

                # Оставляем нужные колонки
                need_columns = [
                    'Канал', 'Дата', 'Название программы', 
                    'Время выхода', 'Время окончания', 'Share', 
                    'Жанр', 'День недели'
                ]

                df = df[need_columns].reset_index(drop = True)

        
            else:
                df.rename(columns = {
                    'tvCompanyName': 'Канал', 
                    'Date': 'Дата', 
                    'programName': 'Название программы', 
                    'programStartTime': 'Время выхода', 
                    'programFinishTime': 'Время окончания', 
                    'programCategoryName': 'Жанр',
                    'researchWeekDay': 'День недели'}, inplace = True)


            # Удаляем 'Сезон N/A / ' из столбца "Описание программы"
            df['Название программы'] = (df['Название программы']
                                        .str.replace('Сезон N/A / ', '', regex = False)
                                        .str.replace('Серия N/A', '', regex = False)
                                        .str.replace('Сезон N/A', '', regex = False))
            
            if self.channel == 'СУББОТА':
                df = df[~df['Название программы'].str.contains('малышарики. умные песенки', case = False, na = False)]
            
            elif self.channel == 'Ю':
                df = df[~df['Название программы'].str.contains('про семью', case = False, na = False)]
                 
            time_slots_columns = ['Время выхода', 'Время окончания']
            for i in range(len(time_slots_columns)):
                df[time_slots_columns[i]] = df[time_slots_columns[i]].dt.time
            
            df['Share'] = df['Share'].round(6)

            full_data = df.sort_values(['Дата'], ascending = [True])

            full_data.reset_index(drop = True)

            dates_unique = full_data['Дата'].unique()

            res = []
            for date in dates_unique:
                t = full_data[full_data['Дата'] == date]
                # Создаем колонку для сортировки на основе времени начала
                t['sort_key'] = t['Время выхода'].apply(MediascopeParser.sort_time)

                # Сортируем по sort_key
                final = t.sort_values('sort_key').reset_index(drop = True)

                # Удаляем вспомогательную колонку
                final = final.drop('sort_key', axis = 1)
                final['Дата'] = pd.to_datetime(final['Дата'])

                res.append(final)
            
            result_data = pd.concat(res).reset_index(drop = True)

            result_data['Время выхода_dt'] = pd.to_datetime(result_data['Время выхода'], format = '%H:%M:%S', errors = 'coerce')
            result_data['Время окончания_dt'] = pd.to_datetime(result_data['Время окончания'], format = '%H:%M:%S', errors = 'coerce')

            # Автоматически корректируем переход через полночь
            result_data['Время окончания_dt'] = np.where(
                result_data['Время окончания_dt'] < result_data['Время выхода_dt'],
                result_data['Время окончания_dt'] + pd.Timedelta(days = 1),
                result_data['Время окончания_dt']
            )

            result_data['Продолжительность'] = (
                pd.to_datetime(result_data['Время окончания_dt']) - pd.to_datetime(result_data['Время выхода_dt'])
            ).dt.total_seconds()

            # Форматирование
            result_data['Продолжительность'] = result_data['Продолжительность'].apply(
                lambda x: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
            )

            result_data = result_data[
                [
                    'Канал', 'Дата', 'Название программы',
                    'Время выхода', 'Время окончания',
                    'Продолжительность', 
                    'Share', 'Жанр', 'День недели'
                ]
            ]

            return result_data
    

    def update_web_table(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
            Метод для обновления таблицы с сеткой Mediascope
        """
        new = pd.DataFrame()

        # Чтение данных из файла
        old_data = pd.read_excel(f'{self.web_filepath}')
        old_data['Дата'] = pd.to_datetime(old_data['Дата'])

        old_data['Время выхода'] = pd.to_datetime(old_data['Время выхода'], format = '%H:%M:%S', errors = 'coerce')
        old_data['Время окончания'] = pd.to_datetime(old_data['Время окончания'], format = '%H:%M:%S', errors = 'coerce')

        new_data['Время выхода'] = pd.to_datetime(new_data['Время выхода'], format = '%H:%M:%S', errors = 'coerce')
        new_data['Время окончания'] = pd.to_datetime(new_data['Время окончания'], format = '%H:%M:%S', errors = 'coerce')

        # Отбираем уникальные даты из старых и новых данных
        old_unique_dates = old_data['Дата'].unique()
        new_unique_dates = new_data['Дата'].unique()
        
        old_ones = []

        # Фильтруем даты, которые уже присутствуют в данных
        for new_date in new_unique_dates:
                
            if new_date in old_unique_dates:
                old_ones.append(pd.to_datetime(new_date))

        if len(old_ones) != 0:
            min_date_str = min(old_ones).strftime('%Y-%m-%d')

            # Оставляем только те даты, которые не встречаются в новых, если таковые нашлись
            filtered = old_data[old_data['Дата'] < min_date_str]

            if len(filtered) != 0:
        
                # Обновляем таблицу с фактическими данными
                new = pd.concat([filtered, new_data]).reset_index(drop = True)
        
        # В противном случае просто добавляем новые данные в конец старой таблицы
        else:
            new = pd.concat([old_data, new_data]).reset_index(drop = True)

        sorted_by_dates = new.sort_values('Дата').reset_index(drop = True)

        full = sorted_by_dates.sort_values(['Дата'], ascending = [True])

        full.reset_index(drop = True)

        dates_unique = full['Дата'].unique()

        res = []
        for date in dates_unique:
            t = full[full['Дата'] == date]
            # Создаем колонку для сортировки на основе времени начала
            t['sort_key'] = t['Время выхода'].apply(MediascopeParser.sort_time)

            # Сортируем по sort_key
            final = t.sort_values('sort_key').reset_index(drop = True)

            # Удаляем вспомогательную колонку
            final = final.drop('sort_key', axis = 1)
            final['Дата'] = pd.to_datetime(final['Дата'])

            res.append(final)

        self.web_df = pd.concat(res).reset_index(drop = True)
        # Если нужно вернуть в строковый формат
        self.web_df['Дата'] =  self.web_df['Дата'].dt.strftime('%Y-%m-%d')
        self.web_df['Время выхода'] =  self.web_df['Время выхода'].dt.strftime('%H:%M:%S')
        self.web_df['Время окончания'] =  self.web_df['Время окончания'].dt.strftime('%H:%M:%S')
        
        return  self.web_df
    
    

    def make_style_of_web_table(self, df: pd.DataFrame, sheet_name: str):
        """
            Функция для генерации внешнего вида таблицы с сеткой Mediascope.
        """
        # Заменяем NaN на None (xlsxwriter преобразует None в пустую ячейку)
        df_clean = df.where(pd.notna(df), None)

        column_configs = [
            {'header': 'Канал', 'width': 24.0, 'format': 'general'},
            {'header': 'Дата', 'width': 12.0, 'format': 'date'},
            {'header': 'Название программы', 'width': 95.0, 'format': 'general'},
            {'header': 'Время выхода', 'width': 14.0, 'format': 'general'},
            {'header': 'Время окончания', 'width': 14.0, 'format': 'general'},
            {'header': 'Продолжительность', 'width': 17.0, 'format': 'general'},
            {'header': 'Share', 'width': 11.0, 'format': 'general'},
            {'header': 'Жанр', 'width': 40.0, 'format': 'general'},
            {'header': 'День недели', 'width': 14.0, 'format': 'general'}
        ]
        
        self.make_style_of_table(
            df = df_clean,
            sheet_name = sheet_name,
            column_configs = column_configs,
            date_columns = ['Дата']
        )



class TVPreprocessing(BaseParser):
    """
        Класс для предобработки файлов с исторической и новыми сетками Федеральных ТВ-каналовс регулярной сеткой.
        Рассчитываются взвешенные доли программ.
    """
    def __init__(self, channel: str, filepath: str, plmrs: pd.DataFrame):
        """
            plmrs: pd.DataFrame: новая сетка Mediascope, которую нужно спарсить.
        """
        super().__init__(filepath)

        # Список допустимых названий каналов
        allowed_channels = [
            'ТНТ4', '2X2', 'КАРУСЕЛЬ', 'СУББОТА', 
            'СТСЛав', 'ЗВЕЗДА', 'МИР', 'МатчТВ', 
            'МузТВ', 'СОЛНЦЕ', 'СПАС', 'ТВЦ', 'ЧЕ', 'Ю'
            ]
        
        # Проверка наличия канала в списке допустимых
        if channel not in allowed_channels:
            raise ValueError(
                f"Канал '{channel}' не существует. Выберите канал из списка: {', '.join(allowed_channels)}"
            )
        
        self.channel = channel

        self.plmrs = plmrs
        self.filepath = filepath

        # Вызываем ensure_file_exists с нужными колонками
        self._ensure_file_exists([
            'Канал', 'Дата', 'Название программы', 'Время выхода', 
            'Время окончания', 'Продолжительность', 'Share', 'Share_weighted', 'Жанр', 'День недели'
        ])



    @staticmethod
    def convert_time(time_str: str):
        """
            Функция для конвертации времени из формата 25:00:00 в 01:00:00 или 5:00:00 в 05:00:00
            Args:
                time_str: время в формате строки
        """
        # Предполагаем стандартный формат HH:MM:SS или H:MM:SS
        if time_str[1] == ':':  # Формат H:MM:SS (одна цифра)
            hours = int(time_str[0])
            rest = time_str[1: ]  # :MM:SS
        else:  # Формат HH:MM:SS (две цифры)
            hours = int(time_str[: 2])
            rest = time_str[2: ]  # :MM:SS
        
        # Применяем преобразование часов
        if hours >= 24:
            hours = hours - 24
        # Форматируем с ведущим нулем
        return f'{hours:02d}{rest}'


    def parse_Palomars(self, start_time_col: str = 'Время выхода', end_time_col: str = 'Время окончания') -> pd.DataFrame:
        """
            Функция для парсинга файла с исторической сеткой Palomars.
            Args:
                filename: полный путь/название файла с исторической сеткой.
                column_1: столбец 1 с названием "Время выхода".
                column_2: столбец 2 с названием "Время окончания".
                program_time_slots: список из названий колонок, где присутствуют времена выхода и окончания программы.
            Returns:
                plmrs: причёсанный DataFrame с исторической сеткой.
        """
        #Чтение файла с данными
        df = self.plmrs.copy()
        #self.plmrs = pd.read_excel(self.filename)
    
        columns_with_time = [start_time_col, end_time_col]
        
        #Конвертация в формат даты столбцов со слотами
        def process_column(col):
            series = df[col].astype(str)
            converted = series.apply(TVPreprocessing.convert_time)
            return pd.to_datetime(converted, format = '%H:%M:%S', errors = 'coerce')

        # Обрабатываем колонки параллельно
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_column, columns_with_time))

        # Обновляем DataFrame
        for i, col in enumerate(columns_with_time):
            df[col] = results[i]

        #Вычисление длительности каждой программы. результат записывается в отдельный столбец
        df['Длительность, мин'] = np.abs(np.round((df[start_time_col] - df[end_time_col]) / np.timedelta64(1, 'm')))
        df['Длительность, мин'] = df['Длительность, мин'].astype(int)
    
        #В столбцах с временем выхода и окончания программы оставляем только время
        for i in range(len(columns_with_time)):
            df[columns_with_time[i]] = df[columns_with_time[i]].dt.time
        return df
    

    def _palomars_round_time(self, df) -> pd.DataFrame:
        """
            Функция для округления времени слотов программ в исторической сетке Palomars для какого-то конкретного дня
        """
        mars = df[['Канал', 'Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Продолжительность', 'Share', 'Жанр', 'День недели']]

        mars['Дата'] = pd.to_datetime(mars['Дата'])
        
        # Округляем время до минут
        share_calc = TVShareCalculator(self.channel, mars)
        mars['Время выхода_1min'] = share_calc.round_time('Время выхода')
        mars['Время окончания_1min'] = share_calc.round_time('Время окончания')
        

        mars_new = mars[
            [
                'Канал', 'Дата', 'Название программы', 
                'Share', 'Время выхода_1min', 'Время окончания_1min', 
                'Продолжительность', 'Жанр', 'День недели'
                ]
        ]
        

        mars_new.rename(columns = {'Время выхода_1min': 'Время выхода', 'Время окончания_1min': 'Время окончания'}, inplace = True)
        
        # Создаем копию оригинального столбца
        mars_new['Время выхода_новое'] = mars_new['Время выхода'].copy()
        
        # Заменяем значения начиная со второго
        for i in range(1, len(mars_new)):
            mars_new.loc[i, 'Время выхода_новое'] = mars_new.loc[i - 1, 'Время окончания']
        
        # Переименовываем колонки для наглядности
        mars_new.rename(columns = {'Время выхода': 'Время выхода_старое', 'Время выхода_новое': 'Время выхода'}, inplace = True)
        
        palomars = mars_new[
            [
                'Канал', 'Дата', 'Название программы', 
                'Share', 'Время выхода', 'Время окончания', 
                'Продолжительность', 'Жанр', 'День недели'
                ]
            ]

        self.palomars_adjusted = TVShareCalculator(self.channel, palomars).adjust_hour_start()
        
        # Эфирные сутки всегда начинаются с 05:00:00
        self.palomars_adjusted.loc[0, 'Время выхода'] = f'05:00:00'
        # Эфирные сутки всегда заканчиваются 04:59:59
        self.palomars_adjusted.loc[len(self.palomars_adjusted) - 1, 'Время окончания'] = f'04:59:59'
        return self.palomars_adjusted
    

    def process_daily_weighted_shares(
                        self, 
                        weighted_auedience: pd.DataFrame, 
                        start_time_col: str = 'Время выхода', 
                        end_time_col: str = 'Время окончания',
                        date_col: str = 'Дата'
                                ) -> Tuple[pd.DataFrame, Dict]:
        """
            Функция для расчета взвешенной доли. 

            Args:
                df: pd.DataFrame: Датафрейм, в котором есть столбцы Долей (Share), Время выхода, Время окончания, Название программы для какого одного дня.
                weighted_auedience: pd.DataFrame: ДатаФрейм с весами слотов, посчитанными через TotalTVAuedience для конкретного дня.
                start_time_col: название столбца с временем выхода программы. По умолчанию "Время выхода".
                end_time_col: название столбца с временем окончания программы. По умолчанию "Время окончания".
                date_col: название столбца с датой

            Returns:
                data: pd.DataFrame: Датафрейм с новой рассчитанной долей
        """
        df = self.parse_Palomars(start_time_col, end_time_col)

        if df.empty:
            raise ValueError('Данные с исторической сеткой из БД Mediascope отсутствуют или не были загружены!')
        
        # 2. Проверяем наличие обязательных колонок
        required_columns = [date_col, start_time_col, end_time_col, 'Share', 'Название программы']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Отсутствуют обязательные колонки: {missing_cols}')

        # Список для хранения конвертированных ДатаФреймов
        results_list = []

        # Словарь для хранения рассчитанных суммарных долей по дням
        shares = {}

        # Отбор уникальных дат для анализа
        dates_unique = df[date_col].unique()

        for date in dates_unique:

            try:

                df = self.plmrs[self.plmrs[date_col] == date].reset_index(drop = True)
                auedience = weighted_auedience[weighted_auedience[date_col] == date].reset_index(drop = True)

                plmrs_new = self._palomars_round_time(df)
                res, share = TVShareCalculator(self.channel, plmrs_new).calculate_weighted_share(auedience)
                
                results_list.append(res)
                shares[date] = share
            
            except Exception as e:
                print(f'Ошибка при обработке {date}: {str(e)}')
        
        # Объединение результатов
        if not results_list:
            print('Нет результатов для объединения')
            return pd.DataFrame(), {}
        
        combined_result = pd.concat(results_list).reset_index(drop = True)


        res = []
        for date in dates_unique:
            t = combined_result[combined_result[date_col] == date]

            t['sort_key'] = t[start_time_col].apply(BaseParser.get_sort_key)

            final = t.sort_values('sort_key').reset_index(drop = True)

            final = final.drop('sort_key', axis = 1)
            res.append(final)
        
        general_result = pd.concat(res).reset_index(drop = True)

        # Округление столбцов с долей
        general_result['Share'] = general_result['Share'].round(5)
        general_result['Share_weighted'] = general_result['Share_weighted'].round(8)

        general_result['Дата'] = general_result['Дата'].dt.strftime('%Y-%m-%d')

        general_result = general_result[
            [
                'Канал', 'Дата', 'Название программы',
                'Время выхода', 'Время окончания', 'Продолжительность',
                'Share', 'Share_weighted', 'Жанр', 'День недели'
                ]
        ]

        return general_result, shares
        

    def make_plmrs_style_of_table(self, df: pd.DataFrame, sheet_name: str):
        """
            Функция для генерации внешнего вида таблицы с сеткой Mediascope.
        """
        # Заменяем NaN на None (xlsxwriter преобразует None в пустую ячейку)
        df_clean = df.where(pd.notna(df), None)

        column_configs = [
            {'header': 'Канал', 'width': 24.0, 'format': 'general'},
            {'header': 'Дата', 'width': 12.0, 'format': 'date'},
            {'header': 'Название программы', 'width': 95.0, 'format': 'general'},
            {'header': 'Время выхода', 'width': 14.0, 'format': 'general'},
            {'header': 'Время окончания', 'width': 14.0, 'format': 'general'},
            {'header': 'Продолжительность', 'width': 17.0, 'format': 'general'},
            {'header': 'Share', 'width': 11.0, 'format': 'general'},
            {'header': 'Share_weighted', 'width': 16.0, 'format': 'general'},
            {'header': 'Жанр', 'width': 40.0, 'format': 'general'},
            {'header': 'День недели', 'width': 14.0, 'format': 'general'}
        ]
        
        self.make_style_of_table(
            df = df_clean,
            sheet_name = sheet_name,
            column_configs = column_configs,
            date_columns = ['Дата']
        )



class VIMBGridProcessor(BaseParser):
    """
        Класс для парсинга сеток VIMB (Сводная таблица)
    """
    
    def __init__(self, folder_path: str, channel_name: str):
        """
            Инициализация парсера VIMB.
            
            Args:
                folder_path: путь к файлам с новыми сетками ТВ-программ.
        """
        super().__init__(folder_path)
        self.folder_path = folder_path

        # Список допустимых названий каналов
        allowed_channels = [
            'ТНТ4', '2X2', 'КАРУСЕЛЬ', 'СУББОТА', 
            'СТСЛав', 'ЗВЕЗДА', 'МИР', 'МатчТВ', 
            'МузТВ', 'СОЛНЦЕ', 'СПАС', 'ТВЦ', 'ЧЕ', 'Ю'
            ]
        
        # Проверка наличия канала в списке допустимых
        if channel_name not in allowed_channels:
            raise ValueError(
                f"Канал '{channel_name}' не существует. Выберите канал из списка: {', '.join(allowed_channels)}"
            )
        
        self.channel_name = channel_name
    

    def parse_VIMB(self, filepath, sheet_name: str = 'ГРАФИК', skiprows = 1):
        """
            Метод для парсинга файла с сеткой VIMB из отчета Размещение -> Сводная таблица
            Args:
                sheet_name: имя листа, который будем считывать из файла. По умолчанию ГРАФИК.
                skiprows: количество строк, которые будем пропускать в файле. По умолчанию 1.
            Returns:
                VIMB: причёсанный DataFrame с сеткой VIMB.
        """
        # Чтение файла
        df = pd.read_excel(filepath, sheet_name = sheet_name, skiprows = skiprows)

        # Оставляем только нужные столбцы
        data = df[['Дата', 'Время выхода', 'Прод-ть', 'Название программы']]

        # Преобразование столбца в datetime
        data['Дата'] = pd.to_datetime(data['Дата'], format = '%d.%m.%Y')
        
        # Вычленяем день недели
        data['День недели'] = data['Дата'].dt.strftime('%A').str.capitalize()


        data['Время выхода_'] = pd.to_timedelta(data['Время выхода'].astype(str))
        data['Время выхода'] = data['Время выхода_'].apply(
            lambda x: f"{(x.days * 24 + x.seconds // 3600) % 24:02d}:{(x.seconds % 3600) // 60:02d}:{x.seconds % 60:02d}"
        )

        data['Прод-ть_'] = pd.to_timedelta(data['Прод-ть'].astype(str))
        data['Прод-ть'] = data['Прод-ть_'].apply(
            lambda x: f"{(x.days * 24 + x.seconds // 3600) % 24:02d}:{(x.seconds % 3600) // 60:02d}:{x.seconds % 60:02d}"
        )

        # Считаем время окончания
        data['Время окончания _'] = data['Время выхода_'] + data['Прод-ть_']

        # Если время окончания превышает 24 часа, корректируем отображение
        data['Время окончания'] = data['Время окончания _'].apply(
            lambda x: f"{(x.days * 24 + x.seconds // 3600) % 24:02d}:{(x.seconds % 3600) // 60:02d}:{x.seconds % 60:02d}"
        )

        # Оставляем только нужные столбцы
        VIMB = data[['Дата', 'Время выхода', 'Время окончания', 'Прод-ть', 'Название программы', 'День недели']]

        # Преобразуем столбец 'Дата' в datetime
        VIMB['Дата'] = pd.to_datetime(VIMB['Дата'])

        # Не на всех каналах эфирные сутки начинаются в 05:00:00. Поэтому нужна дополнительная конвертация на + 1 день
        # Создаем маску и увеличиваем дату 
        if self.channel_name in ['2X2', 'ТНТ4', 'МатчТВ', 'СТСЛав', 'СУББОТА', 'ЧЕ', 'ЗВЕЗДА', 'ТВЦ']:
            time_mask = (pd.to_timedelta(VIMB['Время выхода']) >= pd.Timedelta(hours = 5)) & (pd.to_timedelta(VIMB['Время выхода']) < pd.Timedelta(hours = 6))
            VIMB.loc[time_mask, 'Дата'] = VIMB.loc[time_mask, 'Дата'] + pd.Timedelta(days = 1)
        
        
        VIMB['День недели'] = VIMB['Дата'].dt.strftime('%A').str.capitalize()

        # Если нужно вернуть в строковый формат
        VIMB['Дата'] = VIMB['Дата'].dt.strftime('%Y-%m-%d')

        # Удаляем рекламные блоки и межпрограммные заставки
        mask = VIMB['Название программы'].str.contains('межпрограм', case = False, na = False) | \
               VIMB['Название программы'].str.contains('межпрограммный блок', case = False, na = False) | \
               VIMB['Название программы'].str.contains('межпрограммный', case = False, na = False) | \
               VIMB['Название программы'].str.contains('рекламный блок', case = False, na = False)
        VIMB = VIMB[~mask]

        # Убираем строки, которые содержат Р/Б. Применительно с детским каналам
        VIMB = VIMB[~VIMB['Название программы'].str.contains('р/б', case = False, na = False)]

        # Для канала Карусель удаляем программы "Новости", "Погода"
        if self.channel_name == 'КАРУСЕЛЬ':
            VIMB = VIMB[~VIMB['Название программы'].str.contains('погода', case = False, na = False)]
        
        # Для канала СТС Лав удаляем программы "это надо знать", "распаковка", "экодело"
        elif self.channel_name == 'СТСЛав':
            # список из программ, которые не нужны. Возможно, это реклама
            stop_words = ['это надо знать', 'распаковка', 'экодело', 'открывариум']
            pattern = '|'.join(stop_words)
            VIMB = VIMB[~VIMB['Название программы'].str.contains(pattern, case = False, na = False)]
        
        elif self.channel_name == 'ТВЦ':
            VIMB = VIMB[~VIMB['Название программы'].str.contains('погода', case = False, na = False)]

        return VIMB


    def parse_new_vimb_grids(
                self, 
                file_format: str = '*.xlsm', 
                date_column: str = 'Дата', 
                time_column: str = 'Время выхода'
            ) -> pd.DataFrame:
        """
            Метод для чтения новых сеток ТВ-программ из VIMB (Сводная таблица) для какого-то одного канала. (Применительно к историческим данным)
            
            Args:
                file_format: формат файлов с новыми сетками ТВ-программ. По умолчанию '*.xlsm'.
                date_column: название колонки с датой. По умолчанию 'Дата'.
                time_column: название колонки с временем выхода программы. По умолчанию 'Время выхода'.
                
            Returns:
                combined: pd.DataFrame: фулл-таблица с новыми сетками с сортировкой по дате и слоту от 05:00-29:00.
        """
        # Проверяем, что путь действительно существует
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f'Указанный путь {self.folder_path} не существует!')

        xlsx_files = glob.glob(os.path.join(self.folder_path, file_format))

        files = []
        # Перебираем найденные файлы и читаем их
        for file_path in xlsx_files:
            try:
                # Читаем файл в DataFrame
                vimb = self.parse_VIMB(file_path)
                files.append(vimb)
        
            except Exception as e:
                print(f'Ошибка при чтении файла {file_path}: {e}\n')

        # Полный датафрейм со всеми сетками (неотсортированный)
        full_vimb = pd.concat(files).reset_index(drop = True)


        # Устанавливаем правильные сортировки для столбцов с датой и временем начала программы
        full_vimb[date_column] = pd.to_datetime(full_vimb[date_column])
        sorted_vimb = full_vimb.sort_values(date_column).reset_index(drop = True)
        
        # Создаем столбец с Месяцем
        sorted_vimb['Месяц'] = sorted_vimb[date_column].dt.month
        sorted_vimb[date_column] = sorted_vimb[date_column].dt.strftime('%Y-%m-%d')
        
        months_unique = sorted_vimb['Месяц'].unique()
        
        result = {}
        for month in months_unique:
            df = sorted_vimb[sorted_vimb['Месяц'] == month].reset_index(drop = True)
            data = df.drop('Месяц', axis = 1)
            
            dates_unique = data[date_column].unique()
            res = []
            for date in dates_unique:
                t = data[data[date_column] == date]
        
                t['sort_key'] = t[time_column].apply(BaseParser.get_sort_key)
        
                final = t.sort_values('sort_key').reset_index(drop = True)
        
                final = final.drop('sort_key', axis = 1)
                res.append(final)
            
            general_result = pd.concat(res).reset_index(drop = True)
        
            result[month] = general_result

        result_df = pd.concat(result.values(), ignore_index = True)

        vimb = result_df.copy()

        # ВОТ ИСПРАВЛЕНИЕ - правильная обработка времени
        vimb['datetime_obj'] = pd.to_datetime(vimb['Прод-ть'], format = '%H:%M:%S')
        vimb['hour_start'] = pd.to_datetime(vimb['Время выхода']).dt.hour
        vimb['hour_end'] = pd.to_datetime(vimb['Время окончания']).dt.hour
        vimb['duration'] = vimb['datetime_obj'].dt.hour * 60 + vimb['datetime_obj'].dt.minute + vimb['datetime_obj'].dt.second / 60
        vimb.drop('datetime_obj', axis = 1, inplace = True)
        
        vimb['original_index'] = vimb.index
        vimb['original_index'] = vimb['original_index'].round().astype(int)

        # ПРАВИЛЬНАЯ ФИЛЬТРАЦИЯ - преобразуем время в datetime для сравнения
        # Создаем временные колонки для сравнения
        vimb['time_start_dt'] = pd.to_datetime(vimb['Время выхода'], format = '%H:%M:%S')
        vimb['time_end_dt'] = pd.to_datetime(vimb['Время окончания'], format = '%H:%M:%S')
        
        # Исправляем время окончания для программ, переходящих через полночь
        # Если время окончания меньше времени начала, значит программа переходит через полночь
        mask_overnight = vimb['time_end_dt'] < vimb['time_start_dt']
        vimb.loc[mask_overnight, 'time_end_dt'] += pd.Timedelta(days = 1)
        
        # Теперь корректно фильтруем программы, пересекающие 5:00
        split_time = pd.to_datetime('05:00:00', format = '%H:%M:%S')

        mask_crosses_5am = (
        # Случай 1: начинается до 05:00, заканчивается после 05:00 (включая переход через полночь)
        (vimb['time_start_dt'] < vimb['time_end_dt']) &  # обычный случай (без перехода через полночь)
        (vimb['time_start_dt'] < split_time) & 
        (vimb['time_end_dt'] > split_time)
        ) | (
        # Случай 2: переходит через полночь (start > end без коррекции)
        # Но time_end_dt уже скорректирован +1 день
        # Так что time_end_dt всегда > time_start_dt после коррекции
        # Поэтому этот случай уже покрыт Случаем 1
        (vimb['time_start_dt'] >= split_time) & 
        (vimb['time_end_dt'] > split_time + pd.Timedelta(days=1))
        )
        
        df = vimb[mask_crosses_5am].reset_index(drop = True)

        # Удаляем временные колонки
        vimb = vimb.drop(['time_start_dt', 'time_end_dt'], axis = 1)
        
        df = df[
            [
                'Дата', 'Время выхода', 'Время окончания', 
                'Прод-ть', 'Название программы', 'День недели', 'original_index'
             ]
        ]
    
        weekdays = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']

        new_rows = []

        end_time_new = '04:59:59'
        start_time_part_2 = '05:00:00'

        for i in range(len(df)):
            current_date = df.iloc[i]['Дата']
            time_start = df.iloc[i]['Время выхода']
            time_end = df.iloc[i]['Время окончания']
            pr_name = df.iloc[i]['Название программы']
            current_weekday = df.iloc[i]['День недели']
            idx_orig = df.iloc[i]['original_index']

            # Часть 1: До 05:00:00
            row_1 = {
                'Дата': current_date,
                'Время выхода': time_start,
                'Время окончания': '04:59:59',
                'Прод-ть': VIMBGridProcessor.calculate_duration(time_start, '04:59:59'),
                'Название программы': pr_name,
                'День недели': current_weekday,
                'original_index': idx_orig
            }
            new_rows.append(row_1)

            
            # Часть 2: После 05:00:00    
            next_date = pd.to_datetime(current_date, format = '%Y-%m-%d', errors = 'coerce') + pd.Timedelta(days = 1)

            row_2 = {
                'Дата': next_date.strftime('%Y-%m-%d'),
                'Время выхода': start_time_part_2,
                'Время окончания': time_end,
                'Прод-ть': VIMBGridProcessor.calculate_duration(start_time_part_2, time_end),
                'Название программы': pr_name,
                'День недели': weekdays[next_date.weekday()],
                'original_index': idx_orig
            }
            new_rows.append(row_2)

        df_new = pd.DataFrame(new_rows)

        vimb = vimb[['Дата', 'Время выхода', 'Время окончания', 'Прод-ть', 'Название программы', 'День недели', 'original_index']]

        indices_to_remove = df['original_index'].unique()

        vimb_cleaned = vimb[~vimb['original_index'].isin(indices_to_remove)].copy()

        vimb_new = pd.concat([vimb_cleaned, df_new], ignore_index = True)

        vimb_new['Дата'] = pd.to_datetime(vimb_new['Дата'])
        vimb_new = vimb_new.sort_values('Дата').reset_index(drop = True)


        vimb_new = vimb_new[['Дата', 'Время выхода', 'Время окончания', 'Прод-ть', 'Название программы', 'День недели']]

        # Сортируем по дате и времени
        res = []
        dates_unique = vimb_new['Дата'].unique()
        for date in dates_unique:
            t = vimb_new[vimb_new['Дата'] == date]

            t['sort_key'] = t['Время выхода'].apply(BaseParser.get_sort_key)

            final = t.sort_values('sort_key').reset_index(drop = True)

            final = final.drop('sort_key', axis = 1)
            res.append(final)

        general_result = pd.concat(res).reset_index(drop = True)

        general_result['Дата'] = pd.to_datetime(general_result['Дата'], errors='coerce')
            
        # Затем преобразуем в строку
        general_result['Дата'] = general_result['Дата'].dt.strftime('%Y-%m-%d')

        general_result = self.adjust_end_time(general_result)

        # Убедимся, что дата в строковом формате
        general_result['Дата'] = general_result['Дата'].astype(str)

        general_result = general_result[general_result['Прод-ть'] != '00:00:01'].reset_index(drop = True)

        # Проверяем разрывы
        warnings = self.check_program_gaps(general_result)
        # Выводим результаты
        if warnings:
            print(Color.RED + f'Найдены разрывы более 30 мин для канала {self.channel_name}' + Color.END)
            for i, warning in enumerate(warnings, 1):
                print(f"\n{i}. Для {warning['дата']} предупреждение: {warning['предупреждение']}")
                print("-" * 80)

        return general_result
    

    @staticmethod
    def calculate_duration(start_time, end_time):
        """
            Вычисляет продолжительность программы, учитывая переход через полночь.
            Учитывает часы, минуты и секунды.
            
            Args:
                start_time: время начала в формате 'HH:MM:SS'
                end_time: время окончания в формате 'HH:MM:SS'
            
            Returns:
                Продолжительность в формате 'HH:MM:SS'
        """
        # Разбиваем время на часы, минуты и секунды
        start_h, start_m, start_s = map(int, start_time.split(':'))
        end_h, end_m, end_s = map(int, end_time.split(':'))
        
        # Преобразуем в секунды от полуночи
        start_total_sec = start_h * 3600 + start_m * 60 + start_s
        end_total_sec = end_h * 3600 + end_m * 60 + end_s
        
        # Если время окончания меньше времени начала - переход через полночь
        if end_total_sec < start_total_sec:
            # Продолжительность = (24:00:00 - начало) + окончание
            duration_sec = (24 * 3600 - start_total_sec) + end_total_sec
        else:
            # Обычный случай
            duration_sec = end_total_sec - start_total_sec
        
        # Преобразуем обратно в часы:минуты:секунды
        duration_h = duration_sec // 3600
        duration_m = (duration_sec % 3600) // 60
        duration_s = duration_sec % 60
        
        return f"{duration_h:02d}:{duration_m:02d}:{duration_s:02d}"
    

    def check_program_gaps(self, df: pd.DataFrame):
        """
            Проверяет разрывы между программами в сетке телепрограмм.
            
            Параметры:
            ----------
                df: pd.DataFrame
            
            Returns:
            -------
                warnings: list
                Список предупреждений о разрывах более 30 минут
        """
        warnings = []

        df_copy = df.copy()
        df_copy['Время выхода'] = pd.to_datetime(df_copy['Время выхода'], format='%H:%M:%S')
        df_copy['Время окончания'] = pd.to_datetime(df_copy['Время окончания'], format='%H:%M:%S')
        
        # Группируем по датам
        grouped = df_copy.groupby('Дата')
        
        for date_key, group in grouped:
            # Сортируем по времени выхода
            group_sorted = group.sort_values('Время выхода').reset_index(drop=True)
            
            for i in range(len(group_sorted) - 1):
                current_end = group_sorted.loc[i, 'Время окончания']
                next_start = group_sorted.loc[i + 1, 'Время выхода']
                current_program = group_sorted.loc[i, 'Название программы']
                next_program = group_sorted.loc[i + 1, 'Название программы']
                
                # Вычисляем разрыв
                gap = next_start - current_end
                
                # Если разрыв больше 30 минут
                if gap > timedelta(minutes=30):
                    # Если date_key - это уже datetime объект
                    if hasattr(date_key, 'strftime'):
                        date_str = date_key.strftime('%d.%m.%Y')
                    else:
                        # Если date_key - строка или другой тип, преобразуем
                        date_str = pd.to_datetime(date_key).strftime('%d.%m.%Y')
                    
                    warnings.append({
                        'дата': date_key,
                        'предыдущая_программа': current_program,
                        'следующая_программа': next_program,
                        'время_окончания': current_end,
                        'время_начала': next_start,
                        'разрыв_минут': gap.total_seconds() / 60,
                        'предупреждение': f"{date_str} наблюдается разрыв между программами '{current_program}' "
                                        f"(окончание в {current_end.strftime('%H:%M')}) и '{next_program}' (начало в {next_start.strftime('%H:%M')}) "
                                        f"продолжительностью {int(gap.total_seconds() / 60)} минут"
                    })
        
        return warnings



    def adjust_end_time(
                    self, 
                    df: pd.DataFrame, 
                    time_col: str = 'Время окончания'
                ) -> pd.DataFrame:
        """
            Корректировка времени окончания для обработки границ часов. Если время окончания, например, 05:00:00, то будет сделана замена на 04:59:59.
            Отдельно обрабатывается перескок через полночь.

            Args:
                df: датафрейм, в котором хотим произвести конвертацию времени.
                time_col: str: название колонки, в которой хотим сделать конвертацию. По умолчанию 'Время окончания'.

            Returns:
                Датафрейм df с конвертированными слотами Времени окончания программ.
        """

        def adjust_time(time_str: str) -> str:
            h, m, s = map(int, time_str.split(':'))

            if m == 0 and s == 0:

                # Если полночь
                if h == 0:
                    return '23:59:59'

                return f'{(h - 1):02d}:59:59'

            return time_str

        df_ = df.copy()
        df_[time_col] = df_[time_col].apply(adjust_time)
        return df_
    

    def check_start__and__end_day(self, df: pd.DataFrame, date_column: str = 'Дата'):
        """
            Метод для проверки, что каждый день начинается и заканчивается в 05:00:00. Эфирные сутки 05:00:00-29:00:00 (05:00:00 следующего дня).

            Args:
                df: pd.DataFrame: датафрейм, который хотим проверить.
                date_column: str: название столбца с датой. Даты в формате строки
            Returns:
        """
        need_start = '05:00:00'
        need_end = '04:59:59'

        need_start_dt = pd.to_datetime(need_start)
        need_end_dt = pd.to_datetime(need_end)

        df_check = df.copy()

        # Конвертация дат в строковый формат, если требуется
        try:
            # Если даты в datetime
            if pd.api.types.is_datetime64_any_dtype(df_check[date_column]):
                df_check[date_column] = df_check[date_column].dt.strftime('%Y-%m-%d')

            # Если даты не в строковом формате
            elif not all(isinstance(x, str) for x in df_check[date_column].dropna().head(10)):
                df_check[date_column] = pd.to_datetime(df_check[date_column], errors = 'coerce').dt.strftime('%Y-%m-%d')

        except Exception as e:
            print(f'Ошибка конвертации дат: {e}')

        # Проверка хронологии дат
        # Группируем даты по годам и месяцам
        month_days = defaultdict(set)
        
        for date_str in df_check[date_column].dropna().unique():
            try:
                year, month, day = map(int, date_str.split('-'))
                month_days[(year, month)].add(day)
            except (ValueError, AttributeError):
                continue
        
        # Проверяем каждый месяц
        for (year, month), days_set in month_days.items():
            # Получаем правильное количество дней для этого месяца
            if month == 2:  # Февраль - проверяем високосность
                _, correct_days = calendar.monthrange(year, month)
            else:
                correct_days = calendar.monthrange(year, month)[1]
            
            # Проверяем, есть ли все дни месяца
            actual_days = sorted(days_set)
            expected_days = set(range(1, correct_days + 1))
            
            missing_days = expected_days - days_set
            extra_days = days_set - expected_days
            
            if missing_days:
                print(f'❌ ОШИБКА: В {year}-{month:02d} отсутствуют дни: {sorted(missing_days)}')
                print(f'   Должно быть дней: {correct_days}, имеется: {len(days_set)}')
            
            if extra_days:
                print(f'❌ ОШИБКА: В {year}-{month:02d} найдены лишние дни: {sorted(extra_days)}')
            
            # Проверяем непрерывность дней
            if actual_days and len(actual_days) != actual_days[-1] - actual_days[0] + 1:
                print(f'⚠️ ПРЕДУПРЕЖДЕНИЕ: В {year}-{month:02d} дни идут не подряд')
                print(f'Присутствуют дни: {actual_days}')
        
        #print(f"\nПроверка месяцев завершена. Всего уникальных месяцев: {len(month_days)}")


        corrections_made = False
        # Проверка, что каждый день начинается в 05:00:00 и заканчивается в 05:00:00
        dates_unique = df_check[date_column].unique()

        # Создаем маску для удаления строк с профилактикой
        delete_mask = pd.Series(False, index = df.index)

        for date in dates_unique:

            # Находим индексы в исходном df для текущей даты
            date_mask = df_check[date_column] == date
            date_indices = df[date_mask].index.tolist()

            table_check = df_check[df_check['Дата'] == date].reset_index(drop = True)


            # Выделение дней с профилактикой
            mask = table_check['Название программы'].str.contains('профилактика', case = False, na = False) | \
                   table_check['Название программы'].str.contains('профилакт', case = False, na = False)
            prophylactic_count = mask.sum()

            indices = table_check.index.tolist()
            first_idx_check = indices[0]
            last_idx_check = indices[-1]

            # Находим соответствующие индексы в исходном df
            first_idx_original = date_indices[first_idx_check]
            last_idx_original = date_indices[last_idx_check]

            start = table_check[table_check['Время выхода'] == '05:00:00']
            stop = table_check[table_check['Время окончания'] == '04:59:59']

            # Исходное время выхода
            old_start = table_check.at[first_idx_check, 'Время выхода']
            old_start_dt = pd.to_datetime(old_start)
            # Исходное время окончания
            old_end = table_check.at[last_idx_check, 'Время окончания']
            old_end_dt = pd.to_datetime(old_end)

            # Флаг, указывающий, что были изменения времени
            time_changed = False

            # Если дней с профилактикой не обнаружено
            if prophylactic_count == 0:
                if len(start) == 0 and len(table_check) != 1 :
                    print(f'⚠️ Для {date} не найдена стартовая программа дня.')

                    # Если разница между фактической датой старта и нужной больше 45 мин, то замена не производится
                    if abs(old_start_dt - need_start_dt).total_seconds() / 60.0 < 45:
                        print(f'Делаем замену с {old_start} на 05:00:00.')
                        df.loc[first_idx_original, 'Время выхода'] = '05:00:00'
                        corrections_made = True
                        time_changed = True

                elif len(stop) == 0  and len(table_check) != 1:
                    print(f'⚠️ Для {date} не найдена кульминационная программа дня')

                    # Если разница между фактической датой окончания и нужной больше 45 мин, то замена не производится
                    if (abs(old_end_dt - need_end_dt).total_seconds()) / 60.0 < 45:
                        print(f'Делаем замену с {old_end} на 04:59:59.')
                        df.loc[last_idx_original, 'Время окончания'] = '04:59:59'
                        corrections_made = True
                        time_changed = True
            
            # Если обнаружены дни с профилактикой
            else:
                # Помечаем строки с профилактикой для удаления
                prophylactic_indices = [date_indices[i] for i in table_check[mask].index]
                delete_mask.loc[prophylactic_indices] = True
                corrections_made = True

            
            # Если время было изменено, пересчитываем длительность для всех программ этого дня
            if time_changed:
                print(f"Пересчитываем длительность программ для даты {date}...")
                
                # Получаем все индексы для текущей даты
                day_indices = df[df[date_column] == date].index
                
                # Для каждой программы в этом дне пересчитываем длительность
                for idx in day_indices:
                    start_time = df.loc[idx, 'Время выхода']
                    end_time = df.loc[idx, 'Время окончания']
                    
                    # Вычисляем новую длительность
                    new_duration = VIMBGridProcessor.calculate_duration(start_time, end_time)
                    
                    if new_duration:
                        df.loc[idx, 'Прод-ть'] = new_duration
                        if idx == first_idx_original or idx == last_idx_original:
                            print(f"  Программа '{df.loc[idx, 'Название программы']}': новая длительность {new_duration}")
                    else:
                        print(f"  Ошибка при вычислении длительности для программы '{df.loc[idx, 'Название программы']}'")

        # Удаляем строки с профилактикой из исходного df
        if delete_mask.any():
            print(f'Найдены строки с ПРОФИЛАКТИКОЙ. Удалено строк с профилактикой: {delete_mask.sum()}')
            df.drop(df[delete_mask].index, inplace = True)
        
        if corrections_made:
            print('Изменения внесены в исходную таблицу.')
        else:
            print('Изменений не требуется.')

        return df

    

    def update_vimb_file(self, web_new):
        """
            Функция для обновления файла с сетками ТВ-программ VIMB.
        """
        if len(web_new) == 0:
            print('Ошибка! Вы пытаетесь сохранить пустой DataFrame!')

        # Проверяем существование файла
        file_path = Path(self.folder_path)
        
        if not file_path.exists():
            print(f'Файл {file_path} не найден. Создаем новый файл...')
            
            # Подготавливаем данные для записи
            new_cleaned = web_new.copy()
            
            # Приводим все к строковому типу и обрезаем пробелы
            for col in new_cleaned.columns:
                new_cleaned[col] = new_cleaned[col].astype(str).str.strip()
            
            # Проверяем границы дней перед сохранением
            new_cleaned = self.check_start__and__end_day(new_cleaned)
            
            # Создаем Excel файл с форматированием
            self.folder_path = file_path # Добавляем путь для сохранения

            self.make_vimbs_style_of_table(
                df = new_cleaned, 
                sheet_name = 'Sheet1'
            )
            
            print(f'Создан новый файл: {file_path}')
            return
        
        # Файл существует - читаем и обновляем
        try:
            new = web_new.copy()
            # Читаем существующие данные
            old_web = pd.read_excel(self.folder_path)

            # Приводим даты к единому формату
            for col in new.columns:
                new[col] = new[col].astype(str).str.strip()
                old_web[col] = old_web[col].astype(str).str.strip()
            
            full = pd.concat([old_web, new]).reset_index(drop = True)

            df_no_duplicates = full.drop_duplicates(
                subset = ['Дата', 'Время выхода', 'Время окончания', 'Прод-ть', 'Название программы', 'День недели'],
                keep = 'last'
            )
            print(f'Удалено {len(full) - len(df_no_duplicates)} дубликатов.')

            # Проверяем границы дней
            self.check_start__and__end_day(df_no_duplicates)

            self.make_vimbs_style_of_table(
                df = df_no_duplicates, 
                sheet_name = 'Sheet1'
            )

        except Exception as e:
            print(f'Ошибка при обновлении файла: {e}')
        

            # Создаем резервную копию и новый файл
            try:
                backup_path = file_path.with_suffix('-копия.xlsx')
                if file_path.exists():
                    shutil.copy2(file_path, backup_path)
                    print(f'Создана резервная копия: {backup_path}')
                
                # Создаем новый файл с web_new данными
                self.folder_path = file_path

                self.make_vimbs_style_of_table(
                    df = web_new, 
                    sheet_name = 'Sheet1'
                )
                print(f'Создан новый файл с предоставленными данными.')
                
            except Exception as backup_error:
                print(f'Критическая ошибка при создании резервной копии: {backup_error}')
    


    def make_vimbs_style_of_table(self, df: pd.DataFrame, sheet_name: str):
        """
            Функция для генерации внешнего вида таблицы с сеткой ВИМБ.
        """
        column_configs = [
            {'header': 'Дата', 'width': 14.0, 'format': 'date'},
            {'header': 'Время выхода', 'width': 14.0, 'format': 'general'},
            {'header': 'Время окончания', 'width': 14.2, 'format': 'general'},
            {'header': 'Прод-ть', 'width': 14.0, 'format': 'general'},
            {'header': 'Название программы', 'width': 72.0, 'format': 'general'},
            {'header': 'День недели', 'width': 12.0, 'format': 'general'}
        ]
        
        self.make_style_of_table(
            df = df,
            sheet_name = sheet_name,
            column_configs = column_configs,
            date_columns = ['Дата']
        )


class ProgramMatcher(BaseParser):
    """
        Класс для сопоставления телепрограмм из разных источников: Mediascope и VIMB.
        Обеспечивает нормализацию названий программ и поиск временных совпадений.
    """
    def __init__(self, channel: str, channel_vocabulary: pd.DataFrame, folder_path: str, palomars_grid: pd.DataFrame, vimb_grid: pd.DataFrame):
        """
        Инициализация ProgramMatcher
        
        Args:
            channel_vocabulary: pd.DataFrame: датафрейм со справочником программ
            folder_path: str: путь к файлу с данными.
            palomars_grid:  str:историческая сетка Mediascope.
            vimb_grid: str: историческая сетка VIMB.
        """
        super().__init__(folder_path)

        self.channel = channel
        self.channel_vocabulary = channel_vocabulary
        self.folder_path = folder_path
        self.palomars_grid = palomars_grid
        self.vimb_grid = vimb_grid


    @staticmethod
    def find_common_base_names(names: List[str])  -> Dict[str, str]:
        """
            Находит базовые названия программ, заменяя длинные варианты на короткие.

            Args:
                names: Список названий программ
                
            Returns:
                Словарь маппинга {длинное_название: базовое_название}
        """
        # Уникальные названия
        unique_names = sorted(set(names), key = len)
        mapping = {}
        
        # Сначала создаем маппинг для каждого названия на себя
        for name in unique_names:
            mapping[name] = name
        
        # Ищем подстроки
        for i, short_name in enumerate(unique_names):
            for long_name in unique_names[i + 1:]:
                # Если короткое название является подстрокой длинного
                if short_name in long_name:
                    mapping[long_name] = short_name
        
        return mapping
    

    @staticmethod
    def check_cartoons(
            df: pd.DataFrame, 
            target_name_cartoons: list, 
            full_list:list, 
            replacement_name: str,
            debug = False
        ):
        """
            Если в столбце "Название программы" встречается название "маша и медведь", а также любое из списка other_list,
            то производится замена названий на "мультфильм о маше".
            В противном случае, если присутствует только "маша и медведь", то замена не производится.
        """
        unique_values = set(df['Название программы'].unique())
        
        has_cartoon = any(cartoon in unique_values for cartoon in target_name_cartoons)
        
        if has_cartoon:
            if debug:
                print(Color.BROWN + 'Встретились мультфильмы о Маше или Коте Леопольде. Делаю замену на ' + \
                Color.BOLD + f'"{replacement_name}".' + Color.END)
            # Заменяем всё, включая "маша и медведь"
            mask = df['Название программы'].str.lower().isin(full_list)
            df.loc[mask, 'Название программы'] = replacement_name
        return df


    def match_vimb_with_palomars_grids(self, cities_path: str, minutes: int = 10):
        """
            Смэтчивает сетки VIMB и Palomars между собой.
            Args:
                cities_path: str: словарь из городов, где ключ - страна, значение - список городов, присущих этой стране.
                minutes: int: количество минут, до которых округляем столбцы "Время начала", "Время окончания" программы. По дефолту равно 10.
            Returns:
        """

        vimb_full = self.vimb_grid.copy()
        plmrs = self.palomars_grid.copy()

        # Отбираем уникальные даты в сетке VIMB
        dates_unique = vimb_full['Дата'].unique()

        result_webs = {}
        not_matched_programs = {}   # список программ, которые встретились в VIMB, но не встретились в Palomars

        for target_date in dates_unique:

            # Отбор конкретной даты в ВИМБ
            vimb = vimb_full[vimb_full['Дата'] == target_date].reset_index(drop = True)

            # Отбираем дату, которую будем анализировать
            palomars = plmrs[plmrs['Дата'] == target_date].reset_index(drop = True)

            vimb_prgms = []
            vimb_df = pd.DataFrame()

            palomars_prgms = []
            palomars_df = pd.DataFrame()
            ######################## НОВЫЙ КУСОК ########################
            if self.channel in [
                'СОЛНЦЕ', 'КАРУСЕЛЬ', 'СУББОТА', 'СТСЛав', 
                '2X2', 'ТНТ4', 'ЧЕ', 'МИР', 'Ю', 'ЗВЕЗДА', 
                'ТВЦ', 'СПАС']:

                # 1. Программы в VIMB
                cleaner = GeneralTextCleaner(self.channel)
                vimb_prgms, vimb_df = cleaner.clean_dataframe(vimb)

                # 2. Программы в PALOMARS
                palomars_prgms, palomars_df = cleaner.clean_dataframe(palomars)
            
            elif self.channel == 'МатчТВ':
                
                # Загрузка справочника с городами
                cities_loaded = Dict_Operations.load_pkl_file(cities_path)
                
                # 1. Программы в VIMB
                sport_cleaner = SportChannelCleaner(self.channel)
                vimb_prgms, vimb_df = sport_cleaner.clean_dataframe(vimb, 'vimb', cities_loaded)

                # 2. Программы в PALOMARS
                palomars_prgms, palomars_df = sport_cleaner.clean_dataframe(palomars, 'palomars', cities_loaded)
            
            elif self.channel == 'МузТВ':
                
                # 1. Программы в VIMB
                music_cleaner = MusicChannelCleaner(self.channel)
                vimb_prgms, vimb_df = music_cleaner.clean_dataframe(vimb)

                # 2. Программы в PALOMARS
                palomars_prgms, palomars_df = music_cleaner.clean_dataframe(palomars)

            ######################## КОНЕЦ НОВОГО КУСКА ########################
            
            # Делаем поиск по схожим программам
            similar = CosineSimilarity(palomars_prgms, vimb_prgms, palomars_df, vimb_df)
            result, not_matched, comparison = similar.comparison(self.channel_vocabulary, min_similarity = 0.5, use_vocabulary = True)

            # Добавляем ненайденные программы в список с ненайденными
            if len(not_matched) != 0:
                not_matched_programs[target_date] = not_matched
            
            # Заменяем названия передач, если какие-то не совпадают
            df = result[result['similarity'].round(5) != 1.00000]
            
            programs_replace = {}
            for i in range(len(df)):
                programs_replace[df.iloc[i]['Программа Palomars']] = df.iloc[i]['Программа VIMB']
                
            palomars_df['program_name'].replace(programs_replace, inplace = True)
            
            # Находим базовые названия программ. Производим замену
            base_names = ProgramMatcher.find_common_base_names(palomars_df['program_name'].tolist())

            palomars_df['Базовое_название'] = palomars_df['program_name'].map(base_names)    
            
            # Оставляем только нужные столбцы для анализа
            Pal = palomars_df[['Дата', 'Название программы', 'Базовое_название', 'Время выхода', 'Время окончания', 'Share_weighted', 'Жанр']]
            
            Pal.rename(columns = 
                    {
                        'Название программы': 'Название программы palomars',
                        'Базовое_название': 'Название программы', 
                        'Share_weighted': 'Share'
                    }, 
                    inplace = True)
            Pal['Название программы'] = Pal['Название программы'].str.lower()
            
            VIMB = vimb_df[['Дата', 'Название программы', 'program_name', 'Время выхода', 'Время окончания']]
            VIMB.rename(columns = {
                'Название программы': 'Название программы vimb',
                'program_name': 'Название программы'
                }, inplace = True)
            
            # =============== Замена названий мультфильмов, связанных с Машей и котом Леопольдом ===============
            dataframes = [VIMB, Pal]
            for df in dataframes:
                df = ProgramMatcher.check_cartoons(
                                        df = df,
                                        target_name_cartoons = ['маша и медведь'],
                                        full_list = ['машины сказки', 'машины песенки', 'машины страшилки', 'маша и медведь', 'машкины страшилки'],
                                        replacement_name = 'мультфильм о маше'
                                        )
                df['Название программы'] = np.where(
                        df['Название программы'].str.contains('леопольд', case = False, na = False), 
                        'мультфильм о коте леопольде', df['Название программы']
                    )
                
            # =========================================================================================================================
            VIMB_init = VIMB.copy()
            Pal_init = Pal.copy()

            # ======== НОВЫЙ КУСОК. ПРИНУДИТЕЛЬНАЯ ЗАМЕНА НА "СЕРИЯ МУЛЬТФИЛЬМОВ" В PALOMARS, ИСПОЛЬЗУЯ ИНФОРМАЦИЮ ИЗ VIMB. ========
            # Будем делать манипуляции, описанные ниже только в том случае, если в столбце "Название программы" таблицы VIMB фигурирует "серия мультфильмов"
            if 'серия мультфильмов' in VIMB_init['Название программы'].unique():
                print(Color.VIOLET + f'Делаю предобработку "серии мультфильмов" для канала {self.channel}' + Color.END)

                pr = TVScheduleProcessor(self.channel, VIMB_init, Pal_init)
                # Схлопываем программы VIMB, чтобы более наглядно увидеть, где именно была "серия мультфильмов"
                vimb_joined = pr.join_broadcasts(VIMB_init, 'vimb', include_share = False)

                # Отбираем только те слоты, в которых фигурирует название 'серия мультфильмов'
                cartoons_series = vimb_joined[vimb_joined['Название программы'] == 'серия мультфильмов'].reset_index(drop = True)

                # Переводим время в Palomars в datetime для удобной фильтрации
                Pal_init['time_start_dt'] = pd.to_datetime(Pal_init['Время выхода'], format = '%H:%M:%S')
                Pal_init['time_end_dt'] = pd.to_datetime(Pal_init['Время окончания'], format = '%H:%M:%S')

                # Коррекция перехода через полночь
                mask_night = Pal_init['time_end_dt'] < Pal_init['time_start_dt']
                Pal_init.loc[mask_night, 'time_end_dt'] = Pal_init.loc[mask_night, 'time_end_dt'] + timedelta(days = 1)

                # Создаем столбец с флагом. Если во встретившемся слоте Palomars в таблице VIMB в это время была "серия мультфильмов", то
                # мы в новый столбец записываем "серия мультфильмов"
                Pal_init['cartoon_series_flag'] = ''

                for i in range(len(cartoons_series)):
                    # Берем конкретную строку
                    row = cartoons_series.iloc[i]
                    
                    # Преобразуем время
                    start_time = pd.to_datetime(row['Время выхода'], format='%H:%M:%S')
                    end_time = pd.to_datetime(row['Время окончания'], format='%H:%M:%S')
                    
                    # Обработка перехода через полночь для целевого интервала
                    if end_time < start_time:
                        end_time = end_time + timedelta(days = 1)
                    
                    # Интервал с запасом.
                    # Не всегда "Время начала" в Palomars совпадает с "Время начала" в VIMB. Даём небольшой люфт.
                    start_threshold = start_time - timedelta(minutes = 5)
                    end_threshold = end_time + timedelta(minutes = 5)

                    # Создаем копии для корректировки перехода через полночь в Pal_init
                    Pal_init['time_start_temp'] = Pal_init['time_start_dt']
                    Pal_init['time_end_temp'] = Pal_init['time_end_dt']
                    
                    # Корректируем время окончания для программ, идущих через полночь
                    night_mask = Pal_init['time_end_dt'] < Pal_init['time_start_dt']
                    Pal_init.loc[night_mask, 'time_end_temp'] = Pal_init.loc[night_mask, 'time_end_dt'] + timedelta(days = 1)

                    ################################################################################
                    # Отбираем строки в интервале
                    mask = (Pal_init['time_start_temp'] >= start_threshold) & \
                        (Pal_init['time_end_temp'] <= end_threshold)
                    result_indices = Pal_init[mask].index
                    
                    for idx in result_indices:
                        pr_name = Pal_init.loc[idx, 'Название программы']
                        start_time_pr = Pal_init.loc[idx, 'time_start_dt']
                        end_time_pr = Pal_init.loc[idx, 'time_end_dt']
                        
                        # Корректировка для проверяемой программы
                        if end_time_pr < start_time_pr:
                            end_time_pr_check = end_time_pr + timedelta(days=1)
                        else:
                            end_time_pr_check = end_time_pr
                        
                        # Проверяем условие
                        condition = (start_time_pr <= end_threshold) and (end_time_pr_check >= start_threshold)
                        
                        if condition:
                            Pal_init.loc[idx, 'cartoon_series_flag'] = 'серия мультфильмов'
                    
                # Удаляем временные столбцы
                Pal_init = Pal_init.drop(['time_start_temp', 'time_end_temp'], axis = 1)

                # Производим замену названий. Если в столбце "cartoon_series_flag" фигурирует название "серия мультфильмов", то в столбце
                # "Название программы" заменяем значение на "серия мультфильмов".
                Pal_init.loc[Pal_init['cartoon_series_flag'].notna() & \
                            (Pal_init['cartoon_series_flag'] != ''), 'Название программы'] = 'серия мультфильмов'

                columns_to_remain = [
                    'Дата', 'Название программы palomars', 'Название программы', 
                    'Время выхода', 'Время окончания', 'Share', 'Жанр'
                ]

                # Оставляем только нужные столбцы для дальнейшего анализа
                Pal_init = Pal_init[columns_to_remain]
            # ========================================== КОНЕЦ НОВОГО КУСКА ==========================================

            result_df = TVScheduleProcessor(self.channel, VIMB_init, Pal_init).find_matches(minutes, target_date)
            
            # Считаем длительности программ
            result_df['Время выхода_dt'] = pd.to_datetime(result_df['Время выхода'])
            result_df['Время окончания_dt'] = pd.to_datetime(result_df['Время окончания'])
    
            # Автоматически корректируем переход через полночь
            result_df['Время окончания_dt'] = np.where(
                result_df['Время окончания_dt'] < result_df['Время выхода_dt'],
                result_df['Время окончания_dt'] + pd.Timedelta(days = 1),
                result_df['Время окончания_dt']
            )
    
            result_df['Продолжительность'] = (
                pd.to_datetime(result_df['Время окончания_dt']) - pd.to_datetime(result_df['Время выхода_dt'])
            ).dt.total_seconds()
    
            # Форматирование
            result_df['Продолжительность'] = result_df['Продолжительность'].apply(
                lambda x: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
            )
    
            result_df = result_df[
                [
                    'Дата', 'Название программы', 'Время выхода', 'Время окончания',
                    'Продолжительность', 'Share', 'Название программы init', 
                    'Жанр'
                    ]
            ]
            result_webs[target_date] = result_df
            
        webs_converted = pd.concat(result_webs.values(), ignore_index = True)

        webs_converted['Дата'] = pd.to_datetime(webs_converted['Дата'])
        sorted_webs = webs_converted.sort_values('Дата').reset_index(drop = True)
        
        res = []
        for date in dates_unique:
            date_dt = pd.to_datetime(date)
            
            t = sorted_webs[sorted_webs['Дата'] == date_dt]

            t['sort_key'] = t['Время выхода'].apply(BaseParser.get_sort_key)

            final = t.sort_values('sort_key').reset_index(drop = True)

            final = final.drop('sort_key', axis = 1)
            res.append(final)
        
        general_result = pd.concat(res).reset_index(drop = True)
        general_result['Дата'] = general_result['Дата'].dt.strftime('%Y-%m-%d')
        general_result.rename(columns = {'Share': 'Share_weighted'}, inplace = True)

        general_result['Share_weighted'] = general_result['Share_weighted'].round(6)
        
        return general_result, not_matched_programs
    

    def update_file(self, web_new):
        """
            Функция для обновления файла с сетками ТВ-программ VIMB.
        """
        if len(web_new) == 0:
            print('Ошибка! Вы пытаетесь сохранить пустой DataFrame!')

        # Проверяем существование файла
        file_path = Path(self.folder_path)
        
        if not file_path.exists():
            print(f'Файл {file_path} не найден. Создаем новый файл...')
            
            # Подготавливаем данные для записи
            new_cleaned = web_new.copy()
            
            # Приводим все к строковому типу и обрезаем пробелы
            for col in new_cleaned.columns:
                new_cleaned[col] = new_cleaned[col].astype(str).str.strip()
            
            # Создаем Excel файл с форматированием
            self.folder_path = file_path # Добавляем путь для сохранения

            self.style_of_table(
                df = new_cleaned, 
                sheet_name = 'Sheet1'
            )
            
            print(f'Создан новый файл: {file_path}')
        
        # Файл существует - читаем и обновляем
        try:
            new = web_new.copy()
            # Читаем существующие данные
            old_web = pd.read_excel(self.folder_path)

            # Приводим даты к единому формату
            for col in new.columns:
                new[col] = new[col].astype(str).str.strip()
                old_web[col] = old_web[col].astype(str).str.strip()
            
            full = pd.concat([old_web, new]).reset_index(drop = True)

            df_no_duplicates = full.drop_duplicates(
                subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
                keep = 'first'
            )
            print(f'Удалено {len(full) - len(df_no_duplicates)} дубликатов.')

            # Проверяем границы дней
            #(df_no_duplicates)

            self.style_of_table(
                df = df_no_duplicates, 
                sheet_name = 'Sheet1'
            )

        except Exception as e:
            print(f'Ошибка при обновлении файла: {e}')
        

            # Создаем резервную копию и новый файл
            try:
                backup_path = file_path.with_suffix('-копия.xlsx')
                if file_path.exists():
                    shutil.copy2(file_path, backup_path)
                    print(f'Создана резервная копия: {backup_path}')
                
                # Создаем новый файл с web_new данными
                self.folder_path = file_path

                self.style_of_table(
                    df = web_new, 
                    sheet_name = 'Sheet1'
                )
                print(f'Создан новый файл с предоставленными данными.')
                
            except Exception as backup_error:
                print(f'Критическая ошибка при создании резервной копии: {backup_error}')
    

    def style_of_table(self, df: pd.DataFrame, sheet_name: str):
        """
            Функция для генерации внешнего вида таблицы с сеткой ВИМБ.
        """
        column_configs = [
            {'header': 'Дата', 'width': 14.0, 'format': 'date'},
            {'header': 'Название программы', 'width': 72.0, 'format': 'general'},
            {'header': 'Время выхода', 'width': 14.0, 'format': 'general'},
            {'header': 'Время окончания', 'width': 14.2, 'format': 'general'},
            {'header': 'Продолжительность', 'width': 17.0, 'format': 'general'},
            {'header': 'Share_weighted', 'width': 16.0, 'format': 'general'},
            {'header': 'Название программы init', 'width': 72.0, 'format': 'general'},
            {'header': 'Жанр', 'width': 40.0, 'format': 'general'}
        ]
        
        self.make_style_of_table(
            df = df,
            sheet_name = sheet_name,
            column_configs = column_configs,
            date_columns = ['Дата']
        )
    

    def pipeline(self):
        """
            Пайплайн по совмещению сеток между собой.
        """
        # 1. Сопоставляем сетки между собой
        result_webs = self.match_vimb_with_palomars_grids()

        # 2. Обновляем/создаем файл с фактическими данными
        self.update_file(result_webs)

        return result_webs