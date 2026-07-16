import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


class Preprocessing:
    """
        Класс для предобработки файлов с исторической и новыми сетками Федеральных ТВ-каналовс регулярной сеткой
    """
    def __init__(self, filename: str):
        """
            filename: str: полный путь/название файла, который будем парсить.
        """
        self.filename = filename


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
            rest = time_str[1:]  # :MM:SS
        else:  # Формат HH:MM:SS (две цифры)
            hours = int(time_str[:2])
            rest = time_str[2:]  # :MM:SS
        
        # Применяем преобразование часов
        if hours >= 24:
            hours = hours - 24
        # Форматируем с ведущим нулем
        return f'{hours:02d}{rest}'


    def parse_Palomars(self, column_1: str, column_2: str):
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
        plmrs = pd.read_excel(self.filename, index_col = 0)
    
        columns_with_time = [column_1, column_2]
        
        #Конвертация в формат даты столбцов со слотами
        for i in range(len(columns_with_time)):
            plmrs[columns_with_time[i]] = plmrs[columns_with_time[i]].astype(str)
            list_of_dates = list(plmrs[columns_with_time[i]])
            converted = [Preprocessing.convert_time(t) for t in list_of_dates]
            plmrs[columns_with_time[i]] = plmrs[columns_with_time[i]].replace(list_of_dates, converted)
            plmrs[columns_with_time[i]] = pd.to_datetime(plmrs[columns_with_time[i]], format = '%H:%M:%S')
    
        #Вычисление длительности каждой программы. результат записывается в отдельный столбец
        plmrs['Длительность, мин'] = np.abs(np.round((plmrs[column_1] - plmrs[column_2]) / np.timedelta64(1, 'm')))
        plmrs['Длительность, мин'] = plmrs['Длительность, мин'].astype(int)
    
        #В столбцах с временем выхода и окончания программы оставляем только время
        for i in range(len(columns_with_time)):
            plmrs[columns_with_time[i]] = plmrs[columns_with_time[i]].dt.time
        return plmrs


    def parse_VIMB(self, sheet_name: str):
        """
            Функция для парсинга файла с новой сеткой из VIMBа
            Args:
                filename:
                sheet_name:
            Returns:
                VIMB: причёсанный DataFrame с сеткой VIMB.
        """
        #Чтение файла с данными
        vimb = pd.read_excel(self.filename, sheet_name = sheet_name)
        
        #Конвертация в формат времени столбца с Датой
        vimb['Дата'] = pd.to_datetime(vimb['Дата'])
    
        #Здесь идет разбивка по рекламным блокам. Из-за этого есть дубликаты. Избавимся от них
        vimb_cleaned = vimb.drop_duplicates(subset = [ 'Дата', 'День', 'Канал', 'Программа', 'Выпуск', 'Начало', 'Окончание'], 
                                            keep='first')
        #Оставляем только нужные столбцы для дальйнешего анализа
        VIMB = vimb_cleaned[['Дата', 'Программа', 'Выпуск', 'Начало', 'Окончание', 'День']].reset_index(drop = True)
    
        #Переименование столбцов ВИМБА так, чтобы они стали совпадать с Паломарс
        VIMB = VIMB.rename(columns = {
                                                    'Выпуск': 'Название программы',
                                                    'Начало': 'Время выхода',
                                                    'Окончание': 'Время окончания',
                                                    'День': 'День недели'
                                                })
        VIMB['День недели'] = VIMB['День недели'].replace({
                                                                            'Пн': 'Понедельник',
                                                                            'Вт': 'Вторник',
                                                                            'Ср': 'Среда',
                                                                            'Чт': 'Четверг',
                                                                            'Пт': 'Пятница',
                                                                            'Сб': 'Суббота',
                                                                            'Вс': 'Воскресенье'
                                                                        })
        
        #Название программы 'Камеди клаб' записано по-разному. Переименуем в Комеди клаб
        if 'Камеди клаб' in list(VIMB['Название программы']):
            VIMB['Название программы'].replace('Камеди клаб', 'Комеди Клаб', inplace = True)
        return VIMB