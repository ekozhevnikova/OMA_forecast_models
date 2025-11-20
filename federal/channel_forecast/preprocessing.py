import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import locale
locale.setlocale(locale.LC_ALL, 'ru_RU')

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
            rest = time_str[1: ]  # :MM:SS
        else:  # Формат HH:MM:SS (две цифры)
            hours = int(time_str[: 2])
            rest = time_str[2: ]  # :MM:SS
        
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
        def process_column(col):
            series = plmrs[col].astype(str)
            converted = series.apply(Preprocessing.convert_time)
            return pd.to_datetime(converted, format='%H:%M:%S', errors='coerce')

        # Обрабатываем колонки параллельно
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_column, columns_with_time))

        # Обновляем DataFrame
        for i, col in enumerate(columns_with_time):
            plmrs[col] = results[i]

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
                sheet_name: имя листа, который будем считывать из файла.
            Returns:
                VIMB: причёсанный DataFrame с сеткой VIMB.
        """
        #Чтение файла с данными
        vimb = pd.read_excel(self.filename, sheet_name = sheet_name)
        
        #Конвертация в формат времени столбца с Датой
        vimb['Дата'] = pd.to_datetime(vimb['Дата'])
    
        #Здесь идет разбивка по рекламным блокам. Из-за этого есть дубликаты. Избавимся от них
        vimb_cleaned = vimb.drop_duplicates(subset = [ 'Дата', 'День', 'Канал', 'Программа', 'Выпуск', 'Начало', 'Окончание'], 
                                            keep = 'first')
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
        #if 'Камеди клаб' in list(VIMB['Название программы']):
        #    VIMB['Название программы'].replace('Камеди клаб', 'Комеди Клаб', inplace = True)
        return VIMB
    
    
    def parse_VIMB_un_altro(self, sheet_name: str = 'ГРАФИК', skiprows = 1):
        """
            Ещё один метод для парсинга файла с сеткой VIMB из отчета Размещение -> Сводная таблица
            Args:
                sheet_name: имя листа, который будем считывать из файла. По умолчанию ГРАФИК.
                skiprows: количество строк, которые будем пропускать в файле. По умолчанию 1.
            Returns:
                VIMB: причёсанный DataFrame с сеткой VIMB.
        """
        # Чтение файла
        df = pd.read_excel(self.filename, sheet_name = sheet_name, skiprows = skiprows)

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

        # Создаем маску и увеличиваем дату
        mask = VIMB['Время выхода'] == '05:00:00'
        VIMB.loc[mask, 'Дата'] = VIMB.loc[mask, 'Дата'] + pd.Timedelta(days = 1)

        # Если нужно вернуть в строковый формат
        VIMB['Дата'] = VIMB['Дата'].dt.strftime('%Y-%m-%d')

        #Название программы 'Камеди клаб' записано по-разному. Переименуем в Комеди клаб
        if 'Камеди клаб' in list(VIMB['Название программы']):
            VIMB['Название программы'].replace('Камеди клаб', 'Комеди Клаб', inplace = True)
        return VIMB