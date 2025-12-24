import pandas as pd
import numpy as np
import os
import shutil
import glob
from pathlib import Path
import xlsxwriter
from typing import Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import locale
locale.setlocale(locale.LC_ALL, 'ru_RU')

from OMA_tools.federal.channel_forecast.calculator import *

import warnings
warnings.filterwarnings('ignore')


class TVPreprocessing:
    """
        Класс для предобработки файлов с исторической и новыми сетками Федеральных ТВ-каналовс регулярной сеткой
    """
    def __init__(self, filename: str):
        """
            filename: str: полный путь/название файла, который будем парсить.
        """
        self.filename = filename
        self.plmrs = None
        self.palomars_adjusted = None


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



    def parse_total_tv_auedience(self, date_col: str = 'Date', statistic_col: str = 'TTVRtg000') -> pd.DataFrame:
        """
            Метод для парсинга файла с Total TV Auedience.
            Args:
                date_col: название колонки с датой. По умолчанию "Date"
                statistic_col: название колонки со статистикой Total TV Auedience. По умолчанию "TTVRtg000"
            Returns:
                total_tv_audiece: pd.DataFrame: датафрейм с Total TV Auedience

        """
        total_tv_audiece = pd.read_excel(self.filename, index_col = 0)
        total_tv_audiece['Date'] = pd.to_datetime(total_tv_audiece['Date'])
        total_tv_audiece['TTVRtg000'] = total_tv_audiece['TTVRtg000'].astype(float)
        return total_tv_audiece


    def parse_Palomars(self, start_time_col: str = 'Время выхода', end_time_col: str = 'Время окончания'):
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
        self.plmrs = pd.read_excel(self.filename, index_col = 0)
    
        columns_with_time = [start_time_col, end_time_col]
        
        #Конвертация в формат даты столбцов со слотами
        def process_column(col):
            series = self.plmrs[col].astype(str)
            converted = series.apply(TVPreprocessing.convert_time)
            return pd.to_datetime(converted, format = '%H:%M:%S', errors = 'coerce')

        # Обрабатываем колонки параллельно
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_column, columns_with_time))

        # Обновляем DataFrame
        for i, col in enumerate(columns_with_time):
            self.plmrs[col] = results[i]

        #Вычисление длительности каждой программы. результат записывается в отдельный столбец
        self.plmrs['Длительность, мин'] = np.abs(np.round((self.plmrs[start_time_col] - self.plmrs[end_time_col]) / np.timedelta64(1, 'm')))
        self.plmrs['Длительность, мин'] = self.plmrs['Длительность, мин'].astype(int)
    
        #В столбцах с временем выхода и окончания программы оставляем только время
        for i in range(len(columns_with_time)):
            self.plmrs[columns_with_time[i]] = self.plmrs[columns_with_time[i]].dt.time
        return self.plmrs
    

    def _palomars_convert_time(self, df):
        """
            Функция для округления времени слотов программ в исторической сетке Palomars для какого-то конкретного дня
        """
        mars = df[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share']]
        mars['Дата'] = pd.to_datetime(mars['Дата'])
        
        # Округляем время до минут
        share_calc = TVShareCalculator(mars)
        mars['Время выхода_1min'] = share_calc.round_time('Время выхода')
        mars['Время окончания_1min'] = share_calc.round_time('Время окончания')
        
        mars_new = mars[['Дата', 'Название программы', 'Share', 'Время выхода_1min', 'Время окончания_1min']]
        mars_new.rename(columns = {'Время выхода_1min': 'Время выхода', 'Время окончания_1min': 'Время окончания'}, inplace = True)
        
        # Создаем копию оригинального столбца
        mars_new['Время выхода_новое'] = mars_new['Время выхода'].copy()
        
        # Заменяем значения начиная со второго
        for i in range(1, len(mars_new)):
            mars_new.loc[i, 'Время выхода_новое'] = mars_new.loc[i - 1, 'Время окончания']
        
        # Переименовываем колонки для наглядности
        mars_new.rename(columns = {'Время выхода': 'Время выхода_старое', 'Время выхода_новое': 'Время выхода'}, inplace = True)
        
        palomars = mars_new[['Дата', 'Название программы', 'Share', 'Время выхода', 'Время окончания']]
        self.palomars_adjusted = TVShareCalculator(palomars).adjust_hour_start()
        
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
                auedience: pd.DataFrame: ДатаФрейм с весами слотов, посчитанными через TotalTVAuedience для конкретного дня.
                column_1: столбец 1 с названием "Время выхода".
                column_2: столбец 2 с названием "Время окончания".

            Returns:
                data: pd.DataFrame: Датафрейм с новой рассчитанной долей
        """
        self.plmrs = self.parse_Palomars(start_time_col, end_time_col)

        if self.plmrs.empty:
            raise ValueError('Данные с исторической сеткой из БД Mediascope отсутствуют или не были загружены!')
        
        # 2. Проверяем наличие обязательных колонок
        required_columns = [date_col, start_time_col, end_time_col, 'Share', 'Название программы']
        missing_cols = [col for col in required_columns if col not in self.plmrs.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        # Список для хранения конвертированных ДатаФреймов
        results_list = []

        # Словарь для хранения рассчитанных суммарных долей по дням
        shares = {}

        # Отбор уникальных дат для анализа
        dates_unique = self.plmrs[date_col].unique()

        for date in dates_unique:

            try:
            
                df = self.plmrs[self.plmrs[date_col] == date].reset_index(drop = True)
                auedience = weighted_auedience[weighted_auedience['Date'] == date].reset_index(drop = True)

                plmrs_new = self._palomars_convert_time(df)
                res, share = TVShareCalculator(plmrs_new).calculate_weighted_share(auedience)
                
                results_list.append(res)
                shares[date] = share
            
            except Exception as e:
                print(f'Ошибка при обработке {date}: {str(e)}')
        
        # Объединение результатов
        if not results_list:
            print("Нет результатов для объединения")
            return pd.DataFrame(), {}
        
        combined_result = pd.concat(results_list).reset_index(drop = True)


        res = []
        for date in dates_unique:
            t = combined_result[combined_result[date_col] == date]

            t['sort_key'] = t[start_time_col].apply(TVPreprocessing.get_sort_key)

            final = t.sort_values('sort_key').reset_index(drop = True)

            final = final.drop('sort_key', axis = 1)
            res.append(final)
        
        general_result = pd.concat(res).reset_index(drop = True)

        return general_result, shares
   
    
    def parse_VIMB(self, sheet_name: str = 'ГРАФИК', skiprows = 1):
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
        time_mask = (pd.to_timedelta(VIMB['Время выхода']) >= pd.Timedelta(hours = 5)) & (pd.to_timedelta(VIMB['Время выхода']) < pd.Timedelta(hours = 6))
        VIMB.loc[time_mask, 'Дата'] = VIMB.loc[time_mask, 'Дата'] + pd.Timedelta(days = 1)
        
        VIMB['День недели'] = VIMB['Дата'].dt.strftime('%A').str.capitalize()

        # Если нужно вернуть в строковый формат
        VIMB['Дата'] = VIMB['Дата'].dt.strftime('%Y-%m-%d')

        #Название программы 'Камеди клаб' записано по-разному. Переименуем в Комеди клаб
        if 'Камеди клаб' in list(VIMB['Название программы']):
            VIMB['Название программы'].replace('Камеди клаб', 'Комеди Клаб', inplace = True)
        return VIMB


class VIMBGridProcessor:
    """
        Класс для парсинга сеток VIMB (Сводная таблица)
    """
    def __init__(self, folder_path: str):
        """
            Атрибуты:
                folder_path: путь к файлам с новыми сетками ТВ-программ.
        """
        self.folder_path = folder_path


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
        xlsx_files = glob.glob(os.path.join(self.folder_path, file_format))

        files = []
        # Перебираем найденные файлы и читаем их
        for file_path in xlsx_files:
            try:
                # Читаем файл в DataFrame
                vimb = TVPreprocessing(file_path).parse_VIMB()
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
        
                t['sort_key'] = t[time_column].apply(TVPreprocessing.get_sort_key)
        
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
        
        df = df[['Дата', 'Время выхода', 'Время окончания', 'Прод-ть', 'Название программы', 'День недели', 'original_index']]
    
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

            t['sort_key'] = t['Время выхода'].apply(TVPreprocessing.get_sort_key)

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

        # Проверка, что каждый день начинается в 05:00:00 и заканчивается в 05:00:00
        dates_unique = df_check[date_column].unique()

        for date in dates_unique:

            table = df_check[df_check['Дата'] == date]
            start = table[table['Время выхода'] == '05:00:00']
            stop = table[table['Время окончания'] == '04:59:59']

            if len(start) == 0 and len(table) != 1:
                print(f'⚠️ Для {date} не найдена стартовая программа дня.')

            elif len(stop) == 0  and len(table) != 1:
                print(f'⚠️ Для {date} не найдена кульминационная программа дня.')
    

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
            
            # Проверяем границы дней перед сохранением
            self.check_start__and__end_day(new_cleaned)
            
            # Создаем Excel файл с форматированием
            self.folder_path = file_path # Добавляем путь для сохранения

            self.make_style_of_table(
                output_df = new_cleaned, 
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
                subset = ['Дата', 'Время выхода', 'Время окончания', 'Прод-ть', 'Название программы', 'День недели'],
                keep = 'first'
            )
            print(f'Удалено {len(full) - len(df_no_duplicates)} дубликатов.')

            # Проверяем границы дней
            self.check_start__and__end_day(df_no_duplicates)

            self.make_style_of_table(
                output_df = df_no_duplicates, 
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

                self.make_style_of_table(
                    output_df = web_new, 
                    sheet_name = 'Sheet1'
                )
                print(f'Создан новый файл с предоставленными данными.')
                
            except Exception as backup_error:
                print(f'Критическая ошибка при создании резервной копии: {backup_error}')

    

    def make_style_of_table(self, output_df, sheet_name):
        """
            Функция для генерации внешнего вида таблицы с сеткой ВИМБ.
            Args:
                filepath: путь к файлу, в который будем сохранять итоговый результат
                output_df: DataFrame, который будем стилизировать
                sheet_name: имя листа, на который это будет записываться.
            Returns:
                Стилизированная таблица в файле xlsx
        """
        with pd.ExcelWriter(self.folder_path, 
                        date_format = '%Y-%m-%d',
                        datetime_format = '%Y-%m-%d',
                        engine = 'xlsxwriter') as writer:
            
            output_df.to_excel(writer, index = None)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
        
            #Стиль шапки таблицы
            header_format = workbook.add_format({'bold': True,
                                                'text_wrap': True, #перенос текста
                                                'align': 'center', #выравнение текста в ячейке
                                                'align': 'vcenter', #выравнение текста в ячейке
                                                'center_across': True
                                                })
            
            #Стиль тела таблицы для Канала, Месяца
            table_fmt = workbook.add_format({'bold': False, 'align': 'center', 'border': 0})
            
            worksheet.write('A1', 'Дата', header_format)
            worksheet.write('B1', 'Время выхода', header_format)
            worksheet.write('C1', 'Время окончания', header_format)
            worksheet.write('D1', 'Прод-ть', header_format)
            worksheet.write('E1', 'Название программы', header_format)
            worksheet.write('F1', 'День недели', header_format)
            worksheet.set_column('A:A', 14.0, table_fmt)
            worksheet.set_column('B:B', 14.0, table_fmt)
            worksheet.set_column('C:C', 14.2, table_fmt)
            worksheet.set_column('D:D', 14.0, table_fmt)
            worksheet.set_column('E:E', 72.0, table_fmt)
            worksheet.set_column('F:F', 12.0, table_fmt)