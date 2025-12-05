import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import locale
locale.setlocale(locale.LC_ALL, 'ru_RU')

from OMA_tools.federal.channel_forecast.core.simple_models import TVShareCalculator

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
            converted = series.apply(Preprocessing.convert_time)
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

            t['sort_key'] = t[start_time_col].apply(Preprocessing.get_sort_key)

            final = t.sort_values('sort_key').reset_index(drop = True)

            final = final.drop('sort_key', axis = 1)
            res.append(final)
        
        general_result = pd.concat(res).reset_index(drop = True)

        return general_result, shares



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