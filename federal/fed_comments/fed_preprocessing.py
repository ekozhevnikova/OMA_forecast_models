import pandas as pd
import numpy as np
import re
from datetime import datetime


class Federal_Preprocessing:
    """
        Класс для ПредОбработки данных, необходимых для генерации комментариев.
    """
    def __init__(self, df):
        self.df = df


    @staticmethod
    def extract_date_from_filename(filename):
        """
            Функция для извлечения даты из файла с использованием регулярных выражений.
        """
        # Регулярное выражение для поиска даты в формате DD.MM.YYYY
        date_pattern = r'\((\d{2})\.(\d{2})\.(\d{4})\)'
        
        match = re.search(date_pattern, filename)
        if match:
            # Извлечение группы с найденной датой
            date_str = match.group(0)[1: -1]  # Убираем скобки
            # Преобразование строки в объект datetime
            return datetime.strptime(date_str, '%d.%m.%Y')
        else:
            raise ValueError("Дата не найдена в названии файла.")


    @staticmethod
    def read_cubik(filename, sheet_name = 'прогнозВИ'):
        """
            Функция для чтения данных из Федерального Кубика
            Args:
                filename: Полный путь к файлу с данными из Федерального Кубика
                sheet_name: имя листа, с которого будем брать данные (по умолчанию "прогнозВИ")
            Return:
                data_cubik: Данные из Федерального кубика
        """
        try:
            data_cubik = pd.read_excel(filename, sheet_name = sheet_name, skiprows = 2)
        
            #Конвертация столбца с Датой выгрузки в формат даты
            data_cubik['Дата историрования'] = pd.to_datetime(data_cubik['Дата историрования'])
            
            old_dates = list(data_cubik['Дата историрования'])
            new_dates = []
            for i in range(len(old_dates)):
                new_date = old_dates[i].strftime('%Y-%m-%d')
                new_dates.append(new_date)
            data_cubik['Дата историрования'] = data_cubik['Дата историрования'].replace(old_dates, new_dates)
            
            #Сортировка значений в столбце с Периодом по возрастанию
            data_cubik.sort_values(by = 'Дата историрования', inplace = True)
            data_cubik.set_index('Дата историрования', inplace = True)
            
            #Замена формата значений на тип int
            for column in data_cubik.columns[1:]:
                data_cubik[column] = data_cubik[column].astype(float)
            
            #Конвертация названий столбцов в капс
            cols_transform = {}
            for col in list(data_cubik.columns):
                cols_transform[col] = col.upper()
            data_cubik.rename(cols_transform, axis = 'columns', inplace = True)
            
            #Переименование некоторых каналов
            data_cubik.rename({'ПЕРВЫЙ': 'ПЕРВЫЙ КАНАЛ', 
                            '5 КАНАЛ': 'ПЯТЫЙ КАНАЛ', 
                            'ТВ3': 'ТВ-3', 
                            'ТНТ4': 'ТНТ 4', 
                            '2Х2': '2X2',
                            'СТС ЛАВ': 'СТС LOVE'}, 
                            axis = 'columns', 
                            inplace = True)
            return data_cubik
        except FileNotFoundError:
            print('Файл с данными из Федерального кубика не найден! Пожалуйста, добавьте его в соответствующую папку!')
    
    
    def cut_data_cubik(self, start_date: str):
        """
            Функция для выделения данных за определенный период для анализа из Федерального Кубика
            Args:
                data_cubik: DataFrame Федеральный кубик
                start_date: дата в формате Год-месяц-день ('%Y-%m-%d')
            Return:
                need_data: Обрезанный DataFrame из Федерального кубика
        """
        #Определение даты, начиная с которой будем смотреть изменения GRP
        #start_date = '2025-04-03 15:00:00'
        start_date = pd.to_datetime(start_date, format = '%Y-%m-%d')
        
        dates = self.df.index.to_list()
        idx_of_start_date = dates.index(start_date)
        
        #Обрезание DataFrame до нужной даты
        need_data = self.df.iloc[idx_of_start_date:]
        need_data.reset_index(inplace = True)
        return need_data
    

    @staticmethod
    def calculate_differencies(need_data, df_limits):
        """
            Функция для расчета изменений по дням, исходя из Федерального кубика.
            Args:
                need_data: обрезанные данные из Федерального кубика
                df_limits: DataFrame с порогами
            Return:
                general_df_by_dates: DataFrame с изменениями по дням для каждого канала   
                df_by_dates_need_comment: DataFrame с изменениями по дням для каждого канала, приведенный к определенному виду
        """
        #data_full_by_days = {}
        #for column in need_data.columns[2:]:
        #    df = need_data[['Дата историрования', 'ПЕРИОД', column]]
        #    
        #    data_with_difference_by_dates = {}
        #    dict_differences = {}
        #    #Расчет изменений GRP по дням для каждого канала
        #    for i in range(len(df)):
        #        if i > 0:
        #            for j in range(i - 1, -1, -1):
        #                if df.iloc[i]['ПЕРИОД'] == df.iloc[j]['ПЕРИОД'] and \
        #                (pd.to_datetime(df.iloc[i][0]) - pd.to_datetime(df.iloc[j][0])).days == 1:
        #                    diff = df.iloc[i][-1] - df.iloc[j][-1]
        #                    #Добавить условие на порог
        #                    period = df.iloc[i]['ПЕРИОД']
        #                    dict_differences = {
        #                        'Канал': column,
        #                        'Месяц': f'{period}', 
        #                        'Дата': df.iloc[i][0],
        #                        'Изменение GRP': diff,
        #                        'Flag': np.abs(diff) > int(df_limits[column])
        #                    }
        #            data_with_difference_by_dates[i] = dict_differences
        #            
        #    df_difference_per_day = pd.DataFrame(data_with_difference_by_dates).T
        #    df_difference_per_day.dropna(inplace = True)
        #    df_difference_per_day.sort_values(by = 'Месяц', inplace = True)
        #    #Запись в словарь изменений по дням
        #    data_full_by_days[column] = df_difference_per_day
        #print(data_full_by_days['ДОМАШНИЙ'])
        #    
        ##Сбор изменений за каждую дату в единый DataFrame
        #results_by_dates = []
        #for channel, accumulated_summs in data_full_by_days.items():
        #    results_by_dates.append(data_full_by_days[channel])
        #general_df_by_dates = pd.concat(results_by_dates).reset_index(drop = True)
        ##Отбор каналов и дат, которые вылетели за порог
        #df_by_dates_need_comment = general_df_by_dates.loc[(general_df_by_dates['Flag'] == True)]
        #df_by_dates_need_comment = df_by_dates_need_comment.reset_index(drop = True)
        #df_by_dates_need_comment['Дата'] = pd.to_datetime(df_by_dates_need_comment['Дата'], format = '%Y-%m-%d').dt.strftime('%Y-%m-%d')
        #df_by_dates_need_comment['Дата'] = pd.to_datetime(df_by_dates_need_comment['Дата'])
        def round_half_up(x):
            return int(x + 0.5)

        data_full_by_days = {}
        for column in need_data.columns[2:]:
            df = need_data[['Дата историрования', 'ПЕРИОД', column]]
            
            data_with_difference_by_dates = {}
            
            for i in range(1, len(df)):  # Начинаем с 1, т.к. i > 0
                current_date = pd.to_datetime(df.iloc[i]['Дата историрования'])
                current_period = df.iloc[i]['ПЕРИОД']
                current_grp = df.iloc[i][-1]

                # Ищем предыдущий день с тем же периодом
                for j in range(i - 1, -1, -1):
                    prev_date = pd.to_datetime(df.iloc[j]['Дата историрования'])
                    prev_period = df.iloc[j]['ПЕРИОД']
                    
                    if (prev_period == current_period and 
                        (current_date - prev_date).days == 1):
                        
                        diff = int(round_half_up(current_grp) - round_half_up(df.iloc[j][-1]))
                        
                        # СОЗДАЕМ НОВЫЙ СЛОВАРЬ ДЛЯ КАЖДОЙ НАЙДЕННОЙ ПАРЫ
                        dict_difference = {
                            'Канал': column,
                            'Месяц': f'{current_period}', 
                            'Дата': current_date.strftime('%Y-%m-%d'),
                            'Изменение GRP': diff,
                            'Flag': np.abs(diff) > int(df_limits[column])
                        }
                        
                        # УНИКАЛЬНЫЙ КЛЮЧ ДЛЯ КАЖДОЙ ПАРЫ
                        key = f"{i}_{j}"
                        data_with_difference_by_dates[key] = dict_difference
                        break  # Прерываем после нахождения ближайшего предыдущего дня
            
            df_difference_per_day = pd.DataFrame(data_with_difference_by_dates).T
            if not df_difference_per_day.empty:
                df_difference_per_day.dropna(inplace=True)
                df_difference_per_day.sort_values(by='Месяц', inplace=True)
                #print(df_difference_per_day)
            
            data_full_by_days[column] = df_difference_per_day

        # Объединение результатов
        results_by_dates = []
        for channel in data_full_by_days:
            if not data_full_by_days[channel].empty:
                results_by_dates.append(data_full_by_days[channel])

        general_df_by_dates = pd.concat(results_by_dates).reset_index(drop=True) if results_by_dates else pd.DataFrame()
        #Отбор каналов и дат, которые вылетели за порог
        df_by_dates_need_comment = general_df_by_dates.loc[(general_df_by_dates['Flag'] == True)]
        df_by_dates_need_comment = df_by_dates_need_comment.reset_index(drop = True)
        df_by_dates_need_comment['Дата'] = pd.to_datetime(df_by_dates_need_comment['Дата'], format = '%Y-%m-%d').dt.strftime('%Y-%m-%d')
        df_by_dates_need_comment['Дата'] = pd.to_datetime(df_by_dates_need_comment['Дата'])
        return general_df_by_dates, df_by_dates_need_comment
    
    
    def calculate_accumulated_diff(self, general_df_by_dates, df_limits):
        """
            Функция для расчета накопленных измененй за несколько дней.
            Args:
                general_df_by_dates: словарь с изменениями по дням, согласно данным из Федерального кубика
                df_limits: DataFrame с порогами
            Returns:
                df_summ_need_comment: DataFrame с накопленными изменениями за определенный период
        """
        dates = self.df.index.to_list()
        #Сбор суммарнных изменений за период в единый DataFrame
        summed_data = general_df_by_dates.groupby(['Канал', 'Месяц'], as_index = False)['Изменение GRP'].sum()
        flags = []
        for i in range(len(summed_data)):
            channel = summed_data.iloc[i]['Канал']
            flags.append(np.abs((summed_data.iloc[i]['Изменение GRP'])) > int(df_limits[channel]))
        summed_data['Flag'] = flags
        #Отбор каналов и дат, которые вылетели за порог
        df_summ_need_comment = summed_data.loc[(summed_data['Flag'] == True)]
        df_summ_need_comment = df_summ_need_comment.reset_index(drop = True)
        #Добавление столбца с датой и дальнейшее его преобразования
        df_summ_need_comment['Дата'] = dates[-1]
        df_summ_need_comment['Дата'] = pd.to_datetime(df_summ_need_comment['Дата'], format = '%Y-%m-%d').dt.strftime('%Y-%m-%d')
        df_summ_need_comment['Дата'] = pd.to_datetime(df_summ_need_comment['Дата'])
        #Установка столбцов в правильном порядке
        df_summ_need_comment = df_summ_need_comment[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Flag']]
        df_summ_need_comment = df_summ_need_comment.sort_values(by = 'Месяц')
        return df_summ_need_comment


    @staticmethod
    def influence_out_house(kus_file):
        """
            Функция для чтения файла с коэффициентами внедома
            Args:
                kus_file: путь к файлу с прогнозом КУСа.
            Returns:
                KUS_koeff_cleaned: DataFrame c коэффициентами внедома
        """
        try:
            KUS_koeff = pd.read_excel(kus_file, sheet_name = 'коэф.внедом', skiprows = 2)
            KUS_koeff = KUS_koeff[['Канал', 'январь.2', 'февраль.2', 'март.2', 'апрель.2', 'май.2',
                'июнь.2', 'июль.2', 'август.2', 'сентябрь.2', 'октябрь.2', 'ноябрь.2',
                'декабрь.2']]
            KUS_koeff_cleaned = KUS_koeff.dropna() 
            
            KUS_koeff_cleaned.rename(columns = {
                'январь.2': 'Январь.2',
                'февраль.2': 'Февраль.2',
                'март.2': 'Март.2',
                'апрель.2': 'Апрель.2',
                'май.2': 'Май.2',
                'июнь.2': 'Июнь.2',
                'июль.2': 'Июль.2',
                'август.2': 'Август.2',
                'сентябрь.2': 'Сентябрь.2',
                'октябрь.2': 'Октябрь.2',
                'ноябрь.2': 'Ноябрь.2',
                'декабрь.2': 'Декабрь.2'
                },
                inplace = True)
            
            #KUS_koeff_cleaned = Federal_Comments.change_channels_name(channel_names_init, KUS_koeff_cleaned, 'Канал')
            KUS_koeff_cleaned['Канал'] = KUS_koeff_cleaned['Канал'].str.upper()
            
            #Изменение столбца с каналами
            channels_need_replace = {
                        '2Х2': '2X2',
                        '5 КАНАЛ': 'ПЯТЫЙ КАНАЛ',
                        'ПЕРВЫЙ': 'ПЕРВЫЙ КАНАЛ',
                        'СТС ЛАВ': 'СТС LOVE',
                        'ТВ3': 'ТВ-3',
                        'ТНТ4': 'ТНТ 4'
                    }
            channels_old = list(KUS_koeff_cleaned['Канал'])
            channels_new = []
            for i in range(len(channels_old)):
                channels_new.append(channels_old[i].upper())
            KUS_koeff_cleaned['Канал'] = KUS_koeff_cleaned['Канал'].replace(channels_old, channels_new)
            KUS_koeff_cleaned['Канал'].replace(channels_need_replace, inplace = True)

            try:
                extracted_date = Federal_Preprocessing.extract_date_from_filename(kus_file)
                extracted_date_ = pd.to_datetime(extracted_date)
            except ValueError as e:
                print('Дата не найдена.')
            return KUS_koeff_cleaned, extracted_date_
        except FileNotFoundError:
            print('Файл с прогнозом КУСа не найден! Пожалуйста, добавьте его в соответствующую папку!')