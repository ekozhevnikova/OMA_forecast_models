import pandas as pd
import numpy as np


class Federal_Preprocessing:
    """
        Класс для ПредОбработки данных, необходимых для генерации комментариев.
    """
    def __init__(self, df):
        self.df = df

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
            data_cubik[column] = data_cubik[column].astype(int)
        
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
        data_full_by_days = {}
        for column in need_data.columns[2:]:
            df = need_data[['Дата историрования', 'ПЕРИОД', column]]
            
            data_with_difference_by_dates = {}
            dict_differences = {}
            #Расчет изменений GRP по дням для каждого канала
            for i in range(len(df)):
                if i > 0:
                    for j in range(i - 1, -1, -1):
                        if df.iloc[i]['ПЕРИОД'] == df.iloc[j]['ПЕРИОД'] and \
                        (pd.to_datetime(df.iloc[i][0]) - pd.to_datetime(df.iloc[j][0])).days == 1:
                            diff = df.iloc[i][-1] - df.iloc[j][-1]
                            #Добавить условие на порог
                            period = df.iloc[i]['ПЕРИОД']
                            dict_differences = {
                                'Канал': column,
                                'Месяц': f'{period}', 
                                'Дата': df.iloc[i][0],
                                'Изменение GRP': diff,
                                'Flag': np.abs(diff) > int(df_limits[column])
                            }
                    data_with_difference_by_dates[i] = dict_differences
                    
            df_difference_per_day = pd.DataFrame(data_with_difference_by_dates).T
            df_difference_per_day.dropna(inplace = True)
            df_difference_per_day.sort_values(by = 'Месяц', inplace = True)
            #Запись в словарь изменений по дням
            data_full_by_days[column] = df_difference_per_day
            
        #Сбор изменений за каждую дату в единый DataFrame
        results_by_dates = []
        for channel, accumulated_summs in data_full_by_days.items():
            results_by_dates.append(data_full_by_days[channel])
        general_df_by_dates = pd.concat(results_by_dates).reset_index(drop = True)
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