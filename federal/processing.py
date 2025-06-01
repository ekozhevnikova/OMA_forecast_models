import pandas as pd
import numpy as np
from OMA_tools.federal.fed_preprocessing import Federal_Preprocessing
from OMA_tools.federal.smi_info import SMI_info
from OMA_tools.federal.comments import Federal_Comments

import locale
locale.setlocale(locale.LC_ALL, 'ru_RU')

import warnings
warnings.filterwarnings('ignore')

channels_need_replace = {
                        '2Х2': '2X2',
                        '5 КАНАЛ': 'ПЯТЫЙ КАНАЛ',
                        'ПЕРВЫЙ': 'ПЕРВЫЙ КАНАЛ',
                        'СТС ЛАВ': 'СТС LOVE',
                        'ТВ3': 'ТВ-3',
                        'ТНТ4': 'ТНТ 4'
                        }

channel_names_init = {
    '2Х2': '2X2',
    'ПЯТЫЙ КАНАЛ': '5 канал',
    'ДОМАШНИЙ': 'Домашний',
    'ЗВЕЗДА': 'Звезда',
    'КАРУСЕЛЬ': 'Карусель',
    'СОЛНЦЕ': 'Солнце',
    'СУББОТА': 'Суббота',
    'СПАС': 'Спас',
    'ЧЕ': 'Че',
    'МАТЧ ТВ': 'Матч ТВ',
    'ТВ Центр': 'ТВ ЦЕНТР',
    'ТВ-3': 'ТВ3', 
    'ПЕРВЫЙ КАНАЛ': 'Первый',
    'ПЯТНИЦА': 'Пятница',
    'РЕН ТВ': 'РЕН',
    'РОССИЯ 1': 'Россия 1',
    'РОССИЯ 24': 'Россия 24',
    'СТС LOVE': 'СТС ЛАВ'
}


class Processing:
    def __init__(self, limits_file, forecast_comparison_file, cubik_file, smi_file):
        self.limits_file = limits_file
        self.forecast_comparison_file = forecast_comparison_file
        self.cubik_file = cubik_file
        self.smi_file = smi_file
    

    @staticmethod
    def define_start_stop_day(df):
        #Определение даты старта и даты конца
        start_date = pd.to_datetime(start_date)
        start = start_date.day
        month_name_start = start_date.strftime('%B')
        
        stop_date = df.index[-1]
        stop = stop_date.day
        month_name_stop = stop_date.strftime('%B')
        return start, month_name_start, stop, month_name_stop
    
    @staticmethod
    def define_limits(filepath):
        """
            Функция для чтения файла с порогами
        """
        df_limits = pd.read_excel(filepath)
        df_limits['Канал'] = df_limits['Канал'].str.upper()
        df_limits.set_index('Канал', inplace = True)
        df_limits = df_limits.T
        return df_limits
    

    @staticmethod
    def replace_name_of_months(df, column_name_with_month: str, year: str):
        """
            Функция для замены названий месяцев на формат Май'25 вместо Май.
            Args:
                df: DataFrame, в котором нужно произвести замену столбца с Месяцем
                column_name_with_month: Название колокни с месяцем
                year: Год в формате строки
            Returns:
                df: Отформартированный DataFrame
        """
        year_formatted = '\'' + year[-2:]
        months = list(df[column_name_with_month])
        formatted_months = [month + year_formatted for month in months]
        df[column_name_with_month] = df[column_name_with_month].replace(list(df[column_name_with_month]), formatted_months)
        return df
    
    
    def get_data_per_analys(self, start_date):
        """
            Вспомогательная функция для чтения файла с Порогами, Таблицы со Сравнением прогнозов и Федерального Кубика
        """
        #Определение порогов
        df_limits = Processing.define_limits(self.limits_file)
        
        #Считывание файла со сравнением прогнозов
        forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 5, sheet_name = 'Сводная')
        
        columns = ['Канал', 'Значения', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май',
            'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь',
            'Январь.1', 'Февраль.1', 'Март.1', 'Апрель.1',
            'Май.1', 'Июнь.1', 'Июль.1', 'Август.1', 'Сентябрь.1', 'Октябрь.1',
            'Ноябрь.1', 'Декабрь.1', 'Январь.2', 'Февраль.2', 'Март.2', 'Апрель.2', 'Май.2', 'Июнь.2', 'Июль.2',
            'Август.2', 'Сентябрь.2', 'Октябрь.2', 'Ноябрь.2', 'Декабрь.2']
        forecast_comparison = forecast_comparison[columns]
        
        #Чтение данных из федерального кубика
        data_cubik = Federal_Preprocessing.read_cubik(filename = self.cubik_file)

        #Обрезание данных Федерального Кубика
        prepr = Federal_Preprocessing(data_cubik)
        need_data = prepr.cut_data_cubik(start_date)

        general_df_by_dates, df_by_dates_need_comment = Federal_Preprocessing.calculate_differencies(need_data, 
                                                                                                df_limits
                                                                                                )
        return df_limits, forecast_comparison, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment

    
    def BY_DAYS(self, start_date: str, year: str):
        """
            Функция для генерация комментариев по дням
            Args:
                start_date: Дата, от которой начинаем смотреть изменения
                limits_file: Имя файла/Путь к файлу с Порогами
                cubik_file: Имя файла/Путь к файлу с Порогами
                forecast_comparison_file: Имя файла/Путь к файлу со Сравнением прогнозов
                smi_file: Имя файла/Путь к файлу по Переброскам - Сокращениям
                year: Год
            Returns:
                Обновленный файл с Комментариями
                smi_by_days: Комментарии с изменениями объемов по дням
                general_by_days: Комментарии с изменения по дням, исходя из таблицы со сравнением прогнозов, а также файла от СМИ
        """
        df_limits, forecast_comparison, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment = self.get_data_per_analys(start_date)
        ################# Генерация комментариев, исходя из файла со сравнением прогнозов #################
        data_output_dates = Federal_Comments(forecast_comparison, 
                                            df_by_dates_need_comment).get_result(df_limits)
        
        smi = SMI_info(self.smi_file)
        smi_by_days = smi.get_volumes_comments(delta_df = df_by_dates_need_comment, 
                                                channels_need_replace = channels_need_replace,
                                                year = 2025)
        smi_by_days_ = smi_by_days.copy()
        #Форматирование столбца с Месяцем
        smi_by_days = Processing.replace_name_of_months(smi_by_days, 'Месяц', year)

        flag = input('Вчера или сегодня было плановое обновление?')
        #Если было плановое обновление Сегодня или Вчера, то делаем merge Комментариев СМИ и Сравнения Прогнозов
        if flag == 'Да' or flag == 'да' or flag == 'ДА' or flag == 'дА' or flag == 'YES' or flag == 'Yes' or flag == 'yEs' or flag == 'yeS' or flag == 'yes':
            merged_df = pd.merge(data_output_dates, smi_by_days_, on = ['Канал', 'Дата', 'Месяц'], how = 'left')
            merged_df['Комментарий'] = merged_df.apply(Federal_Comments.combine_columns, axis = 1)
            general_by_days = merged_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]

            #Форматирование столбца с Месяцем
            general_by_days = Processing.replace_name_of_months(general_by_days, 'Месяц', year)
            Federal_Comments.change_channels_name(channel_names_init, general_by_days, 'Канал')

        #Если НЕ было планового обновления Сегодня или Вчера, то возвращаем исключительно Комментариев СМИ
        else:
            Federal_Comments.change_channels_name(channel_names_init, smi_by_days, 'Канал')
        return smi_by_days, general_by_days


    def SUMM(self, start_date: str, year: str):
        """
            Функция для генерация комментариев по дням
            Args:
                start_date: Дата, от которой начинаем смотреть изменения
                limits_file: Имя файла/Путь к файлу с Порогами
                cubik_file: Имя файла/Путь к файлу с Порогами
                forecast_comparison_file: Имя файла/Путь к файлу со Сравнением прогнозов
                smi_file: Имя файла/Путь к файлу по Переброскам - Сокращениям
            Returns:
                Обновленный файл с Комментариями
                smi_by_days: Комментарии с изменениями объемов по дням
                general_by_days: Комментарии с изменения по дням, исходя из таблицы со сравнением прогнозов, а также файла от СМИ
        """
        df_limits, forecast_comparison, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment = self.get_data_per_analys(start_date)
        #Выделение каналов и дат с накопленными изменениями из Федерального Кубика
        prepr = Federal_Preprocessing(data_cubik)
        df_summ_need_comment = prepr.calculate_accumulated_diff(general_df_by_dates, 
                                                                df_limits)
        #Генерация первичных комментариев с накопленными изменениями, исходя из данных Фед Кубика и таблицы со сравнением прогнозов
        data_output_summ = Federal_Comments(forecast_comparison, 
                                            df_summ_need_comment).get_result(df_limits)

        #Генерация комментариев по изменениям Объемов
        smi = SMI_info(self.smi_file)
        smi_summ = smi.get_volumes_comments(delta_df = df_summ_need_comment, 
                                                channels_need_replace = channels_need_replace,
                                                year = 2025)
        if len(smi_summ) == 0:
            #Форматирование столбца с Месяцем
            data_output_summ = Processing.replace_name_of_months(data_output_summ, 'Месяц', year)
        
            #Определение даты старта и даты конца
            day_start, month_name_start, day_stop, month_name_stop = Processing.define_start_stop_day(data_cubik)
            if month_name_start == month_name_stop:
                data_output_summ['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}.'
            else:
                data_output_summ['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}.'
            return data_output_summ
        
        else:
            smi_summ_ = smi_summ.copy()
            #Join комментариев со сравнением прогнозов и СМИ
            merged_df = pd.merge(data_output_summ, smi_summ_, on = ['Канал', 'Дата', 'Месяц'], how = 'left')
            merged_df['Комментарий'] = merged_df.apply(Federal_Comments.combine_columns, axis = 1)
            general_summ = merged_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
            #Изменение названий каналов в соответствии с тем, что было изначально
            Federal_Comments.change_channels_name(channel_names_init, general_summ, 'Канал')
            
            #Форматирование столбца с Месяцем
            general_summ = Processing.replace_name_of_months(general_summ, 'Месяц', year)
            
            ##Определение даты старта и даты конца
            day_start, month_name_start, day_stop, month_name_stop = Processing.define_start_stop_day(data_cubik)
            if month_name_start == month_name_stop:
                general_summ['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}.'
            else:
                general_summ['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}.'
            return general_summ
    

    @staticmethod
    def comments_per_period(start_date: str, data, comments_filepath_init: str, criteria = 2.0 / 3.0):
        """
            Функция для написания Накопленных Комментариев за определенный Период, начиная с какой-то даты
            Args:
                start_date: Дата, с которой начинаем смотреть изменения.
                data: Полный DataFrame с суммарными изменениями за определенный период.
                comments_filepath_init: Название файла с Комментариями.
                criteria: Критерий по дефолту равен 2/3 - если объяснено инвентеря меньше чем 2/3 недельного инвентаря, то нужно дописать комментарии.
            Returns:
                result: Датафрейм с Итоговыми Суммарными изменениями за Период, который нужно потом добавить к основной массе комментариев.
                delta_per_period: Словарь с изменениями, где ключ: Канал, значение: Суммарное изменение, которое 
                было уже объяснено за Период.
        """
        #Чтение исходного файла с Комментариями
        comments = pd.read_excel(comments_filepath_init)
        comments['Изменение GRP'] = comments['Изменение GRP'].astype(int)
        comments['Порог'] = comments['Порог'].astype(int)
        comments.rename(columns = {'условие': 'Доп столбец'}, inplace = True) 
        comments['Дата'] = pd.to_datetime(comments['Дата'])
        comments.set_index('Дата', inplace = True)
        
        comments_cleaned = Federal_Preprocessing(comments).cut_data_cubik(start_date)
        comments_cleaned = comments_cleaned[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]

        delta_per_period = {}
        res = []
        for i in range(len(data)):
            channel = data.iloc[i]['Канал']
            month = data.iloc[i]['Месяц']
            delta_grp_summ = data.iloc[i]['Изменение GRP']
            #Отбор значений только по интересующему каналу и месяцу
            df = comments_cleaned[((comments_cleaned['Канал'] == channel) & (comments_cleaned['Месяц'] == month))]
            if not df.empty:
                delta = list(df['Изменение GRP'])
                delta_cleaned = []
                if len(delta) != 0:
                    #Проверяем, чтобы изменения за период были одного знака с изменениями по дням.
                    #Отбираем только те изменения, которые одного знака
                    for d in range(len(delta)):
                        if delta[d] * delta_grp_summ > 0:
                            delta_cleaned.append(delta[d])
                #Суммируем найденные изменения за период и проверяем, чтобы было объяснено более 2/3 недельного инвентаря
                if np.abs(np.sum(delta_cleaned)) < np.abs(delta_grp_summ * criteria):
                    delta_per_period[channel] = np.sum(delta_cleaned)
                    data_need = data[((data['Канал'] == channel) & (data['Месяц'] == month))]
                    res.append(data_need)
            else:
                continue
        
        result = pd.concat(res)
        return result, delta_per_period
    

    @staticmethod
    def make_style_of_table(filepath, output_df, sheet_name):
        """
            Функция для генерации внешнего вида таблицы с Комментариями.
            Args:
                filepath: путь к файлу, в который будем сохранять итоговый результат
                output_df: DataFrame, который будем стилизировать
                sheet_name: имя листа, на который это будет записываться.
            Returns:
                Стилизированная таблица в файле xlsx
        """
        with pd.ExcelWriter(filepath, 
                        date_format = 'DD.MM.YYYY', 
                        datetime_format = 'DD.MM.YYYY',
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
            header_format_2 = workbook.add_format({'bold': True,
                                                'text_wrap': True, #перенос текста
                                                'align': 'center', #выравнение текста в ячейке
                                                'align': 'vcenter', #выравнение текста в ячейке
                                                'center_across': True,
                                                'bg_color': '#FFFFE1'
                                                })
            
            #Стиль тела таблицы для Канала, Месяца
            table_fmt_1 = workbook.add_format({'bold': True, 'align': 'left', 'border': 0})
            
            #Стиль тела таблицы для Канала, Месяца
            table_fmt_2 = workbook.add_format({'align': 'left'})
        
            #Стиль тела таблицы для Канала, Месяца
            table_fmt_3 = workbook.add_format({'align': 'left', 'italic': True})
        
            #Стиль тела таблицы для Даты, Изменения и Порога
            table_fmt_4 = workbook.add_format({'align': 'right'})
        
            format_column = workbook.add_format({'align': 'right', 'bg_color': '#FFFFE1'})
        
            
            worksheet.write('A1', 'Канал', header_format)
            worksheet.write('B1', 'Месяц', header_format)
            worksheet.write('C1', 'Дата', header_format)
            worksheet.write('D1', 'Изменение GRP', header_format)
            worksheet.write('E1', 'Порог', header_format_2)
            worksheet.write('F1', 'Доп столбец', header_format)
            worksheet.write('G1', 'Комментарий', header_format)
            worksheet.set_column('A:A', 14.0, table_fmt_1)
            worksheet.set_column('B:B', 11.7, table_fmt_1)
            worksheet.set_column('C:C', 13.9, table_fmt_4)
            worksheet.set_column('D:D', 13.9, table_fmt_4)
            worksheet.set_column('E:E', 10.7, format_column)
            worksheet.set_column('F:F', 44.0, table_fmt_3)
            worksheet.set_column('G:G', 85.0, table_fmt_2)
    

    @staticmethod
    def update_comments_file(filepath, data_new):
        """
            Функция для обновления файла с Комментариями. В процессе работы считывается файл с исходными Комментариями и в конец добавляются новые.
            В конце файл сохраняется.
        """
        comments = pd.read_excel(filepath)
        comments_full = pd.concat([comments, data_new]).reset_index(drop = True)
        Processing.make_style_of_table(filepath = filepath, 
                                       output_df = comments_full, 
                                       sheet_name = 'Sheet1')