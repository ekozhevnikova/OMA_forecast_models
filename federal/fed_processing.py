import pandas as pd
import numpy as np
import re
import pymorphy3 as pmrph
import datetime
from OMA_tools.io_data.operations import Dict_Operations
from OMA_tools.federal.fed_preprocessing import Federal_Preprocessing
from OMA_tools.federal.fed_postprocessing import Federal_Postprocessing
from OMA_tools.federal.smi_info import SMI_info
from OMA_tools.federal.comments import Federal_Comments

import locale
locale.setlocale(locale.LC_ALL, 'ru_RU')

import warnings
warnings.filterwarnings('ignore')

class color:
   BOLD = '\033[1m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   PURPLE = '\033[95m'
   RED = '\033[91m'
   END = '\033[0m'
   UNDERLINE = '\033[4m'

channels_need_replace = {
                        '2Х2': '2X2',
                        '5 КАНАЛ': 'ПЯТЫЙ КАНАЛ',
                        'ПЕРВЫЙ': 'ПЕРВЫЙ КАНАЛ',
                        'СТС ЛАВ': 'СТС LOVE',
                        'ТВ3': 'ТВ-3',
                        'ТНТ4': 'ТНТ 4'
                        }

channel_names_init = {
    '2X2': '2х2',
    'ПЯТЫЙ КАНАЛ': '5 канал',
    'ДОМАШНИЙ': 'Домашний',
    'ЗВЕЗДА': 'Звезда',
    'КАРУСЕЛЬ': 'Карусель',
    'СОЛНЦЕ': 'Солнце',
    'СУББОТА': 'Суббота',
    'СПАС': 'Спас',
    'ЧЕ': 'Че',
    'МАТЧ ТВ': 'Матч ТВ',
    'ТВ ЦЕНТР': 'ТВ Центр',
    'ТВ-3': 'ТВ3', 
    'ПЕРВЫЙ КАНАЛ': 'Первый',
    'ПЯТНИЦА': 'Пятница',
    'РЕН ТВ': 'РЕН',
    'РОССИЯ 1': 'Россия 1',
    'РОССИЯ 24': 'Россия 24',
    'СТС LOVE': 'СТС ЛАВ'
}


class Federal_Processing:
    def __init__(self, limits_file, forecast_comparison_file, cubik_file, smi_file):
        self.limits_file = limits_file
        self.forecast_comparison_file = forecast_comparison_file
        self.cubik_file = cubik_file
        self.smi_file = smi_file
    

    @staticmethod
    def define_start_stop_day(df, start_date):
        #Определение даты старта и даты конца
        start_date = pd.to_datetime(start_date)
        start = start_date.day
        month_name_start = start_date.strftime('%B')
        
        stop_date = df.index[-1]
        stop = stop_date.day
        month_name_stop = stop_date.strftime('%B')
        return start, month_name_start, stop, month_name_stop
    

    @staticmethod
    def define_limits(limits_file):
        """
            Функция для чтения файла с порогами
        """
        df_limits = pd.read_excel(limits_file)
        df_limits['Канал'] = df_limits['Канал'].str.upper()
        df_limits.set_index('Канал', inplace = True)
        df_limits = df_limits.T
        return df_limits
    
    
    def get_data_per_analys(self, start_date, flag = True):
        """
            Вспомогательная функция для чтения файла с Порогами, Таблицы со Сравнением прогнозов и Федерального Кубика
        """
        #Определение порогов
        df_limits = Federal_Processing.define_limits(self.limits_file)
        #Чтение данных из федерального кубика
        data_cubik = Federal_Preprocessing.read_cubik(filename = self.cubik_file)

        #Обрезание данных Федерального Кубика
        prepr = Federal_Preprocessing(data_cubik)
        need_data = prepr.cut_data_cubik(start_date)

        general_df_by_dates, df_by_dates_need_comment = Federal_Preprocessing.calculate_differencies(need_data, 
                                                                                                df_limits
                                                                                                )
        if flag:
            #Считывание файла со сравнением прогнозов
            forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 2, sheet_name = 'Сводная')

            #date_old_forecast = forecast_comparison.iloc[0]['Дата обновления']
            date_new_forecast = forecast_comparison.iloc[0]['Unnamed: 23']

            forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 5, sheet_name = 'Сводная')
            
            columns = ['Канал', 'Значения', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май',
                'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь',
                'Январь.1', 'Февраль.1', 'Март.1', 'Апрель.1',
                'Май.1', 'Июнь.1', 'Июль.1', 'Август.1', 'Сентябрь.1', 'Октябрь.1',
                'Ноябрь.1', 'Декабрь.1', 'Январь.2', 'Февраль.2', 'Март.2', 'Апрель.2', 'Май.2', 'Июнь.2', 'Июль.2',
                'Август.2', 'Сентябрь.2', 'Октябрь.2', 'Ноябрь.2', 'Декабрь.2']
            forecast_comparison = forecast_comparison[columns]
            return df_limits, forecast_comparison, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment, date_new_forecast
        
        else:
            return df_limits, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment

    
    def BY_DAYS(self, start_date: str, year: str, smi_criteria):
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
        question = input('Вчера или сегодня было плановое обновление?')

        #Если было плановое обновление Сегодня или Вчера, то делаем merge Комментариев СМИ и Сравнения Прогнозов
        if question == 'Да' or question == 'да' or question == 'ДА' or question == 'дА' or question == 'YES' or question == 'Yes' or question == 'yEs' or question == 'yeS' or question == 'yes':
            problem_channels = {}
            df_limits, forecast_comparison, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment, date_new_forecast = self.get_data_per_analys(start_date, flag = True)
            ################# Генерация комментариев, исходя из файла со сравнением прогнозов #################
            data_output_dates, channels_not_exist, channels_not_enough_reasons = Federal_Comments(forecast_comparison, 
                                                df_by_dates_need_comment).get_result(df_limits,  date_new_forecast, flag = True)
            
            smi = SMI_info(self.smi_file)
            smi_by_days, channels_not_found_smi = smi.get_volumes_comments(delta_df = df_by_dates_need_comment, 
                                                    channels_need_replace = channels_need_replace,
                                                    year = 2025, 
                                                    df_limits = df_limits, 
                                                    smi_criteria = smi_criteria)
            smi_by_days_ = smi_by_days.copy()
            if len(smi_by_days) == 0:
                data_output_dates_ = data_output_dates[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
                #Форматирование столбца с Месяцем
                by_days = Federal_Postprocessing(data_output_dates_).replace_name_of_months('Месяц', year)
                Federal_Comments.change_channels_name(channel_names_init, by_days, 'Канал')
                Federal_Postprocessing(by_days).comments_dublicates_actualize()
                problem_channels = {
                    'Channel not exist': channels_not_exist,
                    'Not enough reasons': channels_not_enough_reasons,
                    'SMI not': channels_not_found_smi
                }
                return by_days, problem_channels
            else:
                merged_df = pd.merge(data_output_dates, smi_by_days_, on = ['Канал', 'Дата', 'Месяц'], how = 'left')
                merged_df['Комментарий'] = merged_df.apply(Federal_Comments.combine_columns, axis = 1)

                general_by_days = merged_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]

                #Форматирование столбца с Месяцем
                general_by_days = Federal_Postprocessing(general_by_days).replace_name_of_months('Месяц', year)
                general_by_days_ = Federal_Comments.change_channels_name(channel_names_init, general_by_days, 'Канал')
                Federal_Postprocessing(general_by_days_).comments_dublicates_actualize()
                problem_channels = {
                    'Channel not exist': channels_not_exist,
                    'Not enough reasons': channels_not_enough_reasons,
                    'SMI not': channels_not_found_smi
                }
                return general_by_days_, problem_channels
            
        else:
            df_limits, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment = self.get_data_per_analys(start_date, flag = False)

            smi = SMI_info(self.smi_file)
            smi_by_days, channels_not_found_smi = smi.get_volumes_comments(delta_df = df_by_dates_need_comment, 
                                                    channels_need_replace = channels_need_replace,
                                                    year = 2025, 
                                                    df_limits = df_limits, 
                                                    smi_criteria = smi_criteria)
            
            #Замена столбца с месяцем
            months_init = list(df_by_dates_need_comment['Месяц'])
            months_new = []
            for old_month in months_init:
                month_new = str(old_month).split('\'')[0].title()
                months_new.append(month_new)
            df_by_dates_need_comment['Месяц'] = df_by_dates_need_comment['Месяц'].replace(months_init, months_new)

            merged_df = pd.merge(df_by_dates_need_comment, smi_by_days, on = ['Канал', 'Дата', 'Месяц'], how = 'left')

            #Добавление пустого столбца для комментариев руководителя
            merged_df['Доп столбец'] = ''

            #Join комментариев с порогами
            by_days = pd.merge(merged_df, df_limits.T, on = ['Канал'], how = 'inner')

            by_days = by_days[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]

            by_days_final = Federal_Comments.change_channels_name(channel_names_init, by_days, 'Канал')
            by_days_final_ = Federal_Postprocessing(by_days_final).replace_name_of_months('Месяц', year)
            Federal_Postprocessing(by_days_final_).comments_dublicates_actualize()
            problem_channels = {
                    'Channel not exist': '',
                    'Not enough reasons': '',
                    'SMI not': channels_not_found_smi
                }
            return by_days_final_, problem_channels


    def SUMM(self, start_date: str, year: str, smi_criteria):
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
        df_limits, forecast_comparison, data_cubik, need_data, general_df_by_dates, df_by_dates_need_comment, date_new_forecast = self.get_data_per_analys(start_date)
        #Выделение каналов и дат с накопленными изменениями из Федерального Кубика
        prepr = Federal_Preprocessing(data_cubik)
        df_summ_need_comment = prepr.calculate_accumulated_diff(general_df_by_dates, 
                                                                df_limits)
        #print(df_summ_need_comment)
        #Генерация первичных комментариев с накопленными изменениями, исходя из данных Фед Кубика и таблицы со сравнением прогнозов
        data_output_summ, channels_not_exist, channels_not_enough_reasons = Federal_Comments(forecast_comparison, 
                                                                                             df_summ_need_comment).get_result(df_limits, date_new_forecast, flag = True)
        #Генерация комментариев по изменениям Объемов
        smi = SMI_info(self.smi_file)
        smi_summ, channels_not_found_smi = smi.get_volumes_comments(delta_df = df_summ_need_comment, 
                                                channels_need_replace = channels_need_replace,
                                                year = 2025, 
                                                df_limits = df_limits, 
                                                smi_criteria = smi_criteria)
        if len(smi_summ) == 0:
            #Изменение названий каналов в соответствии с тем, что было изначально
            Federal_Comments.change_channels_name(channel_names_init, data_output_summ, 'Канал')
            #Форматирование столбца с Месяцем
            data_output_summ = Federal_Postprocessing(data_output_summ).replace_name_of_months('Месяц', year)
        
            #Определение даты старта и даты конца
            day_start, month_name_start, day_stop, month_name_stop = Federal_Processing.define_start_stop_day(data_cubik, start_date)
            if month_name_start == month_name_stop:
                data_output_summ['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}.'
            else:
                data_output_summ['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}.'

            problem_channels = {
                    'Channel not exist': channels_not_exist,
                    'Not enough reasons': channels_not_enough_reasons,
                    'SMI not': channels_not_found_smi
                }
            return data_output_summ, problem_channels
        
        else:
            smi_summ_ = smi_summ.copy()
            #Join комментариев со сравнением прогнозов и СМИ
            merged_df = pd.merge(data_output_summ, smi_summ_, on = ['Канал', 'Дата', 'Месяц'], how = 'left')
            merged_df['Комментарий'] = merged_df.apply(Federal_Comments.combine_columns, axis = 1)
            general_summ = merged_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
            #Изменение названий каналов в соответствии с тем, что было изначально
            Federal_Comments.change_channels_name(channel_names_init, general_summ, 'Канал')
            
            #Форматирование столбца с Месяцем
            general_summ = Federal_Postprocessing(general_summ).replace_name_of_months('Месяц', year)
            
            ##Определение даты старта и даты конца
            day_start, month_name_start, day_stop, month_name_stop = Federal_Processing.define_start_stop_day(data_cubik, start_date)
            if month_name_start == month_name_stop:
                general_summ['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}.'
            else:
                general_summ['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}.'

            problem_channels = {
                    'Channel not exist': channels_not_exist,
                    'Not enough reasons': channels_not_enough_reasons,
                    'SMI not': channels_not_found_smi
                }
            return general_summ, problem_channels
    

    @staticmethod
    def read_comments(comments_filepath_init: str):
        """
            Функция для чтения файла с ФУЛЛ-комментариями.
            Args:
                comments_filepath_init: Название файла с комментариями или полный путь до файла
            Returns:
                comments: DataFrame с ФУЛЛ-комментариями
        """
        #Чтение исходного файла с Комментариями
        comments = pd.read_excel(comments_filepath_init)
        comments['Изменение GRP'] = comments['Изменение GRP'].astype(int)
        comments['Порог'] = comments['Порог'].astype(int)
        comments.rename(columns = {'условие': 'Доп столбец'}, inplace = True) 
        comments['Дата'] = pd.to_datetime(comments['Дата'])
        return comments


    @staticmethod
    def comments_per_period(start_date: str, 
                            data, 
                            comments_filepath_init: str, 
                            criteria = 0.55):
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
        #Изменения начинаем смотреть с даты начала периода + 1 (если период 2 - 9 мая, то изменения начинаем смотреть с 3 мая.)
        start_date = pd.to_datetime(start_date, format = '%Y-%m-%d')
        start_date_modified = start_date + datetime.timedelta(days = 1)
        start_date_modified = start_date_modified.strftime('%Y-%m-%d')

        #Чтение исходного файла с Комментариями
        comments = pd.read_excel(comments_filepath_init)
        comments['Дата'] = pd.to_datetime(comments['Дата'])
        comments.sort_values(by = ['Дата'], inplace = True)
        comments.set_index('Дата', inplace = True)
        
        comments_cleaned = Federal_Preprocessing(comments).cut_data_cubik(start_date_modified)
        comments_cleaned['Изменение GRP'] = comments_cleaned['Изменение GRP'].astype(int)
        comments_cleaned['Порог'] = comments_cleaned['Порог'].astype(int)
        comments_cleaned.rename(columns = {'условие': 'Доп столбец'}, inplace = True) 
        comments_cleaned = comments_cleaned[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]

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
                    if len(delta) > 1:
                        for d in range(len(delta)):
                            if delta[d] * delta_grp_summ > 0:
                                delta_cleaned.append(delta[d])

                    elif len(delta) == 1:
                        delta_cleaned.append(delta)
                #Суммируем найденные изменения за период и проверяем, чтобы было объяснено более 2/3 недельного инвентаря
                if np.abs(np.sum(delta_cleaned)) < np.abs(delta_grp_summ * criteria):
                    data_need = data[((data['Канал'] == channel) & (data['Месяц'] == month))]
                    res.append(data_need)
            else:
                data_need = data[((data['Канал'] == channel) & (data['Месяц'] == month))]
                res.append(data_need)

        if len(res) == 0:
            return res
        else:
            result = pd.concat(res)
            return result
        
    
    @staticmethod
    def print_warning_comments(data, problem_channels, type_of_comments: str):
        """
            Функция для написания финальных ВОРНИНГОВ для накопленных изменений за период.
            С помощью данной функции можно понять, для каких каналов стоит дописать причины или наоборот написать причины САМОСТОЯТЕЛЬНО.
            Args:
                channels_not_exist: Словарь из каналов, для которых не нашлось релевантных причин для объяснения изменений. Ключ: Месяц, Значение: Канал.
                channels_not_enough_reasons: Словарь из каналов, которые нуждаются в дополнительных комментариях по изменению инвентаря. Ключ: Месяц, Значение: Канал.
                type_of_comments: by days или summ
            Returns:
                Комментарии-ворнинги
        """
        def support_func_per_comment(dict_1, dict_2, comment):
            """
                Вспомогательная функция для вывода комментариев в консоль.
            """
            for month, channel in list(dict_1.items()):
                    # Проверяем, существует ли ключ во втором словаре
                    if month in dict_2:
                        # Оставляем только те значения, которые есть во втором словаре
                        dict_1[month] = [value for value in dict_1[month] if value in dict_2[month]]
                    else:
                        # Если ключа нет во втором словаре, можем удалить его из dict1
                        del dict_1[month]

            #Написание комментария для каналов, для которых не нашлось причин и требуется написать комментарий самостоятельно
            if all(value in [] for value in dict_1.values()):
                print('')
            else:
                for month, channels in dict_1.items():
                    if channels == []:
                        print('')
                    else:
                        morph = pmrph.MorphAnalyzer()
                        month_ = morph.parse(month)[0].inflect({'loct'}).word.capitalize()
                        print(comment + f' для: {", ".join(list(channels))} в {month_}.' + color.END, end = '\n\n')


        #Вывод комментариев для изменений по дням
        if type_of_comments == 'by days':
            not_enough_reasons = []
            not_found = []
            not_found_smi = []
            for i in range(len(data)):
                channel = data.iloc[i]['Канал']
                limit = data.iloc[i]['Порог']
                comment = data.iloc[i]['Комментарий']
                if not pd.isnull(comment):
                    numbers = re.findall(r'\b\d+\b', comment)
                    numbers_int = []
                    for item in numbers:
                        numbers_int.append(int(item))
                    if np.abs(np.sum(numbers_int)) < limit:
                        not_enough_reasons.append(channel)
                else:
                    not_found.append(channel)

            if len(not_enough_reasons) != 0:
                print(color.BOLD + color.GREEN + f'Требуется дописать причины для {", ".join(set(not_enough_reasons))}.' + color.END, end = '\n\n')
            else:
                print('')

            if len(not_found) != 0:
                print(color.BOLD + color.RED + 'Требуется ' + \
                      color.UNDERLINE + 'САМОСТОЯТЕЛЬНО' + color.END + color.BOLD + color.RED + \
                      f' написать комментарий для: {", ".join(set(not_found))}.' + color.END, end = '\n\n')
            else:
                print('')

            if len(not_enough_reasons) == 0 and len(not_found) == 0:
                print('')

        #Вывод комментариев для накопленных изменений
        elif type_of_comments == 'summ':
            channels = list(data['Канал'])
            data_ = data.copy()

            months_init = list(data['Месяц'])
            months_new = []
            for old_month in months_init:
                month_new = str(old_month).split('\'')[0].title()
                months_new.append(month_new)
            data_['Месяц'] = data_['Месяц'].replace(months_init, months_new)

            #Группируем по месяцу и собираем каналы в списки
            months__and__channels = data_.groupby('Месяц')['Канал'].apply(list).to_dict()
#
            #################################### ДЛЯ НЕ НАЙДЕННЫХ КАНАЛОВ ####################################
            channels_not_exist = problem_channels['Channel not exist']
            channels_not_exist_new = Dict_Operations(channels_not_exist).rename_items_in_dict(channel_names_init)
            support_func_per_comment(channels_not_exist_new, 
                         months__and__channels, 
                         color.BOLD + color.RED + 'Требуется написать комментарий ' + \
                              color.UNDERLINE + 'САМОСТОЯТЕЛЬНО' + color.END + color.BOLD + color.RED)
            
            #################################### ДЛЯ НЕ НАЙДЕННЫХ КАНАЛОВ СМИ ################################
            channels_not_found_smi = problem_channels['SMI not']
            channels_not_exist_smi = Dict_Operations(channels_not_found_smi).rename_items_in_dict(channel_names_init)
            support_func_per_comment(channels_not_exist_smi, 
                         months__and__channels, 
                         color.BOLD + color.PURPLE + 'Не найдено данных от ' + color.UNDERLINE + 'СМИ' + color.END + ' ' + \
                              color.BOLD + color.PURPLE)
            
            #################################### ДЛЯ КАНАЛОВ, ДЛЯ КОТОРЫХ НАЙДЕНО НЕДОСТАТОЧНО ПРИЧИН ########
            channels_not_enough_reasons = problem_channels['Not enough reasons']
            channels_not_enough_reasons_new = Dict_Operations(channels_not_enough_reasons).rename_items_in_dict(channel_names_init)
            support_func_per_comment(channels_not_enough_reasons_new, 
                         months__and__channels, 
                         color.BOLD + color.GREEN + 'Найдено ' + color.UNDERLINE + 'НЕДОСТАТОЧНО' + \
                              color.END + color.BOLD + color.GREEN + ' причин')
        else:
            print(color.BOLD + color.RED + 'ПЕРЕМЕННАЯ type_of_comments ДОЛЖНА БЫТЬ ЛИБО by days, либо summ' + color.END)