import pandas as pd
import numpy as np
import re
import pymorphy3 as pmrph
from pathlib import Path
import datetime
from OMA_tools.io_data.operations import Table, Dict_Operations
from OMA_tools.federal.fed_comments.fed_preprocessing import Federal_Preprocessing
from OMA_tools.federal.fed_comments.fed_postprocessing import Federal_Postprocessing
from OMA_tools.federal.fed_comments.smi_info import SMI_info
from OMA_tools.federal.fed_comments.comments import Federal_Comments

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
    def __init__(self, limits_file, cubik_file, smi_file):
        self.limits_file = limits_file
        self.cubik_file = cubik_file
        self.smi_file = smi_file
        self.forecast_comparison_file = None
    
    
    @staticmethod
    def extract_month_name(month_str):
        """
            Вспомогательный метод для извлечения названия месяца
        """
        month_part = month_str.split("'")[0]  # Берем часть до апострофа
        return month_part.capitalize()  # Делаем первую букву заглавной

    @staticmethod
    def generate_month_order(start_year: int, months_count: int = 12):
        """
            Генерирует список месяцев в порядке следования в формате: "Январь'25", "Февраль'25", "Март'25" и тд.
            
            Parameters:
                start_year (int): начальный год
                months_count (int): количество месяцев для генерации
                language (str): язык названий месяцев ('ru' или 'en')
            
            Returns:
                list: список месяцев в формате "Месяц'ГГ"
        """
        
        months_ru = [
            'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
            'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'
        ]
        

        months = months_ru
        
        month_order = []
        current_year = start_year
        month_index = 0
        
        for i in range(months_count):
            month_name = months[month_index]
            short_year = str(current_year)[-2:]  # Последние 2 цифры года
            month_order.append(f"{month_name}'{short_year}")
            
            # Переходим к следующему месяцу
            month_index += 1
            if month_index >= 12:  # Если дошли до декабря
                month_index = 0
                current_year += 1
        
        return month_order
    

    @staticmethod
    def define_start_stop_day(df, start_date):
        morph = pmrph.MorphAnalyzer()  

        #Определение даты старта и даты конца
        start_date = pd.to_datetime(start_date)
        start = start_date.day
        month_name_start = str(start_date.strftime('%B'))
        parsed_word_start_month = morph.parse(month_name_start)[0]  # анализируем слово  
        result_1 = parsed_word_start_month.inflect({'gent'}).word
        
        stop_date = df.index[-1]
        stop = stop_date.day
        month_name_stop = str(stop_date.strftime('%B'))
        parsed_word_stop_month = morph.parse(month_name_stop)[0]  # анализируем слово  
        result_2 = parsed_word_stop_month.inflect({'gent'}).word
        return start, result_1, stop, result_2
    

    @staticmethod
    def define_limits(limits_file):
        """
            Функция для чтения файла с порогами
        """
        try:
            df_limits = pd.read_excel(limits_file)
            df_limits['Канал'] = df_limits['Канал'].str.upper()
            df_limits.set_index('Канал', inplace = True)
            df_limits = df_limits.T
            return df_limits
        except FileNotFoundError:
            print('Файл с Порогами не найден! Пожалуйста, вставьте его в соответствующую папку!')
        
    
    
    def get_data_per_analys(self, flag = True):
        """
            Вспомогательная функция для чтения файла с Порогами, Таблицы со Сравнением прогнозов и Федерального Кубика.
            Args:
                start_date: дата, начиная с которой начинаем смотреть изменения GRP.
                flag: Если True, то парсим файл со сравнением прогнозов.
        """
        if flag:
            try:
                #Считывание файла со сравнением прогнозов
                forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 2, sheet_name = 'Сводная')

                date_new_forecast = forecast_comparison.iloc[0]['Unnamed: 23']

                forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 5, sheet_name = 'Сводная')
                
                columns = ['Канал', 'Значения', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май',
                    'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь',
                    'Январь.1', 'Февраль.1', 'Март.1', 'Апрель.1',
                    'Май.1', 'Июнь.1', 'Июль.1', 'Август.1', 'Сентябрь.1', 'Октябрь.1',
                    'Ноябрь.1', 'Декабрь.1', 'Январь.2', 'Февраль.2', 'Март.2', 'Апрель.2', 'Май.2', 'Июнь.2', 'Июль.2',
                    'Август.2', 'Сентябрь.2', 'Октябрь.2', 'Ноябрь.2', 'Декабрь.2']
                forecast_comparison = forecast_comparison[columns]

                return forecast_comparison, date_new_forecast
            
            except FileNotFoundError:
                print('Файл со сравнением прогнозов не найден! Пожалуйста, вставьте его в соответствующую папку!')
        
        else:
            return None
    

    ### НОВАЯ ВЕРСИЯ МЕТОДА BY_DAYS ###
    def BY_DAYS(self, folder_path: str, start_date: str, smi_criteria):
        """
            Функция для генерация комментариев по дням.
            Args:
                folder_path: путь к папке, в которой хранятся файлы для анализа
                start_date: Дата, от которой начинаем смотреть изменения
                smi_criteria: критерий для отбора значений СМИ
            Returns:
                Обновленный файл с Комментариями
                smi_by_days: Комментарии с изменениями объемов по дням
                general_by_days: Комментарии с изменения по дням, исходя из таблицы со сравнением прогнозов, а также файла от СМИ
        """
        #Определение порогов
        df_limits = Federal_Processing.define_limits(self.limits_file)
        #Чтение данных из федерального кубика
        data_cubik = Federal_Preprocessing.read_cubik(filename = self.cubik_file)

        #Обрезание данных Федерального Кубика
        prepr = Federal_Preprocessing(data_cubik)
        need_data = prepr.cut_data_cubik(start_date)

        # Формирование словарей с изменениями 
        general_dict, by_dates_dict = Federal_Preprocessing.calculate_differencies(need_data, df_limits)

        # Создание словаря для хранения результатов по годам
        year_results = {}

        years = list(by_dates_dict.keys())

        # Фильтруем года - оставляем только те, у которых есть непустые датафреймы
        non_empty_years = [
            year for year in years 
            if isinstance(by_dates_dict.get(year), pd.DataFrame) 
            and not by_dates_dict[year].empty
        ]

        if len(non_empty_years) > 1:
            print(color.BOLD + color.RED + f'Найдено {len(non_empty_years)} непустых года для анализа.' + color.END)
            print('\n')

        elif len(non_empty_years) == 1:
            print(color.BOLD + color.RED + f'Найден {len(non_empty_years)} непустой год для анализа.' + color.END)
            print('\n')
        
        else:
            print(color.BOLD + color.RED + 'Не найдено данных для анализа!')
            print('\n')

        for year in years:
            print(color.BOLD + color.BLUE + f'Обрабатываем {year} год' + color.END)
            #try:

            # Генерация нужного порядка месяцев нужного формат, основываясь на годе
            month_order = Federal_Processing.generate_month_order(year)

            # Датафрейм, который анализируем
            df_by_dates_need_comment = by_dates_dict[year]

            # Поиск нужных файлов со Сравнением прогнозов и КУС, исходя из года и даты
            self.forecast_comparison_file = Federal_Preprocessing.find_data_file(folder_path, start_date, year, 'Сравнение прогнозов')
            kus_file = Federal_Preprocessing.find_data_file(folder_path, start_date, year, 'КУС')

            if self.forecast_comparison_file is None:
                print('\n')
                print(color.BOLD + f'😢 Файл со сравнением прогнозов для {year} года не найден! Пожалуйста, вставьте его в соответствующую папку!' +
                      f' Проверьте, чтобы дата, с которой хотим смотреть изменения, совпадала с первоначальной датой в файле со Сравнением прогнозов по {year} году.' + 
                      color.END)

            #kus_file = Federal_Preprocessing.find_data_file(folder_path, start_date, year, 'КУС')

            elif kus_file is None:
                print('\n')
                print(color.BOLD + f'😢 Файл с прогнозом КУСа для {year} года не найден! Пожалуйста, вставьте его в соответствующую папку!' +
                      ' Файл нужно добавлять даже, если не было корректировки прогноза!' + color.END)
            
            else:
                print(color.GREEN + f'✅ Файлы для {year} года найдены! Начинаю анализ. Пожалуйста, подождите ...' + color.END)

            problem_channels = {}

            # Чтение файла со сравнением прогнозов
            if self.forecast_comparison_file is not None:

                # Поиск последней даты планового обновления.
                forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 2, sheet_name = 'Сводная')
                date_new_forecast = forecast_comparison.iloc[0]['Unnamed: 23']

                forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 5, sheet_name = 'Сводная')
                
                columns = ['Канал', 'Значения', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май',
                    'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь',
                    'Январь.1', 'Февраль.1', 'Март.1', 'Апрель.1',
                    'Май.1', 'Июнь.1', 'Июль.1', 'Август.1', 'Сентябрь.1', 'Октябрь.1',
                    'Ноябрь.1', 'Декабрь.1', 'Январь.2', 'Февраль.2', 'Март.2', 'Апрель.2', 'Май.2', 'Июнь.2', 'Июль.2',
                    'Август.2', 'Сентябрь.2', 'Октябрь.2', 'Ноябрь.2', 'Декабрь.2']
                forecast_comparison = forecast_comparison[columns]

                ################# Генерация комментариев, исходя из файла со сравнением прогнозов #################
                fed_com = Federal_Comments(forecast_comparison, df_by_dates_need_comment)
                data_output_dates, channels_not_exist, channels_not_enough_reasons = fed_com.get_result(
                                                                    df_limits,  date_new_forecast, kus_file, 
                                                                    cummulative_diff_flag = False, flag = True
                                                                    )
                
                smi = SMI_info(self.smi_file)
                smi_by_days, channels_not_found_smi = smi.get_volumes_comments(
                                                            delta_df = df_by_dates_need_comment, 
                                                            channels_need_replace = channels_need_replace,
                                                            year = year, 
                                                            df_limits = df_limits, 
                                                            flag_by_days = True,
                                                            smi_criteria = smi_criteria)
                
                smi_by_days_ = smi_by_days.copy()
                if  len(data_output_dates) == 0 and len(smi_by_days) == 0:
                    print('НЕ НАЙДЕНО ДАННЫХ ДЛЯ УКАЗАННОГО ПЕРИОДА')
                
                #Если не нашлось релеватных данных от СМИ
                elif len(smi_by_days) == 0 and len(data_output_dates) != 0:
                    data_output_dates_ = data_output_dates[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
                    #Форматирование столбца с Месяцем
                    by_days = Federal_Postprocessing(data_output_dates_).replace_name_of_months('Месяц', str(year))
                    by_days_sorted = Table(by_days).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')
                    by_days_sorted_ = Federal_Postprocessing(df_by_dates_need_comment).clean_comments(by_days_sorted, date_new_forecast)
                    by_days_FINAL = Federal_Comments.change_channels_name(channel_names_init, by_days_sorted_, 'Канал')
                    problem_channels = {
                        'Channel not exist': channels_not_exist,
                        'Not enough reasons': channels_not_enough_reasons,
                        'SMI not': channels_not_found_smi
                    }

                    year_results[year] = [by_days_FINAL, problem_channels]
                
                # Если не нашлось релеватных данных из файла со сравнением прогнозов
                elif len(data_output_dates) == 0:
                    if len(smi_by_days_) != 0:

                        months_init = list(df_by_dates_need_comment['Месяц'])
                        months_new = []
                        for old_month in months_init:
                            month_new = str(old_month).split('\'')[0].title()
                            months_new.append(month_new)
                            
                        # Замена столбца дат на новый конвертированный столбец
                        df_by_dates_need_comment['Месяц'] = df_by_dates_need_comment['Месяц'].replace(months_init, months_new)

                        merged_df = pd.merge(df_by_dates_need_comment, smi_by_days_, on = ['Канал', 'Дата', 'Месяц'], how = 'left')

                        #Добавление пустого столбца для комментариев руководителя
                        merged_df['Доп столбец'] = ''

                        #Join комментариев с порогами
                        by_days = pd.merge(merged_df, df_limits.T, on = ['Канал'], how = 'inner')

                        by_days = by_days[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий', 'Дата осуществления']]

                        #Форматирование столбца с Месяцем
                        by_days_new = Federal_Postprocessing(by_days).replace_name_of_months('Месяц', str(year))
                        by_days_sorted = Table(by_days_new).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')
                        by_days_sorted_ = Federal_Postprocessing(df_by_dates_need_comment).clean_comments(by_days_sorted, date_new_forecast)
                        by_days_sort = by_days_sorted_[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
                        by_days_FINAL = Federal_Comments.change_channels_name(channel_names_init, by_days_sort, 'Канал')


                        problem_channels = {
                                    'Channel not exist': channels_not_exist,
                                    'Not enough reasons': channels_not_enough_reasons,
                                    'SMI not': channels_not_found_smi
                            }

                        year_results[year] = [by_days_FINAL, problem_channels]
                
                #Если нашлись релеватные данные от СМИ и нашлись объяснения из таблицы со сравнением прогнозов
                else:
                    #columns = ['Канал', 'Месяц']
                    #for col in columns:
                    #    data_output_dates[col] = data_output_dates[col].astype(str)
                    #    smi_by_days_[col] = smi_by_days_[col].astype(str)
                    data_output_dates['Дата'] = pd.to_datetime(data_output_dates['Дата'])
                    smi_by_days_['Дата'] = pd.to_datetime(smi_by_days_['Дата'])

                    merged_df = pd.merge(data_output_dates, smi_by_days_, on = ['Канал', 'Месяц', 'Дата'], how = 'left')

                    merged_df['Комментарий'] = merged_df.apply(Federal_Comments.combine_columns, axis = 1)

                    general_by_days = merged_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий', 'Дата осуществления']]

                    #Форматирование столбца с Месяцем
                    general_by_days = Federal_Postprocessing(general_by_days).replace_name_of_months('Месяц', str(year))
                    general_by_days_sorted = Table(general_by_days).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')
                    general_by_days_sorted_ = Federal_Postprocessing(df_by_dates_need_comment).clean_comments(general_by_days_sorted, date_new_forecast)
                    general_by_days_FINAL = Federal_Comments.change_channels_name(channel_names_init, general_by_days_sorted_, 'Канал')

                    problem_channels = {
                        'Channel not exist': channels_not_exist,
                        'Not enough reasons': channels_not_enough_reasons,
                        'SMI not': channels_not_found_smi
                    }
                    year_results[year] = [general_by_days_FINAL, problem_channels]
        

            # Если не найден релеватный файл со сравнением прогнозов
            else:
                data_output_dates = pd.DataFrame()

                print(f'Не найден файл со Сравнением прогнозов для {year}.')

                smi = SMI_info(self.smi_file)
                smi_by_days, channels_not_found_smi = smi.get_volumes_comments(
                                                            delta_df = df_by_dates_need_comment, 
                                                            channels_need_replace = channels_need_replace,
                                                            year = year, 
                                                            df_limits = df_limits, 
                                                            flag_by_days = True,
                                                            smi_criteria = smi_criteria)
                
                smi_by_days_ = smi_by_days.copy()
                if  len(data_output_dates) == 0 and len(smi_by_days) == 0:
                    print('В файле от СМИ также ничего не найдено.')

                    # Приводим таблицу к выходному виду
                    by_dates = df_by_dates_need_comment[['Канал', 'Месяц', 'Дата', 'Изменение GRP']]
                    
                    # Join результатов с порогами
                    df_res = pd.merge(by_dates, df_limits.T, on = ['Канал'], how = 'inner')

                    #Добавление пустого столбца для комментариев руководителя и столбца с Комментарием
                    df_res['Доп столбец'] = ''
                    df_res['Комментарий'] = ''
                    
                    result = df_res[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]

                    res = Federal_Comments.change_channels_name(channel_names_init, result, 'Канал')

                    # Применяем функцию
                    res['Месяц'] = res['Месяц'].apply(Federal_Processing.extract_month_name)
                    #Форматирование столбца с Месяцем
                    res_with_months = Federal_Postprocessing(res).replace_name_of_months('Месяц', str(year))
                    result_sorted = Table(res_with_months).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')

                    year_results[year] = [result_sorted, {}]

            print('\n')

        # Фильтруем пустые датафреймы
        print(color.BOLD + color.GREEN + f'Объединяю результаты. Пожалуйста, подождите ...' + color.END)
        non_empty_dfs = [year_results[year][0] for year in year_results 
                        if not year_results[year][0].empty]

        if non_empty_dfs:
            combined_df = pd.concat(non_empty_dfs, ignore_index = True)
        else:
            combined_df = pd.DataFrame()
        
        return combined_df.sort_values(['Месяц', 'Дата'], ascending = [True, True]).reset_index(drop = True)
    

    ### НОВАЯ ВЕРСИЯ МЕТОДА SUMM ###
    def SUMM(self, folder_path, start_date: str, smi_criteria):
        """
            Функция для генерация накопленных комментариев за период.
            Args:
                folder_path: путь к папке, в которой хранятся файлы для анализа
                start_date: Дата, от которой начинаем смотреть изменения
                smi_criteria: критерий для отбора значений СМИ
            Returns:
                Обновленный файл с Комментариями
                smi_by_days: Комментарии с изменениями объемов по дням
                general_by_days: Комментарии с изменения по дням, исходя из таблицы со сравнением прогнозов, а также файла от СМИ
        """
        #Определение порогов
        df_limits = Federal_Processing.define_limits(self.limits_file)
        #Чтение данных из федерального кубика
        data_cubik = Federal_Preprocessing.read_cubik(filename = self.cubik_file)

        #Обрезание данных Федерального Кубика
        prepr = Federal_Preprocessing(data_cubik)
        need_data = prepr.cut_data_cubik(start_date)

        # Формирование словарей с изменениями 
        general_dict, by_dates_dict = Federal_Preprocessing.calculate_differencies(need_data, df_limits)

        # Создание словаря для хранения результатов по годам
        year_results = {}

        years = list(by_dates_dict.keys())

        # Фильтруем года - оставляем только те, у которых есть непустые датафреймы
        non_empty_years = [
            year for year in years 
            if isinstance(by_dates_dict.get(year), pd.DataFrame) 
            and not by_dates_dict[year].empty
        ]

        if len(non_empty_years) > 1:
            print(color.BOLD + color.RED + f'Найдено {len(non_empty_years)} непустых года для анализа' + color.END)
            print('\n')

        elif len(non_empty_years) == 1:
            print(color.BOLD + color.RED + f'Найден {len(non_empty_years)} непустой год для анализа' + color.END)
            print('\n')
        
        else:
            print(color.BOLD + color.RED + 'Не найдено данных для анализа!')
            print('\n')

        for year in years:

            print(color.BOLD + color.BLUE + f'Обрабатываем {year} год' + color.END)
            # Генерация нужного порядка месяцев нужного формат, основываясь на годе
            month_order = Federal_Processing.generate_month_order(year)

            # Датафрейм, который анализируем
            general_df_by_dates = general_dict[year]

            #Выделение каналов и дат с накопленными изменениями из Федерального Кубика
            prepr = Federal_Preprocessing(data_cubik)
            df_summ_need_comment = prepr.calculate_accumulated_diff(general_df_by_dates, df_limits)

            # Поиск нужных файлов со Сравнением прогнозов и КУС, исходя из года и даты
            self.forecast_comparison_file = Federal_Preprocessing.find_data_file(folder_path, start_date, year, 'Сравнение прогнозов')
            kus_file = Federal_Preprocessing.find_data_file(folder_path, start_date, year, 'КУС')

            if self.forecast_comparison_file is None:
                print('\n')
                print(color.BOLD + f'😢 Файл со сравнением прогнозов для {year} года не найден! Пожалуйста, вставьте его в соответствующую папку!' +
                      f' Проверьте, чтобы дата, с которой хотим смотреть изменения, совпадала с первоначальной датой в файле со Сравнением прогнозов по {year} году.' + 
                      color.END)

            elif kus_file is None:
                print('\n')
                print(color.BOLD + f'😢 Файл с прогнозом КУСа для {year} года не найден! Пожалуйста, вставьте его в соответствующую папку!' +
                      ' Файл нужно добавлять даже, если не было корректировки прогноза!' + color.END)
            
            else:
                print(color.GREEN + f'✅ Файлы для {year} года найдены! Начинаю анализ. Пожалуйста, подождите ...' + color.END)

            problem_channels = {}

            #Генерация комментариев по изменениям Объемов
            smi = SMI_info(self.smi_file)
            smi_summ, channels_not_found_smi = smi.get_volumes_comments(delta_df = df_summ_need_comment, 
                                                    channels_need_replace = channels_need_replace,
                                                    year = year, 
                                                    df_limits = df_limits, 
                                                    flag_by_days = False,
                                                    smi_criteria = smi_criteria)

            # Чтение файла со сравнением прогнозов
            if self.forecast_comparison_file is not None:
                # Поиск последней даты планового обновления.
                forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 2, sheet_name = 'Сводная')
                date_new_forecast = forecast_comparison.iloc[0]['Unnamed: 23']

                forecast_comparison = pd.read_excel(self.forecast_comparison_file, skiprows = 5, sheet_name = 'Сводная')
                
                columns = ['Канал', 'Значения', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май',
                    'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь',
                    'Январь.1', 'Февраль.1', 'Март.1', 'Апрель.1',
                    'Май.1', 'Июнь.1', 'Июль.1', 'Август.1', 'Сентябрь.1', 'Октябрь.1',
                    'Ноябрь.1', 'Декабрь.1', 'Январь.2', 'Февраль.2', 'Март.2', 'Апрель.2', 'Май.2', 'Июнь.2', 'Июль.2',
                    'Август.2', 'Сентябрь.2', 'Октябрь.2', 'Ноябрь.2', 'Декабрь.2']
                forecast_comparison = forecast_comparison[columns]


                #Генерация первичных комментариев с накопленными изменениями, исходя из данных Фед Кубика и таблицы со сравнением прогнозов
                fed_com = Federal_Comments(forecast_comparison, df_summ_need_comment)
                data_output_summ, channels_not_exist, channels_not_enough_reasons = fed_com.get_result(
                                                                                                df_limits, date_new_forecast, kus_file, 
                                                                                                cummulative_diff_flag = True, flag = True
                                                                                                )
                
                #Если не нашлось релеватных данных от СМИ (учитываются только данные из таблицы со сравнением прогнозов)
                if len(smi_summ) == 0:
                    #Изменение названий каналов в соответствии с тем, что было изначально
                    Federal_Comments.change_channels_name(channel_names_init, data_output_summ, 'Канал')
                    #Форматирование столбца с Месяцем
                    data_output_summ = Federal_Postprocessing(data_output_summ).replace_name_of_months('Месяц', str(year))
                    data_output_summ_sorted = Table(data_output_summ).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')
                    #data_output_summ_sorted_ = Federal_Postprocessing(df_by_dates_need_comment).clean_comments(data_output_summ_sorted, date_new_forecast)
                
                    #Определение даты старта и даты конца
                    day_start, month_name_start, day_stop, month_name_stop = Federal_Processing.define_start_stop_day(data_cubik, start_date)
                    if month_name_start == month_name_stop:
                        data_output_summ_sorted['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}'
                    else:
                        data_output_summ_sorted['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}'
                    

                    problem_channels = {
                            'Channel not exist': channels_not_exist,
                            'Not enough reasons': channels_not_enough_reasons,
                            'SMI not': channels_not_found_smi
                        }
                    if len(data_output_summ_sorted) != 0:
                        year_results[year] = [data_output_summ_sorted, problem_channels]
                        #return data_output_summ_sorted, problem_channels
                    else:
                        print('Все изменения объяснены')

                        year_results[year] = [data_output_summ_sorted, problem_channels]
                        #return data_output_summ_sorted, problem_channels
                
                #Если нашлись релеватные данные от СМИ и из таблицы со сравнением прогнозов (merge этих двух составляющих)
                else:
                    smi_summ_ = smi_summ.copy()
                    # Join комментариев со сравнением прогнозов и СМИ
                    merged_df = pd.merge(data_output_summ, smi_summ_, on = ['Канал', 'Дата', 'Месяц'], how = 'left')
                    merged_df['Комментарий'] = merged_df.apply(Federal_Comments.combine_columns, axis = 1)
                    general_summ = merged_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
                    # Изменение названий каналов в соответствии с тем, что было изначально
                    Federal_Comments.change_channels_name(channel_names_init, general_summ, 'Канал')
                    
                    # Форматирование столбца с Месяцем
                    general_summ = Federal_Postprocessing(general_summ).replace_name_of_months('Месяц', str(year))
                    general_summ_sorted = Table(general_summ).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')
                    #general_summ_sorted_ = Federal_Postprocessing(df_by_dates_need_comment).clean_comments(general_summ_sorted, date_new_forecast)
                    
                    # Определение даты старта и даты конца
                    day_start, month_name_start, day_stop, month_name_stop = Federal_Processing.define_start_stop_day(data_cubik, start_date)
                    if month_name_start == month_name_stop:
                        general_summ_sorted['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}'
                    else:
                        general_summ_sorted['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}'

                    problem_channels = {
                            'Channel not exist': channels_not_exist,
                            'Not enough reasons': channels_not_enough_reasons,
                            'SMI not': channels_not_found_smi
                        }
                    if len(general_summ_sorted) != 0:
                        year_results[year] = [general_summ_sorted, problem_channels]
                        #return general_summ_sorted, problem_channels
                    else:
                        print('Все изменения объяснены')
                        year_results[year] = [general_summ_sorted, problem_channels]
                        #eturn general_summ_sorted, problem_channels

            # Не нашлись релевантные данные из таблицы со сравнением прогнозов
            else:
                data_output_summ = pd.DataFrame()
                print(f'Не найден файл со Сравнением прогнозов для {year}.')

                # Если не нашлись релеватные данные от СМИ
                if len(smi_summ) == 0:
                    print('В файле от СМИ также ничего не найдено.')
                    
                    # Приводим таблицу к выходному виду
                    summ = df_summ_need_comment[['Канал', 'Месяц', 'Дата', 'Изменение GRP']]
                    
                    # Join результатов с порогами
                    df_res = pd.merge(summ, df_limits.T, on = ['Канал'], how = 'inner')

                    #Добавление пустого столбца для комментариев руководителя и столбца с Комментарием
                    df_res['Доп столбец'] = ''
                    df_res['Комментарий'] = ''
                    
                    result = df_res[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]

                    res = Federal_Comments.change_channels_name(channel_names_init, result, 'Канал')

                    # Применяем функцию
                    res['Месяц'] = res['Месяц'].apply(Federal_Processing.extract_month_name)
                    #Форматирование столбца с Месяцем
                    res_with_months = Federal_Postprocessing(res).replace_name_of_months('Месяц', str(year))
                    result_sorted = Table(res_with_months).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')

                    # Определение даты старта и даты конца
                    day_start, month_name_start, day_stop, month_name_stop = Federal_Processing.define_start_stop_day(data_cubik, start_date)
                    if month_name_start == month_name_stop:
                        result_sorted['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}'
                    else:
                        result_sorted['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}'

                    year_results[year] = [result_sorted, {}]
                
                # Если нашлись релеватные данные от СМИ
                else:
                    smi_summ_ = smi_summ.copy()
                    # Join результатов с порогами
                    smi_res = pd.merge(smi_summ_, df_limits.T, on = ['Канал'], how = 'inner')

                    # Приводим таблицу к выходному виду
                    summ = smi_res[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог']]

                    res = Federal_Comments.change_channels_name(channel_names_init, summ, 'Канал')

                    #Форматирование столбца с Месяцем
                    res_with_months = Federal_Postprocessing(res).replace_name_of_months('Месяц', str(year))
                    result_sorted = Table(res_with_months).sort_in_specific_way(month_order, 'Месяц', date_column = 'Дата')

                    # Определение даты старта и даты конца
                    day_start, month_name_start, day_stop, month_name_stop = Federal_Processing.define_start_stop_day(data_cubik, start_date)
                    if month_name_start == month_name_stop:
                        result_sorted['Доп столбец'] = f'Общее изменение с {day_start} по {day_stop} {month_name_stop}'
                    else:
                        result_sorted['Доп столбец'] = f'Общее изменение с {day_start} {month_name_start} по {day_stop} {month_name_stop}'

                    problem_channels = {
                            'Channel not exist': channels_not_exist,
                            'Not enough reasons': channels_not_enough_reasons,
                            'SMI not': channels_not_found_smi
                        }
                    
                    if len(result_sorted) != 0:
                        year_results[year] = [result_sorted, problem_channels]
                        
                    else:
                        print('Все изменения объяснены')
                        year_results[year] = [result_sorted, problem_channels]
                        
            print('\n')

        # Фильтруем пустые датафреймы
        print(color.BOLD + color.GREEN + f'Объединяю результаты. Пожалуйста, подождите ...' + color.END)
        non_empty_dfs = [year_results[year][0] for year in year_results 
                        if not year_results[year][0].empty]

        if non_empty_dfs:
            combined_df = pd.concat(non_empty_dfs, ignore_index = True)
        else:
            combined_df = pd.DataFrame()
        return combined_df
                


    @staticmethod
    def read_comments(comments_filepath_init: str):
        """
            Функция для чтения файла с ФУЛЛ-комментариями.
            Args:
                comments_filepath_init: Название файла с комментариями или полный путь до файла
            Returns:
                comments: DataFrame с ФУЛЛ-комментариями
        """
        try:
            #Чтение исходного файла с Комментариями
            comments = pd.read_excel(comments_filepath_init)
            comments['Изменение GRP'] = comments['Изменение GRP'].astype(int)
            comments['Порог'] = comments['Порог'].astype(int)
            comments.rename(columns = {'условие': 'Доп столбец'}, inplace = True) 
            comments['Дата'] = pd.to_datetime(comments['Дата'])
            return comments
        except FileNotFoundError:
            print('Файл с Комментариями не найден. Пожалуйста, вставьте его в соответствующую папку!')


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
        #Вспомогательная функция для объединения 
        def combine_change(row):
            return f"{row['Доп столбец']} {row['Изменение GRP']} GRP."
        
        #Изменения начинаем смотреть с даты начала периода + 1 (если период 2 - 9 мая, то изменения начинаем смотреть с 3 мая.)
        start_date = pd.to_datetime(start_date, format = '%Y-%m-%d')
        start_date_modified = start_date + datetime.timedelta(days = 1)
        #start_date_modified = start_date_modified.strftime('%Y-%m-%d')

        #Чтение исходного файла с Комментариями
        try:
            comments = pd.read_excel(comments_filepath_init)
            comments['Дата'] = pd.to_datetime(comments['Дата'], dayfirst = True, errors = 'coerce')
            comments.sort_values(by = ['Дата'], inplace = True)
            comments.set_index('Дата', inplace = True)
        except FileExistsError:
            print('Файл с Комментариями не найден. Пожалуйста, вставьте его в соответствующую папку!')

        date_of_start = start_date_modified
        #Если в файле с Комментариями нет подходящей даты для начала отсчета изменений.
        while date_of_start not in comments.index:
            print(f'Текущая дата: {date_of_start} не совпадает с целевой.')
            print('Ищем дальше ...')
            date_of_start += datetime.timedelta(days = 1)
            print(f'Найдена следующая подходящая дата: {date_of_start}')
        
        start_date_modified = date_of_start.strftime('%Y-%m-%d')
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
            # Применяем функцию к DataFrame и создаем новую колонку
            result['Объединение'] = result.apply(combine_change, axis = 1)

            # Оставляем столбец "Изменение GRP" пустым
            result['Изменение GRP'] = ''

            # Выводим результат
            res_updated = result[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Объединение', 'Комментарий']]
            res_updated.rename(columns = {'Объединение': 'Доп столбец'}, inplace = True)
            res_updated = res_updated.reset_index(drop = True)

            #Удаляем дублирующиеся комментарии за период. Оставляем нужные.
            for i in range(len(res_updated)):
                channel = res_updated.iloc[i]['Канал']
                month = res_updated.iloc[i]['Месяц']
                comments_per_week = res_updated.iloc[i]['Комментарий']
                if not pd.isna(comments_per_week):
                    comment_per_week_splitted = comments_per_week.split('. ')

                    filtered_df = comments_cleaned[((comments_cleaned['Канал'] == channel) & (comments_cleaned['Месяц'] == month))]
                    if len(filtered_df) != 0:
                        comments = list(filtered_df['Комментарий'])

                        def check_comments():
                            return all(map(lambda x: x is not None if isinstance(x, str) else not np.isnan(x), comments))
                                    
                        if check_comments():
                            list_of_comments = [item.split('. ') for item in comments]

                            #Получаем вложенный список
                            nested_list = list_of_comments[0]
                        
                            #Удаляем элементы из первого списка, если они содержатся во втором
                            result_list = [item for item in comment_per_week_splitted if item not in nested_list]
                            
                            res_updated.at[i, 'Комментарий'] = '. '.join(result_list)
                    else:
                        continue
                else:
                    res_updated.at[i, 'Комментарий'] = ''
            return res_updated
        
    
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
            if len(channels_not_exist) > 1:
                channels_not_exist_new = Dict_Operations(channels_not_exist).rename_items_in_dict(channel_names_init)
                support_func_per_comment(channels_not_exist_new, 
                            months__and__channels, 
                            color.BOLD + color.RED + 'Требуется написать комментарий ' + \
                                color.UNDERLINE + 'САМОСТОЯТЕЛЬНО' + color.END + color.BOLD + color.RED)
            else:
                print('')
            
            #################################### ДЛЯ НЕ НАЙДЕННЫХ КАНАЛОВ СМИ ################################
            channels_not_found_smi = problem_channels['SMI not']
            if len(channels_not_found_smi) > 1:
                channels_not_exist_smi = Dict_Operations(channels_not_found_smi).rename_items_in_dict(channel_names_init)
                support_func_per_comment(channels_not_exist_smi, 
                            months__and__channels, 
                            color.BOLD + color.PURPLE + 'Не найдено данных от ' + color.UNDERLINE + 'СМИ' + color.END + ' ' + \
                                color.BOLD + color.PURPLE)
            else:
                print('')
            
            #################################### ДЛЯ КАНАЛОВ, ДЛЯ КОТОРЫХ НАЙДЕНО НЕДОСТАТОЧНО ПРИЧИН ########
            channels_not_enough_reasons = problem_channels['Not enough reasons']
            if len(channels_not_enough_reasons) > 1:
                channels_not_enough_reasons_new = Dict_Operations(channels_not_enough_reasons).rename_items_in_dict(channel_names_init)
                support_func_per_comment(channels_not_enough_reasons_new, 
                            months__and__channels, 
                            color.BOLD + color.GREEN + 'Найдено ' + color.UNDERLINE + 'НЕДОСТАТОЧНО' + \
                                color.END + color.BOLD + color.GREEN + ' причин')
            else:
                print('')
        else:
            print(color.BOLD + color.RED + 'ПЕРЕМЕННАЯ type_of_comments ДОЛЖНА БЫТЬ ЛИБО by days, либо summ' + color.END)