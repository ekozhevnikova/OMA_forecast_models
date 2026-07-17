import pandas as pd
import numpy as np

from OMA_tools.io_data.operations import *
from OMA_tools.io_data.colors import *

from OMA_tools.thematic.share import *

import warnings
warnings.filterwarnings('ignore')



class DataUpdater:
    """
        Класс для обновления файлов с фактическими данными по Тематическому ТВ
    """
    def __init__(self, guide_book: pd.DataFrame, current_year: int):
        """
            Атрибуты:
            ----------
                guide_book: pd.DataFrame
                    Таблица-справочник с перечнем все каналом и их описанием
                current_year: int
                    Номер текущего года, по которому хотим обновить данные в истории
        """

        self.guide_book = guide_book
        self.current_year = current_year
        
        self.MONTHS = {
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
        }

        self.VK_DICT = {
            'ЕРК': 'все 25-49',
            'ЖРК': 'ж 25-49',
            'МРК': 'м 25-49',
            'ДРК': 'все 4-40',
        }


    def parse_SQL_file(self, sql_file_path: str):
        """
            Метод для парсинга ежедневной выгрузки со странным названием, которая приходит с адреса NVDBMZ\MZ SQL server <vimb2-sql-svc-mz@vitpc.com>.
            !!! ВАЖНО !!! Очень важно брать последнюю выгрузку для закрытого месяца, чтобы была возможность закрывать прошлый месяц.

            Параметры:
            ----------
                sql_file_path: str
                    Путь к файлу с выгрузкой SQL в формате .csv
            
            Returns:
            ----------
                result_sorted: pd.DataFrame
                    Причёсанная SQL выгрузка в виде pd.DataFrame
        """
        sql_data = pd.read_csv(
            sql_file_path,
            encoding = 'utf-16', names = ['raw'],
            sep = '\t',  # разделитель столбцов - табуляция
            decimal = ','  # десятичный разделитель - запятая
        )

        # Удаляем заголовок
        sql_data = sql_data.iloc[1:].reset_index(drop = True)

        # Разделяем по первому TAB (самый левый разделитель)
        split_data_left = sql_data['raw'].str.split(' ', n = 1, expand = True)
        split_data_left.columns = ['ID канала', 'rest']  # переименовываем

        # Разделяем по первому TAB (самый левый разделитель)
        TVR = split_data_left['rest'].str.rsplit(' ', n = 1, expand = True)
        TVR.columns = ['rest', 'Ср.рейтинг эфира']

        tvr = TVR['rest'].str.rsplit(' ', n = 1, expand = True)
        tvr.columns = ['rest', 'Ср.рейтинг рекламы']

        hour = tvr['rest'].str.rsplit(' ', n = 1, expand = True)
        hour.columns = ['rest', 'Час']

        month = hour['rest'].str.rsplit(' ', n = 1, expand = True)
        month.columns = ['rest', 'Месяц']

        year = month['rest'].str.rsplit(' ', n = 1, expand = True)
        year.columns = ['rest', 'Год']

        id_bca = year['rest'].str.rsplit(' ', n = 1, expand = True)
        id_bca.columns = ['Канал', 'ID ЦА']


        # Шаг 3: ОБЪЕДИНЯЕМ все столбцы
        final_result = pd.DataFrame({
            'ID канала': split_data_left['ID канала'],
            'Канал': id_bca['Канал'],
            'ID ЦА': id_bca['ID ЦА'],
            'Год': year['Год'],
            'Месяц': month['Месяц'],
            'Час': hour['Час'],
            'tvr рекламный': tvr['Ср.рейтинг рекламы'],
            'TVR эфира': TVR['Ср.рейтинг эфира']
        })

        result_filtered = final_result[ final_result['Час'] == 'NULL' ].reset_index(drop = True)

        int_columns = ['ID канала', 'ID ЦА', 'Год', 'Месяц']
        for int_column in int_columns:
            result_filtered[int_column] = result_filtered[int_column].astype(int)
            
        float_columns = ['tvr рекламный', 'TVR эфира']
        for float_column in float_columns:
            result_filtered[float_column] = result_filtered[float_column].str.replace(',', '.')
            result_filtered[float_column] = result_filtered[float_column].astype(float)

        # Создаем столбец с пустыми значениями
        result_filtered['ЦА'] = ''

        result_filtered.loc[result_filtered['ID ЦА'] == 158, 'ЦА'] = 'все 25-49'
        result_filtered.loc[result_filtered['ID ЦА'] == 856, 'ЦА'] = 'ж 25-49'
        result_filtered.loc[result_filtered['ID ЦА'] == 857, 'ЦА'] = 'м 25-49'
        result_filtered.loc[result_filtered['ID ЦА'] == 571, 'ЦА'] = 'все 4-40'

        result_filtered = result_filtered.drop(['Час'], axis = 1)
        result_sorted = result_filtered.sort_values(by = 'Канал')
        return result_sorted.reset_index(drop = True)
    

    def tvr_last_fact_month(self, sql_df: pd.DataFrame):
        """
            Метод для формирования фактических значений tvr по последнему месяцу
        
            Параметры:
            ----------
                sql_df: pd.DataFrame
                    Таблица с выгрузкой SQL
            
            Returns:
            ----------
                new_month_data: dict
                    Словарь с данными по ново закрывшемуся месяцу: 
                        key - название ВК
                        value - pd.DataFrame
        """
        new_month_data = {}
        for VK, bca in self.VK_DICT.items():
            # ШАГ 1. Отбираем каналы ВК для анализа из Справочника
            target_guide = self.guide_book[self.guide_book[VK] == 1].reset_index(drop = True)
            target_guide = target_guide[['Канал', VK]]
            
            # ШАГ 2.  Отбираем ЦА для анализа из SQL выгрузки
            target_sql = sql_df[sql_df['ЦА'] == bca].reset_index(drop = True)

            # ШАГ 3. Join с каналами из справочника
            merged_sql = pd.merge(target_sql, target_guide, on = 'Канал', how = 'inner')

            # Задаем название месяца, который будем закрывать
            self.month_close = f"{self.MONTHS[merged_sql.iloc[0]['Месяц']]} {self.current_year}"
            merged_sql.rename(columns = {'tvr рекламный': self.month_close}, inplace = True)

            merged_sql_filtered = merged_sql[['Канал', self.month_close]]
            merged_sql_sorted = merged_sql_filtered.sort_values(by = 'Канал')
            
            new_month_data[VK] = merged_sql_sorted.reset_index(drop = True)
        
        return new_month_data
    

    def update_historical_data(self, new_data_dict: dict,  history_full: str, history_curr_year: str):
        """
            Метод для обновления файлов с фактическими данными по всей истории, а также истории по текущему году.

            Параметры:
            ----------
                new_data_dict: dict
                    Словарь с новыми данными, где ключ - название ВК, значение - датафрейм с данными.
                history_full: str
                    Путь к файлу с историей, начиная с 2021г.
                history_curr_year: str
                    Путь к файлу с историей по текущему году.
            
            Returns:
            ----------

        """
        # Проверяем, существует ли файл с историей, начиная с 2021 г
        if not os.path.exists(history_full):
            print(f"📁 Файл {history_full} не найден. Будет создан новый файл с текущими данными")
            return new_data_dict
        
        # Проверяем, существует ли файл с историей, начиная с 2021 г
        if not os.path.exists(history_curr_year):
            print(f"📁 Файл {history_curr_year} не найден. Будет создан новый файл с текущими данными")
            return new_data_dict
            
        data_dict = File(history_curr_year).from_file(0)
        hist_curr_year_dict = Dict_Operations(data_dict).replace_keys_in_dict(['ЕРК', 'ЖРК', 'МРК', 'ДРК'])

        full_data_dict = File(history_full).from_file(0)
        full_hist_dict = Dict_Operations(full_data_dict).replace_keys_in_dict(['ЕРК', 'ЖРК', 'МРК', 'ДРК'])

        updated_curr_year_hist = {}
        updated_full_hist = {}

        for vk, curr_year_hist_df in hist_curr_year_dict.items():
            full_hist_df = full_hist_dict[vk]
            # Отбираем свежие данные по конкретному ВК
            table = new_data_dict[vk]

            new_channels = set(table['Канал']) - set(curr_year_hist_df['Канал'])  # Новые каналы (есть в справочнике, нет в истории)

            # Если добавились новые каналы
            if len(new_channels) > 0:
                print(Color.BOLD + Color.GREEN + "Найдены следующие новые каналы. Заполню нулями историю для этих каналов." + Color.END)
                for ch in new_channels:
                    print(f"  - {ch}")
                    # Создаем новую строку для каждого нового канала.
                    # Для всех месяцев в факте заполняем ее нулями
                    full_hist_df[ch] = [0.0] * (full_hist_df.shape[1] - 1)
                    curr_year_hist_df[ch] = [0.0] * (curr_year_hist_df.shape[1] - 1)

            # ОБНОВЛЕНИЕ ТЕКУЩЕГО ГОДА
            curr_year_updated = pd.merge(curr_year_hist_df, table, on = 'Канал', how = 'left')
            curr_year_updated_sorted = curr_year_updated.sort_values(by = 'Канал')
            curr_year_updated_sorted = curr_year_updated_sorted.fillna(0)
            updated_curr_year_hist[vk] = curr_year_updated_sorted


            # ОБНОВЛЕНИЕ ИСТОРИИ, НАЧИНАЯ С 2021г
            full_hist_updated = pd.merge(full_hist_df, table, on = 'Канал', how = 'left')
            # Переставляем колонки, чтобы новый столбец был слева
            full_hist_updated_sorted = full_hist_updated.sort_values(by = 'Канал')
            full_hist_updated_sorted = full_hist_updated_sorted.fillna(0)
            updated_full_hist[vk] = full_hist_updated_sorted

        return updated_curr_year_hist, updated_full_hist
    

    def pipeline_updater(self, sql_file_path: str, history_full_path: str, history_curr_year_path: str):
        """
            Пайплайн для обновления файлов с историческими данными
        """
        # ШАГ 1. Парсинг файла с выгрузкой SQL
        sql_df = self.parse_SQL_file(sql_file_path)

        # ШАГ 2. Получение данных по последнему закрывшемуся месяцу из выгрузки SQL
        new_tvr_dict = self.tvr_last_fact_month(sql_df)

        # ШАГ 3. Обновление таблиц с историческими данными
        curr_year_dict_updated, full_hist_dict_updated = self.update_historical_data(new_tvr_dict, history_full_path, history_curr_year_path)

        # ШАГ 4. Сохранение данных обратно в файл
        ThematicShare.save_with_xlsxwriter(curr_year_dict_updated, history_curr_year_path)
        ThematicShare.save_with_xlsxwriter(full_hist_dict_updated, history_full_path)

        return new_tvr_dict



