import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import pymorphy3 as pmrph
import time
import threading
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
import copy
from typing import Union, List, Dict, Any

from functools import lru_cache

from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.io_data.dates import Dates_Operations
from OMA_tools.regions.data_extraction.task_builder import *
from OMA_tools.io_data.colors import *

import warnings
warnings.filterwarnings('ignore')


class DataConfig:
    """
        Конфигурационные параметры
    """
    STATISTICS = ['Share']
    COMMON_TIME_FILTER = 'timeBand1 >= 50000 AND timeBand1 < 290000'
    COMMON_OPTIONS = {
        "kitId": 3,
        "totalType": "TotalChannels"
    }
    DEFAULT_SORTINGS = {'tvCompanyName': 'ASC'}

    LOCATION_FILTER = None #Если None, то Дом и Дача
    WEEKDAY_FILTER = None #Задаем дни недели
    DAYTYPE_FILTER = None #Задаем тип дня
    TARGETDEMO_FILTER = None #Дополнительный фильтр на ЦА для расчета Affinity
    ADD_CITY_TO_BASEDEMO_FROM_REGION = True
    ADD_CITY_TO_TARGETDEMO_FROM_REGION = True
    # Список БЦА
    BCA_LIST = ['All 18+', 'All 14-59', 'All 10-45', 'All 14-44', 
                'All 14-54', 'All 25-49', 'All 25-54', 'All 4-45', 
                'All 6-54', 'W 14-44', 'W 25-59']
    
    # Конфигурация специальных каналов
    RUSSIA_1_CITIES = [
        'БАРНАУЛ', 'ВЛАДИВОСТОК', 'ВОЛГОГРАД', 'ВОРОНЕЖ', 
        'ЕКАТЕРИНБУРГ', 'ИРКУТСК', 'КАЗАНЬ', 'КЕМЕРОВО', 
        'КРАСНОДАР', 'КРАСНОЯРСК', 'НИЖНИЙ НОВГОРОД', 'НОВОСИБИРСК', 
        'ОМСК', 'ПЕРМЬ', 'РОСТОВ-НА-ДОНУ', 'САМАРА', 
        'САНКТ-ПЕТЕРБУРГ', 'САРАТОВ', 'ТЮМЕНЬ', 'УФА', 
        'ХАБАРОВСК', 'ЧЕЛЯБИНСК', 'ЯРОСЛАВЛЬ', 'СТАВРОПОЛЬ', 
        'ТВЕРЬ', 'ТОМСК'
                      ]
    
    # Список колонок России 1
    RUSSIA_COLUMNS = [
        'Date', 'РОССИЯ 1 (БАРНАУЛ)', 'РОССИЯ 1 (ВЛАДИВОСТОК)', 'РОССИЯ 1 (ВОЛГОГРАД)',  'РОССИЯ 1 (ВОРОНЕЖ)',
        'РОССИЯ 1 (ЕКАТЕРИНБУРГ)',  'РОССИЯ 1 (ИРКУТСК)',  'РОССИЯ 1 (КАЗАНЬ)',  'РОССИЯ 1 (КЕМЕРОВО)',
        'РОССИЯ 1 (КРАСНОДАР)',  'РОССИЯ 1 (КРАСНОЯРСК)',  'РОССИЯ 1 (НИЖНИЙ НОВГОРОД)',  'РОССИЯ 1 (НОВОСИБИРСК)',
        'РОССИЯ 1 (ОМСК)',  'РОССИЯ 1 (ПЕРМЬ)',  'РОССИЯ 1 (РОСТОВ-НА-ДОНУ)',  'РОССИЯ 1 (САМАРА)',
        'РОССИЯ 1 (САНКТ-ПЕТЕРБУРГ)',  'РОССИЯ 1 (САРАТОВ)',  'РОССИЯ 1 (ТЮМЕНЬ)',  'РОССИЯ 1 (УФА)',
        'РОССИЯ 1 (ХАБАРОВСК)',  'РОССИЯ 1 (ЧЕЛЯБИНСК)',  'РОССИЯ 1 (ЯРОСЛАВЛЬ)',  'РОССИЯ 1 (СТАВРОПОЛЬ)',
        'РОССИЯ 1 (ТВЕРЬ)', 'РОССИЯ 1 (ТОМСК)'
    ]
    
    # Список колонок ГТРК
    GTRK_COLUMNS = [
        'Date', 'ГТРК БАРНАУЛ   ВСЕ 18+', 'ГТРК ВЛАДИВОСТОК   ВСЕ 18+',
        'ГТРК ВОЛГОГРАД   ВСЕ 18+', 'ГТРК ВОРОНЕЖ   ВСЕ 18+', 'ГТРК ЕКАТЕРИНБУРГ   ВСЕ 18+',
        'ГТРК ИРКУТСК   ВСЕ 18+', 'ГТРК КАЗАНЬ   ВСЕ 18+', 'ГТРК КЕМЕРОВО   ВСЕ 18+',
        'ГТРК КРАСНОДАР   ВСЕ 18+', 'ГТРК КРАСНОЯРСК   ВСЕ 18+', 'ГТРК НИЖНИЙ НОВГОРОД   ВСЕ 18+',
        'ГТРК НОВОСИБИРСК   ВСЕ 18+', 'ГТРК ОМСК   ВСЕ 18+', 'ГТРК ПЕРМЬ   ВСЕ 18+',
        'ГТРК РОСТОВ-НА-ДОНУ   ВСЕ 18+', 'ГТРК САМАРА   ВСЕ 18+', 'ГТРК САНКТ-ПЕТЕРБУРГ   ВСЕ 18+',
        'ГТРК САРАТОВ   ВСЕ 18+', 'ГТРК ТЮМЕНЬ   ВСЕ 18+', 'ГТРК УФА   ВСЕ 18+',
        'ГТРК ХАБАРОВСК   ВСЕ 18+', 'ГТРК ЧЕЛЯБИНСК  ВСЕ 18+', 'ГТРК ЯРОСЛАВЛЬ   ВСЕ 18+',
        'ГТРК СТАВРОПОЛЬ   ВСЕ 18+', 'ГТРК ТВЕРЬ   ВСЕ 18+', 'ГТРК ТОМСК   ВСЕ 18+'
                ]



class LeaderShipDataExtractor:
    """
        Класс для выгрузки данных для руководителей групп, а также для куратора регионов
    """
    @staticmethod
    def preprocess_constant_files(regions_id_path, table_sample_path, sheet_name_1 = 'for_comments', sheet_name_2 = 'for_team_lead'):
        """
            Функция для обработки файлов, в которых прописан порядок каналов, а также их ID
        """
        data = File(regions_id_path).from_file(skiprows = 0)
        data_ = Dict_Operations(data).replace_keys_in_dict(['All 18+', 'All 14-59', 'All 10-45', 'All 14-44', 'All 14-54', 
                                                            'All 25-49', 'All 25-54', 'All 4-45', 'All 6-54', 'W 14-44', 'W 25-59'])
        dict_data = Dict_Operations.generate_list_or_dict_from_df(dict_with_df = data_, col_name = 'ID', cond = 'dict')
        #Чтение заготовки для составления таблицы для генерации комментариев
        share_table = pd.read_excel(table_sample_path, sheet_name_1)
        #Чтение заготовки для составления таблицы для руководителей групп
        team_lead_table = pd.read_excel(table_sample_path, sheet_name_2)
        #Приведение к верхнему регистру значений в колонках
        columns = ['Channel', 'City', 'BCA']
        for i in columns:
            share_table[i] = share_table[i].str.upper()
        share_table['tvCompanyName'] = share_table['Channel'] + ' (' + share_table['City'] + ')'
        return dict_data, share_table, team_lead_table
    
    @staticmethod
    def get_output(date_filter, data, share_table, statistic = 'Доля'):
        """
            Вспомогательная функция для придания внешнего вида выходной таблицы
        """
        df_new = Table(data).delete_rows_with_substring(col_name = 'tvCompanyName', substring = ' / ДОП ')
        df_new = df_new.rename(columns = {'regionName': 'City'})
        split_df = Table(df_new).split_column(split = '   ', col_name = 'prj_name', col_name_new = ['City', 'BCA'])
        split_df['prj_name'] = split_df['City'] + '   ' + split_df['BCA']
        data_df_fact_month = pd.merge(split_df, df_new, on = ['City', 'prj_name']).drop_duplicates()
        data_output = pd.merge(share_table, data_df_fact_month, on = ['tvCompanyName', 'City', 'BCA'], how = 'left')
        output = data_output.drop(columns = ['prj_name', 'tvCompanyName'])
        output['Share'] = output['Share'].round(5)
        output_ = output.rename(columns = {
            'Channel': 'Телеканал', 
            'City': 'Город', 
            'BCA': 'БЦА', 
            'Share': Dates_Operations(date_filter).generate_date_interval(statistic)})
        return output_
    
    
    @staticmethod
    def sum_2_channels(df1, df2, date_column):
        """
            Args:
                df1: Локальный канал
                df2: Отфильтрованный канал из основной массы
            Return:
        """
        output_df1 = df1.rename(columns = {'regionName': 'Город'})[['Город', 'tvCompanyName', 'Share']]
        data = pd.merge(df2, output_df1, on = 'Город', how = 'left')[['Телеканал', 'Город', 'БЦА', date_column, 'Share']]
        data['SHARE'] = data[[date_column, 'Share']].sum(axis = 1)
        data_updated = data[['Телеканал', 'Город', 'БЦА', 'SHARE']]
        Data = data_updated.rename(columns = {'SHARE': date_column})
        return Data

    
    @staticmethod
    def make_output(df_1, df_2, full, share_table):
        broken_channels = pd.concat([df_1, df_2])
        data_updated = pd.concat([full, broken_channels])
        share_ = share_table.rename(columns  = {'Channel': 'Телеканал', 'City': 'Город', 'BCA': 'БЦА'})
        example_share = share_[list(share_.columns)[:-1]]
        output_data = pd.merge(example_share, data_updated, on = ['Телеканал', 'Город', 'БЦА'], how = 'left')
        return output_data
    

    @staticmethod
    def build_json_tasks(
        date_filter, regions_params, slices, sortings,
        time_filter = DataConfig.COMMON_TIME_FILTER,
        statistics = DataConfig.STATISTICS,
        options = DataConfig.COMMON_OPTIONS,
        location_filter = DataConfig.LOCATION_FILTER,
        weekday_filter = DataConfig.WEEKDAY_FILTER,
        daytype_filter = DataConfig.DAYTYPE_FILTER,
        targetdemo_filter = DataConfig.TARGETDEMO_FILTER,
        add_city_to_basedemo_from_region = DataConfig.ADD_CITY_TO_BASEDEMO_FROM_REGION,
        add_city_to_targetdemo_from_region = DataConfig.ADD_CITY_TO_TARGETDEMO_FROM_REGION
    ):
        import time

        start = time.perf_counter()

        print(Color.BOLD + Color.BLUE + '=== 🕑 ФОРМИРУЮ ЗАДАЧИ В ФОРМАТЕ JSON ДЛЯ ОТПРАВКИ НА СЕРВЕР ===' + Color.END)
        print('Пожалуйста, подождите, этот процесс обычно занимает около  10 мин.')

        json_tasks = {}
        
        for group_name, config in regions_params.items():

            group_tasks = {}

            for i, company in enumerate(config['companies']):
                task = BaseDataService._build_timeband_common_params(
                    date_filter = date_filter,
                    company_filter = company,
                    basedemo_filter = config['basedemos'][i],
                    regions_id = config['regions'][i],
                    targets = config['targets'][i],  # Берем targets из конфига
                    time_filter = time_filter,
                    statistics = statistics,
                    slices = slices,
                    sortings = sortings,
                    options = options,
                    location_filter = location_filter,
                    weekday_filter = weekday_filter,
                    daytype_filter = daytype_filter,
                    targetdemo_filter = targetdemo_filter,
                    add_city_to_basedemo_from_region = add_city_to_basedemo_from_region,
                    add_city_to_targetdemo_from_region = add_city_to_targetdemo_from_region
                )
                
                if config['names'] and i < len(config['names']):
                    key = config['names'][i]
                elif config['key_template']:
                    key = config['key_template'].format(i)
                else:
                    key = f"{group_name}_{i}"
                    
                # Добавляем задачу в группу
                group_tasks[key] = task

            # Добавляем всю группу в основной результат
            json_tasks[group_name] = group_tasks
        
        end = time.perf_counter()

        print(Color.BOLD + Color.GREEN + '⭐ СФОРМИРОВАЛ ЗАДАЧИ!' + Color.END)
        print(f'Время формирования задач по факту вышло: {((end - start) / 60):0.1f} мин.')

        return json_tasks

     

    @staticmethod
    def get_data(
                tasks,
                statistics = DataConfig.STATISTICS,
                slices = ['regionName', #регион
                            'tvCompanyName' #телесеть
                            ]
                     ):
        """
            Метод для генерации задания API для руководителей групп
        """
        #with WrapperNoPrints():
        df = BaseDataService._execute_tasks(tasks)    
        # Приводим порядок столбцов в соответствие с условиями расчета
        df = df[['prj_name'] + slices + statistics]
        df['prj_name'] = df['prj_name'].str.upper()
        return df


    @staticmethod
    def validate_dataframe(df):
        """
            Проверяет DataFrame на валидность
        """
        if df is None:
            return False
        if not isinstance(df, pd.DataFrame):
            return False
        if df.empty:
            return False
        if df.isna().all().all():
            return False
        return True


    @staticmethod
    def process_tasks_with_validation(json_tasks, get_data_func = None, max_workers = 10):
        """
        Обрабатывает задачи и возвращает результат с отчетом
        
        Args:
            json_tasks: словарь с задачами для обработки
            get_data_func: функция для получения данных (по умолчанию LeaderShipDataExtractor.get_data)
            max_workers: максимальное количество потоков
        """
        if get_data_func is None:
            get_data_func = LeaderShipDataExtractor.get_data
        
        full_res = []
        validation_report = {}

        for group, tasks in json_tasks.items():
            group_results = []
            problematic_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
                futures = {executor.submit(get_data_func, json_params): json_params 
                        for json_params in tasks.values()}
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if LeaderShipDataExtractor.validate_dataframe(result):
                        group_results.append(result)
                    else:
                        problematic_count += 1
            
            validation_report[group] = {
                'total': len(tasks),
                'successful': len(group_results),
                'problematic': problematic_count
            }
            
            if group_results:
                full_res.append(pd.concat(group_results, ignore_index = True))

        data_api = pd.concat(full_res, ignore_index=True) if full_res else pd.DataFrame()
        
        # Вывод отчета
        print("Отчет по обработке:")
        for group, report in validation_report.items():
            success_rate = report['successful'] / report['total'] * 100
            print(f"Группа {group}: {report['successful']}/{report['total']} ({success_rate:.1f}%) успешно")
        
        print(f"✅ Итоговый размер data_api: {data_api.shape}")
        
        return data_api
    

    @staticmethod
    def make_api_calculation(date_filter, json_tasks: dict, share_table: pd.DataFrame):
        """
            Метод реализует выгрузку данных для руководителей групп для конкретного временного периода данных.
        """
        # 1. Выгрузка данных для всевозможных групп
        data_api = LeaderShipDataExtractor.process_tasks_with_validation(json_tasks)

        # 2. Изменение названий локальных каналов (оптимизированная версия)
        mask_78 = data_api['tvCompanyName'] == 'ТЕЛЕКАНАЛ 78 (САНКТ-ПЕТЕРБУРГ)'
        mask_spb = data_api['tvCompanyName'] == 'САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)'
        
        data_api.loc[mask_78, 'tvCompanyName'] = 'ТЕЛЕКАНАЛ 78 САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)'
        data_api.loc[mask_spb, 'tvCompanyName'] = 'ТЕЛЕКАНАЛ САНКТ-ПЕТЕРБУРГ (САНКТ-ПЕТЕРБУРГ)'
        
        # 3. Отбор каналов ГТРК (оптимизированная версия)
        gtrk_mask = data_api['tvCompanyName'].str.contains('ГТРК', case = False, na = False, regex = False)
        gtrk = data_api[gtrk_mask].copy()
        
        # 4. Отбор каналов Четвертый канал (Екатеринбург)
        channel_4_mask = data_api['tvCompanyName'] == 'ЧЕТВЕРТЫЙ КАНАЛ (ЕКАТЕРИНБУРГ)'
        channel_4 = data_api[channel_4_mask].copy()
        
        # 5. Формирование выходной таблицы
        output_df = LeaderShipDataExtractor.get_output(date_filter, data_api, share_table)
        
        # Выделяем название колонки с временным периодом в отдельную переменную
        date_column = output_df.columns[3]
        
        # 6. Отбор России 1 (оптимизированные маски)
        russia_1_mask = output_df['Телеканал'] == 'РОССИЯ 1'
        russia_1_only = output_df[russia_1_mask].copy()
        senza_russia_1 = output_df[~russia_1_mask].copy()
        
        # 7. Преобразования с выходным ДатаФреймом
        output = output_df.copy()
        # Оптимизированное создание колонки (избегаем + для строк)
        output['Теканал+Город+БЦА'] = output['Телеканал'].str.cat([output['Город'], output['БЦА']], sep = ' ')
        
        # 8. Отбор Пятницы (Екатеринбург) все 14-44
        pyatniza_mask = output['Теканал+Город+БЦА'] == 'ПЯТНИЦА ЕКАТЕРИНБУРГ ВСЕ 14-44'
        senza_pyatniza_ekb = output[~pyatniza_mask][output.columns[:-1]].copy()
        pyatniza_only = output[pyatniza_mask][output.columns[:-1]].copy()
        
        # 9. Формируем датафрейм без России 1 и Пятницы (Екатеринбург)
        filtered_full_mask = ~pyatniza_mask & ~russia_1_mask
        filtered_full = output[filtered_full_mask][output.columns[:-1]].copy()
        
        # 10. Вычисления
        Russia_1_with_gtrk = LeaderShipDataExtractor.sum_2_channels(gtrk, russia_1_only, date_column)
        Pyatniza_with_channel_4 = LeaderShipDataExtractor.sum_2_channels(channel_4, pyatniza_only, date_column)
        
        output_data = LeaderShipDataExtractor.make_output(Russia_1_with_gtrk, Pyatniza_with_channel_4, filtered_full, share_table)
        return output_data

    

    @staticmethod
    def export_leadership_data(
                selected_periods: list, 
                date_periods: dict, 
                calculation_params: dict, 
                share_table: pd.DataFrame,
                slices = [
                    'regionName', #регион
                    'tvCompanyName' #телесеть
                                ],
                sortings = {'tvCompanyName':'ASC'}
                ):
        """
            Метод реализует выгрузку данных для руководителей групп для серии временных периодов
        """
        results = {}
        for period in selected_periods:
            date_filter = date_periods[period]
            
            # Формирование задач в формате json
            #json_tasks = LeaderShipDataExtractor.build_tasks_threaded(calculation_params, date_filter)
            json_tasks = LeaderShipDataExtractor.build_json_tasks(date_filter, calculation_params, slices, sortings)
            
            ##################### ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ СОЗДАНИЯ ЗАДАЧ. ОСТАВИТЬ!!!!!!!!! #####################
            #json_tasks = {}
            #for key, params in regions_params.items():
            #    json_tasks[key] = LeaderShipDataExtractor.build_json_tasks(date_filter, *params)
            #####################################################################################################
            results[period] = LeaderShipDataExtractor.make_api_calculation(date_filter, json_tasks, share_table)

            time.sleep(45)
    
        return results
        
    
