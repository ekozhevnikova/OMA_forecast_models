import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import xlsxwriter
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from mediascope_api.mediavortex import catalogs as cwc
cats = cwc.MediaVortexCats()

from OMA_tools.regions.data_extraction.task_builder import BaseDataService
from OMA_tools.regions.data_extraction.data_coworker import EmployeeExportService
from OMA_tools.io_data.colors import *

import warnings
warnings.filterwarnings('ignore')


class ForecastNewChannels:
    """
        Класс для построения прогноза новых каналов Тематического ТВ
    """
    def __init__(
            self, 
            start_date: str, 
            stop_date: str, 
            target_data_to_forecast: dict,
            forecast_years: list, 
            output_file: str
        ):
        """
            Атрибуты:
            ----------
                start_date: str
                    Стартовая дата для выгрузки данных в формате str "Y-m-d"
                stop_date: str
                    Конечная дата для выгрузки данных в формате str "Y-m-d"
                target_data_to_forecast: dict
                    Словарь из Целевых Каналов и БЦА, которые попросили спрогнозировать
                forecast_years: list
                    Список из годов, для которых будем строить прогноз. Это может быть и 1 год, и 2. И, например, остаток текущего года
                output_file: str
                    Название/Путь к выходному файлу в формате .xlsx
        """
        self.start_date = start_date
        self.stop_date = stop_date
        self.target_data_to_forecast = target_data_to_forecast
        self.forecast_years = forecast_years
        self.output_file = output_file

        self.options = {
            "kitId": 4, #TV Index Plus All Russia
            "totalType": "TotalChannels", #Расчет Share от Total Channels
        }
        self.weekday_filter = None
        self.daytype_filter = None
        self.basedemo_filter = None
        self.targetdemo_filter = None
        self.location_filter = None

        self.MONTHS = {
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь',
        }
    

    @staticmethod
    def assign_audiences_to_channels(target_channels: pd.DataFrame):

        BCA_OPTIONS = {
            1: 'Все 25-49',
            2: 'Ж 25-49',
            3: 'М 25-49',
            4: 'Все 4-40'
        }

        target_channels = target_channels.sort_values(by = 'name')

        def extract_digits_and_commas(text):
            """
                Оставляет только цифры и запятые
            """
            return re.sub(r'[^0-9,]', '', text)


        print("Доступные БЦА:")
        for key, value in BCA_OPTIONS.items():
            print(f"  {key}. {value}")
        print(Color.BOLD + Color.CORAL + "\nДля выбора нескольких БЦА введите номера через запятую" + Color.END)
        print("Например: 1,2,3\n")


        target_dict = {}

        for idx, row in target_channels.iterrows():
            print(Color.BLUE + f"\nКанал {row['name']} (ID: {row['id']})" + Color.END)
            
            choice = input("Введите номера БЦА: ").strip()
            choice_cleaned = extract_digits_and_commas(choice)
            numbers = [int(x.strip()) for x in choice_cleaned.split(',') if x.strip()]
            
            audiences = [BCA_OPTIONS[n] for n in numbers]
            target_dict[row['name']] = audiences
        
        print(Color.BOLD + '\nПо итогу сгенерированы следующие параметры:' + Color.END)
        for key, value in target_dict.items():
            print(f'• {key}: {value}')
        
        return target_dict

    

    @staticmethod
    def get_channels_id(channels_names: list):
        """
            Метод для поиска каналов по их названиям

            Параметры:
            ----------
                channels_names: list
                    Список из названий каналов
            Returns:
            ----------
                channels_dict_filtered: pd.DataFrame
                    Таблица из найденных названий
            
        """
        # Фильтруем по тем каналам, которые нам нужны
        channels_dict = cats.get_tv_company(name = channels_names)
        channels_dict = channels_dict[['id','name']].reset_index(drop = True)

        # Отбираем только с "(СЕТЕВОЕ ВЕЩАНИЕ)"
        channels_dict['is_network_broadcast'] = channels_dict['name'].str.contains('(СЕТЕВОЕ ВЕЩАНИЕ)', na = False)
        channels_dict_filtered = channels_dict[channels_dict['is_network_broadcast'] == True].reset_index(drop = True)
        channels_dict_filtered = channels_dict_filtered.drop(['is_network_broadcast'], axis = 1)
        channels_dict_filtered['name'] = channels_dict_filtered['name'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))
        channels_dict_filtered = channels_dict_filtered[['name', 'id']]
        channels_dict_filtered = channels_dict_filtered.sort_values(by = 'name')
        return channels_dict_filtered.reset_index(drop = True)
    

    def normalize_channel_name(self, name: str) -> str:
        """
            Нормализует название канала, удаляя временные маркеры и суффиксы
        """
        if not isinstance(name, str):
            return name
        
        # Удаляем все, что в скобках (включая даты, "СЕТЕВОЕ ВЕЩАНИЕ" и т.д.)
        normalized = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
        
        # Удаляем временные маркеры без скобок (если такие есть)
        normalized = re.sub(r'\s*(ДО|С|ПО)\s+\d{2}[/.-]\d{2}[/.-]\d{4}\s*', ' ', normalized)
        
        # Удаляем лишние пробелы в начале и конце
        normalized = normalized.strip()
        
        # Приводим к верхнему регистру (опционально)
        # normalized = normalized.upper()
        
        # Удаляем множественные пробелы между словами
        normalized = ' '.join(normalized.split())
        
        return normalized
    

    def generate_periods(self):
        """
            Метод для генерации периодов для выгрузки данных.
            
            Параметры:
            ----------
                start_date: str
                    Дата старта в формате строки "Y-m-d"
                stop_date: str
                    Дата окончания в формате строки "Y-m-d"
            Returns:
            ----------
                periods: list
                    Список из периодов в разбивке по годам
        """
        # Преобразуем строки в объекты datetime
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        stop = datetime.strptime(self.stop_date, '%Y-%m-%d')
        
        periods = []
        
        # Генерируем периоды по годам
        current = start
        while current <= stop:
            year_start = current
            year_end = datetime(current.year, 12, 31)
            
            # Если конечная дата раньше конца года, используем stop_date
            if year_end > stop:
                year_end = stop
            
            periods.append([ (year_start.strftime('%Y-%m-%d'), year_end.strftime('%Y-%m-%d')) ])
            
            # Переходим к следующему году
            current = datetime(current.year + 1, 1, 1)
        return periods


    def get_data(self, adult_tasks, child_tasks, max_workers = 10):
        """
        Метод для выгрузки данных из БД для взрослой аудитории: Все 25-49, Ж 25-49, М 25-49
        """
        try:
            adult_df = BaseDataService._execute_tasks(adult_tasks)
            child_df = BaseDataService._execute_tasks(child_tasks)
            child_df['prj_name'] = child_df['prj_name'].replace('Total. Ind', 'ВСЕ 4-40')
            full_df = pd.concat([adult_df, child_df]).reset_index(drop=True)
            return full_df

        except Exception as e:
            print(f"❌ Ошибка при загрузке данных: {e}")
            return pd.DataFrame()


    def acquistare_data(
        self, type_of_grouping: str, company_filter: str, 
        statistics: list, periods: list
        ):
        """
            Метод для выгрузки данных из БД.

            Параметры:
            ----------
                type_of_grouping: str
                    Тип разбивки. Может быть "by dates", "by months", "period"
                company_filter: str
                    Выгружаемые телекомпании
                time_filter: str
                    Эфирные сутки (Интервал вещания). Должны быть заданы следующим образом: "timeBand1 >= 60000 AND timeBand1 < 260000"
                statistics: list
                    Выгружаемые статистики

            Returns:
            ----------
                result_by_bca: dict
                    Словарь, где ключ - БЦА, значение - датафрейм с данными по каналу для этой БЦА.
        """
        # Словарь с целевыми аудиториями: ключ - название переменной (target), значение - ее синтаксис (syntax)
        targets = {
            'ВСЕ 25-49': 'age >= 25 AND age <= 49',
            'М 25-49': 'age >= 25 AND age <= 49 AND sex = 1',
            'Ж 25-49': 'age >= 25 AND age <= 49 AND sex = 2',
        }


        sortings = {}
        slices = []

        if type_of_grouping not in ["by dates", "by months", "period"]:
            raise ValueError(
                    f"Задан неверный тип группировки '{type_of_grouping}'. Выберите группировку из списка: {'by dates', 'by months', 'period'}"
            )

        # Определяем сортировки в зависимости от заданной разбивки
        if type_of_grouping == 'by dates':
            sortings = {'tvCompanyName': 'ASC', 'researchDate': 'ASC'}
            slices = ['researchDate', 'tvCompanyName']

        elif type_of_grouping == 'by months':
            sortings = {'tvCompanyName': 'ASC', 'researchMonth': 'ASC'}
            slices = ['researchMonth', 'tvCompanyName']

        else:
            sortings = {'tvCompanyName': 'ASC'}
            slices = ['tvCompanyName']

        ################################################### НОВЫЙ КУСОК ###################################################

        # Разбивка периодов по годам для генерации задач в формате JSON для каждого года отдельно
        periods_dict = {}
        for item in periods:
            start_date, end_date = item[0]
            year = datetime.strptime(start_date, '%Y-%m-%d').year
            periods_dict[year] = [(start_date, end_date)]
        

        import time

        start = time.perf_counter()
        print(Color.BOLD + Color.BLUE + '=== 🕑 ФОРМИРУЮ ЗАДАЧИ В ФОРМАТЕ JSON ДЛЯ ОТПРАВКИ НА СЕРВЕР ===' + Color.END)

        adult_tasks_full = {}
        children_json_tasks_full = {}
        for year, date_filter in periods_dict.items():
            # Генерация задач для взрослых аудиторий
            adult_tasks_full[year] = BaseDataService._build_timeband_common_params(
                            date_filter = date_filter, company_filter = company_filter, 
                            basedemo_filter = None, regions_id = None,          # работаем в Федеральной Базе
                            targets = targets, time_filter = 'timeBand1 >= 60000 AND timeBand1 < 260000', 
                            statistics = statistics, slices = slices, 
                            sortings = sortings, options = self.options,
                            location_filter = self.location_filter, weekday_filter = self.weekday_filter,
                            daytype_filter = self.daytype_filter, targetdemo_filter = self.targetdemo_filter
                    )
            
            # Генерация задач для детской аудитории
            children_json_tasks_full[year] = BaseDataService._build_timeband_common_params(
                            date_filter = date_filter, company_filter = company_filter, 
                            basedemo_filter = 'age >= 4 AND age <= 40', regions_id = None,          # работаем в Федеральной Базе
                            targets = None, time_filter = 'timeBand1 >= 60000 AND timeBand1 < 220000', 
                            statistics = statistics, slices = slices, 
                            sortings = sortings, options = self.options,
                            location_filter = self.location_filter, weekday_filter = self.weekday_filter,
                            daytype_filter = self.daytype_filter, targetdemo_filter = self.targetdemo_filter
                    )
        
        end = time.perf_counter()

        print(Color.BOLD + Color.GREEN + '⭐ СФОРМИРОВАЛ ЗАДАЧИ!' + Color.END)
        print(f'Время формирования задач по факту вышло: {((end - start)):0.1f} сек.')

        print(Color.BOLD + Color.VIOLET + '=== НАЧИНАЮ ВЫГРУЗКУ ДАННЫХ ===' + Color.END)
        # Параллельная обработка годов
        final_results = []
        max_workers = min(len(periods_dict), 10)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_data, adult_tasks_full[year], children_json_tasks_full[year]): year 
                for year in periods_dict.keys()
            }
            
            for future in as_completed(futures):
                year = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        final_results.append(result)
                        print(f"✅ Год {year} обработан успешно")
                except Exception as e:
                    print(f"❌ Ошибка при обработке года {year}: {e}")

        general_result = pd.concat(final_results, ignore_index = True) if final_results else pd.DataFrame()
        general_result['tvCompanyName'] = general_result['tvCompanyName'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))

        ########################################### НОВЫЙ КУСОК ###########################################
        # Добавляем нормализованное имя
        general_result['normalized_name'] = general_result['tvCompanyName'].apply(
            self.normalize_channel_name
        )
        
        # Логируем объединение
        print(Color.BOLD + Color.BLUE + "\n=== ОБЪЕДИНЕНИЕ КАНАЛОВ ПО НОРМАЛИЗОВАННЫМ НАЗВАНИЯМ ===" + Color.END)
        
        # Группируем данные по нормализованному имени
        grouped_data = []
        
        for normalized_name in general_result['normalized_name'].unique():
            # Берем все данные для этого нормализованного имени
            channel_data = general_result[general_result['normalized_name'] == normalized_name].copy()
            
            # Получаем все оригинальные названия для логирования
            original_names = channel_data['tvCompanyName'].unique()
            
            if len(original_names) > 1:
                print(f"📌 Объединены в '{normalized_name}':")
                for name in original_names:
                    print(f"   - {name}")
            
            # Заменяем tvCompanyName на нормализованное имя
            channel_data['tvCompanyName'] = normalized_name
            
            # Удаляем служебные колонки
            channel_data = channel_data.drop(['normalized_name'], axis=1)
            
            # Удаляем ID (он больше не нужен)
            if 'id' in channel_data.columns:
                channel_data = channel_data.drop(['id'], axis=1)
            
            # Удаляем полные дубликаты строк (если данные одинаковые)
            channel_data = channel_data.drop_duplicates()
            
            grouped_data.append(channel_data)
        
        general_result = pd.concat(grouped_data, ignore_index=True)
        
        print(f"\n✅ После объединения: {general_result['tvCompanyName'].nunique()} уникальных каналов")
        ########################################### КОНЕЦ НОВОГО КУСКА ###########################################

        if type_of_grouping == 'by dates':
            general_result.rename(
                columns = {
                    'prj_name': 'БЦА', 
                    'tvCompanyName': 'Канал', 
                    'researchDate': 'Дата'
                }, 
                inplace = True
            )

        elif type_of_grouping == 'by months':
            general_result.rename(
                columns = {
                    'prj_name': 'БЦА', 
                    'tvCompanyName': 'Канал', 
                    'researchMonth': 'Месяц'
                }, 
                inplace = True
            )
            
        elif type_of_grouping == 'period':
            general_result.rename(
                columns = {
                    'prj_name': 'БЦА', 
                    'tvCompanyName': 'Канал'
                }, 
                inplace = True
            )

        # Генерируем результат поканально для каждой БЦА
        results_by_channels = {}
        
        for channel in general_result['Канал'].unique():
            channel_data = general_result[general_result['Канал'] == channel]
            
            results_by_channels[channel] = {
                bca: channel_data[channel_data['БЦА'] == bca].reset_index(drop = True)
                for bca in channel_data['БЦА'].unique()
            }
        return results_by_channels
    

    def form_output_for_forecast(self, dict_data: dict):
        """
            Метод для формирования выходных таблиц для построения прогноза новых каналов 
            !!! ВАЖНО !!! Метод работает только ли с 
            Параметры:
            ----------
                dict_data: dict of dicts
                    Словарь словарей, где
                        КЛЮЧ: Название канала
                        ЗНАЧЕНИЕ: словарь из данных, где
                            ключ: БЦА
                            значение: датафрейм с данными
        """      
        month_indices = list(range(1, 13))  # [1, 2, 3, ..., 12]

        transformed_dict = {}
        
        for channel, bca_dict in dict_data.items():
            
            by_bca = {}
            for bca, data in bca_dict.items():
                if 'Месяц' not in data.columns:
                    raise ValueError(
                            "Отсутствует столбец с названием 'Месяц'. Я не смогу трансформировать таблицы."
                    )
                    
                data = data.drop(['БЦА', 'Канал'], axis=1)
                data['Год'] = pd.to_datetime(data['Месяц']).dt.year
                data['month'] = pd.to_datetime(data['Месяц']).dt.month

                # Создаем словарь {год: [месяцы, которые были в этом году]}
                original_months_by_year = {}
                for year in data['Год'].unique():
                    # Сохраняем индексы месяцев (числа от 1 до 12)
                    original_months_by_year[year] = data[data['Год'] == year]['month'].unique().tolist()
                    #print(f"📊 Год {year}: индексы месяцев с данными = {original_months_by_year[year]}")


                #data['month'].replace(self.MONTHS, inplace = True)
                
                data_transformed = data.pivot_table(
                    index='Год', 
                    columns='month',  # здесь month - это числа (1-12)
                    values='Share'
                ).reset_index()

                # Переименовываем столбцы в названия месяцев для отображения
                data_transformed.columns = ['Год'] + [self.MONTHS[i] for i in range(1, 13)]
                
                # Переименуем столбцы (уберем имя 'month' и сделаем нормальные названия)
                data_transformed = data_transformed.fillna(0)
                
                # Заменяем на NaN только те месяцы, которых НЕ БЫЛО в исходных данных
                # И только для прогнозных лет
                if hasattr(self, 'forecast_years') and self.forecast_years:
                    for year in self.forecast_years:
                        if year in data_transformed['Год'].values:
                            mask = data_transformed['Год'] == year
                            original_months_for_year = original_months_by_year.get(year, [])
                            #print(f"🔍 Год {year}: оригинальные индексы месяцев = {original_months_for_year}")
                            
                            for month_idx in range(1, 13):
                                month_name = self.MONTHS[month_idx]
                                # Сравниваем числа (индексы) с числами
                                if month_idx not in original_months_for_year:
                                    #print(f"  ❌ Месяц {month_name} (индекс {month_idx}) не был в исходных данных, заменяем на NaN")
                                    data_transformed.loc[mask, month_name] = np.nan
                                #else:
                                #    print(f"  ✅ Месяц {month_name} (индекс {month_idx}) был в исходных данных, оставляем как есть")
                
                by_bca[bca] = data_transformed
            
            transformed_dict[channel] = by_bca
        return transformed_dict
    

    # Функция для замены выбросов на медиану
    def replace_outliers_with_median(self, data_list):
        """
            Заменяет выбросы на медиану
        """
        clean_data = [x for x in data_list if x is not None]
        if len(clean_data) == 0:
            return []
        
        Q1 = np.percentile(clean_data, 25)
        Q3 = np.percentile(clean_data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = np.median(clean_data)
        
        result = []
        for x in data_list:
            if x is None:
                result.append(None)
            elif x < lower_bound or x > upper_bound:
                result.append(median)
            else:
                result.append(x)
        return result
        

    def calculate_base(self, df):
        """
            Рассчитывает базу для прогноза на основе последних 12 месяцев с данными
            
            Параметры:
            ----------
                df: pd.DataFrame 
                    Таблица с данными
            
            Возвращает:
            ----------
                float: среднее значение последних 12 месяцев (после очистки от выбросов)
        """
        # Сортируем по году
        df_sorted = df.sort_values('Год').reset_index(drop = True)
        
        # Список месяцев в правильном порядке
        months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 
                'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
        
        # Собираем все значения с указанием года и месяца
        all_values = []  # список кортежей (год, месяц_индекс, значение)
        
        for row_idx in range(len(df_sorted)):
            year = df_sorted.loc[row_idx, 'Год']
            
            for month_idx, month in enumerate(months):
                value = df_sorted.loc[row_idx, month]
                if pd.notna(value):
                    all_values.append((year, month_idx, value))
        
        if not all_values:
            print("  Предупреждение: нет данных для расчета базы")
            return 0.0
        
        # Сортируем по году и месяцу (чтобы гарантировать правильный порядок)
        all_values.sort(key = lambda x: (x[0], x[1]))
        
        # Находим последний месяц с данными
        last_year, last_month_idx, last_month_value = all_values[-1]
        
        print(f"  Последние фактические данные: {last_year} г., {months[last_month_idx]}, значение: {last_month_value}")
        
        # Собираем последние 12 месяцев (включая последний)
        last_12_months = []
        last_12_months_info = []  # для хранения информации о годах и месяцах
        
        # Идем назад от последнего элемента
        start_idx = len(all_values) - 1
        months_collected = 0
        
        for i in range(start_idx, -1, -1):
            if months_collected >= 12:
                break
            year, month_idx, value = all_values[i]
            last_12_months.append(value)
            last_12_months_info.append((year, month_idx, value))
            months_collected += 1
        
        # Переворачиваем списки, чтобы сохранить хронологический порядок
        last_12_months.reverse()
        last_12_months_info.reverse()
        
        # Получаем первый и последний месяц в периоде
        first_year, first_month_idx, first_value = last_12_months_info[0]
        last_year_base, last_month_idx_base, last_value_base = last_12_months_info[-1]
        
        print(f"  Период для расчета базы: с {months[first_month_idx]} {first_year} г. по {months[last_month_idx_base]} {last_year_base} г.")
        print(f"  Первый месяц базы: {months[first_month_idx]} {first_year} г.")
        print(f"  Последний месяц базы: {months[last_month_idx_base]} {last_year_base} г.")
        print('\n')

        
        # Очищаем от выбросов
        cleaned_last_12_months = self.replace_outliers_with_median(last_12_months)
        
        result = np.mean(cleaned_last_12_months)
        
        return result
    

    def filter_data_by_targets(self, all_results: dict):
        """
            Отбирает из всех выгруженных данных только нужные каналы и нужные БЦА, которые попросили спрогнозировать.
    
            Параметры:
            ----------
            all_results: dict
                Словарь из всех выгруженных данных: каналы и все БЦА
            
            Возвращает:
            ----------
            filtered_data: dict
                Словарь из отобранных каналов и БЦА
        """

        filtered_data = {}

        for channel, bcas_to_keep in self.target_data_to_forecast.items():
            # Нормализуем имя канала из запроса
            normalized_target = self.normalize_channel_name(channel)
            
            # Ищем соответствие в выгруженных данных
            found_channel = None
            for available_channel in all_results.keys():
                if self.normalize_channel_name(available_channel) == normalized_target:
                    found_channel = available_channel
                    break
            
            if found_channel is None:
                print(f" 🚩 WARNING: Канал '{channel}' не найден в исходных данных. Проверьте название.")
                continue

            # Получаем словарь БЦА для этого канала
            channel_data = all_results[found_channel]
            # Приводим ключи к нижнему регистру, чтобы проще было совмещать
            lower_channel_data = {k.lower(): v for k, v in channel_data.items()}

            # Отбираем только нужные БЦА
            filtered_channel_data = {}
            for bca in bcas_to_keep:
                if bca.lower() in lower_channel_data:
                    filtered_channel_data[bca] = lower_channel_data[bca.lower()]
                else:
                    print(f" ⏭️ Для канала '{found_channel}' БЦА '{bca}' не найдена")

            # Добавляем канал в результат
            if filtered_channel_data:
                # Используем НОРМАЛИЗОВАННОЕ имя как ключ
                filtered_data[normalized_target] = filtered_channel_data
                print(f"✅ Найден канал: '{normalized_target}' (исходный запрос: '{channel}')")
            else:
                print(f" ⚠️ Для канала '{found_channel}' не найдено ни одной БЦА")

        print(Color.ROYAL_BLUE + f"\n🚀 Итог: отобрано {len(filtered_data)} каналов из {len(all_results)}" + Color.END)
    
        return filtered_data




    # Функция для создания отчета по одному БЦА
    def create_bca_report(self, writer, worksheet, df, bca_name, channel_name, start_row):
        """
        Создает отчет для одного БЦА

        Параметры:
        ----------
            writer: ExcelWriter 
                объект
            df: pd.DataFrame 
                Таблица с данными
            bca_name: str
                название БЦА
            channel_name: str
                название канала
            start_row: int
                начальная строка для размещения отчета (индекс Python)
            forecast_years: list
                список прогнозируемых годов (например, [2025, 2026] или [2025] или [2026])
        """
        workbook = writer.book
        
        # ============= ДОБАВЛЯЕМ СТРОКИ ДЛЯ ПРОГНОЗИРУЕМЫХ ГОДОВ, ЕСЛИ ИХ НЕТ =============
        df_copy = df.copy()
        for forecast_year in self.forecast_years:
            if forecast_year not in df_copy['Год'].values:
                # Добавляем строку для прогнозируемого года
                new_row = {'Год': forecast_year}
                for col in df_copy.columns[1:]:
                    new_row[col] = None
                df_copy = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index = True)
                print(Color.MAROON + f"  Добавлена строка для {forecast_year} года" + Color.END)
        
        df_copy = df_copy.sort_values('Год').reset_index(drop = True)
        
        # Рассчитываем базу для этого БЦА
        base_value = self.calculate_base(df_copy)
        
        # Форматы для этого БЦА
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0
        })
        
        data_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 0, 'num_format': '0.000'
        })
        
        year_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0'
        })
        
        total_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0.000', 'bg_color': '#E8E8E8'
        })
        
        yoy_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0.00', 'bg_color': '#FFE4B5'
        })
        
        jan_to_year_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0.00', 'bg_color': '#E0FFFF'
        })
        
        forecast_year_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0', 'font_color': '#FF0000'
        })
        
        forecast_data_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0.000', 'font_color': '#FF0000'
        })
        
        forecast_yoy_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0.00', 'font_color': '#FF0000'
        })
        
        forecast_jan_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 0,
            'num_format': '0.00', 'font_color': '#FF0000'
        })
        
        # Название БЦА (ячейка A{start_row+2})
        bca_format = {}
        base_format = {}
        
        if bca_name.lower() == 'все 25-49':
            bca_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#99FF66',
                'align': 'center', 'valign': 'vcenter', 'bold': True
            })
            base_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#99FF66',
                'align': 'center', 'valign': 'vcenter', 'num_format': '0.000'
            })
        elif bca_name.lower() == 'ж 25-49':
            bca_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#ffccff',
                'align': 'center', 'valign': 'vcenter', 'bold': True
            })
            base_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#ffccff',
                'align': 'center', 'valign': 'vcenter', 'num_format': '0.000'
            })
        elif bca_name.lower() == 'м 25-49':
            bca_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#66ccff',
                'align': 'center', 'valign': 'vcenter', 'bold': True
            })
            base_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#66ccff',
                'align': 'center', 'valign': 'vcenter', 'num_format': '0.000'
            })
        elif bca_name.lower() == 'все 4-40':
            bca_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#f5f587',
                'align': 'center', 'valign': 'vcenter', 'bold': True
            })
            base_format = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 12, 'bg_color': '#f5f587',
                'align': 'center', 'valign': 'vcenter', 'num_format': '0.000'
            })
        
        worksheet.write(start_row, 0, bca_name, bca_format)
        
        # ============= ЗАГОЛОВКИ ТАБЛИЦЫ =============
        header_row = start_row + 2
        start_col = 1
        
        for col_idx, col_name in enumerate(df_copy.columns):
            worksheet.write(header_row, start_col + col_idx, col_name, header_format)
        
        # ============= ДАННЫЕ =============
        data_start_row = header_row + 1
        
        for row_idx in range(len(df_copy)):
            excel_row = data_start_row + row_idx
            year = df_copy.loc[row_idx, 'Год']
            
            # Определяем, нужен ли прогноз для этого года (есть в списке forecast_years)
            needs_forecast = year in self.forecast_years
            
            # Записываем год (красным только если это год, для которого мы строим прогноз)
            if needs_forecast:
                worksheet.write(excel_row, start_col, int(year), forecast_year_format)
            else:
                worksheet.write(excel_row, start_col, int(year), year_format)
            
            # Записываем значения месяцев
            for col_idx in range(1, len(df_copy.columns)):
                value = df_copy.iloc[row_idx, col_idx]
                if pd.notna(value):
                    # Фактические данные всегда черным
                    worksheet.write(excel_row, start_col + col_idx, value, data_format)
                # Пустые ячейки пока не заполняем (заполним позже, если нужен прогноз)
        
        # ============= ИТОГО =============
        total_col = start_col + len(df_copy.columns)
        worksheet.write(header_row, total_col, 'ИТОГО', header_format)
        
        for row_idx in range(len(df_copy)):
            excel_row = data_start_row + row_idx
            year = df_copy.loc[row_idx, 'Год']
            
            # Для исторических годов (не требующих прогноза) считаем ИТОГО сразу
            if year not in self.forecast_years:
                start_letter = chr(ord('B') + 1)
                end_letter = chr(ord('B') + len(df_copy.columns) - 1)
                formula = f'=AVERAGE({start_letter}{excel_row + 1}:{end_letter}{excel_row + 1})'
                worksheet.write(excel_row, total_col, formula, total_format)
        
        # ============= ГОД К ГОДУ =============
        yoy_col = total_col + 1
        worksheet.write(header_row, yoy_col, 'Год к году', header_format)
        
        for row_idx in range(len(df_copy)):
            excel_row = data_start_row + row_idx  # строка в Excel для записи (0-индексация для write)
            
            if row_idx == 0:
                worksheet.write(excel_row, yoy_col, '', yoy_format)
            else:
                # Номера строк для формулы (1-индексация Excel)
                current_excel_row_num = excel_row + 1      # строка с текущим годом
                prev_excel_row_num = excel_row             # строка с предыдущим годом
                
                formula = f'=O{current_excel_row_num}/O{prev_excel_row_num}'
                worksheet.write(excel_row, yoy_col, formula, yoy_format)
        
        # ============= ЯНВАРЬ К ГОДУ =============
        jan_to_year_col = yoy_col + 1
        worksheet.write(header_row, jan_to_year_col, 'Январь к году', header_format)
        
        for row_idx in range(len(df_copy)):
            excel_row = data_start_row + row_idx
            year = df_copy.loc[row_idx, 'Год']
            
            jan_col = chr(ord('B') + 1)
            jan_cell = f'{jan_col}{excel_row + 1}'
            total_cell = f'${chr(ord("A") + total_col)}${excel_row + 1}'
            formula = f'={jan_cell}/{total_cell}'
            
            if year in self.forecast_years:
                worksheet.write(excel_row, jan_to_year_col, formula, forecast_jan_format)
            else:
                worksheet.write(excel_row, jan_to_year_col, formula, jan_to_year_format)
        
        # ============= БАЗА =============
        base_row = header_row - 2
        base_col = total_col - 2
        worksheet.write(base_row, base_col, 'база', base_format)
        worksheet.write(base_row, base_col + 1, base_value, base_format)
        
        # ============= СЕЗОННЫЕ КОЭФФИЦИЕНТЫ =============
        season_start_row = data_start_row + len(df_copy) + 2
        season_start_col = start_col

        for col_idx, col_name in enumerate(df_copy.columns):
            worksheet.write(season_start_row, season_start_col + col_idx, col_name, header_format)

        season_row = season_start_row + 1
        season_rows_list = []

        # Собираем сезонные коэффициенты для всех полностью заполненных исторических годов
        for row_idx in range(len(df_copy)):
            year = df_copy.loc[row_idx, 'Год']
            
            # Пропускаем года, для которых нужен прогноз (их данные могут быть неполными)
            if year in self.forecast_years:
                continue
            
            # Проверяем, что год полностью заполнен
            if df_copy.iloc[row_idx, 1:].notna().all():
                excel_data_row = data_start_row + row_idx
                
                worksheet.write(season_row, season_start_col, int(year), year_format)
                
                for col_idx in range(1, len(df_copy.columns)):
                    month_col_letter = chr(ord('B') + col_idx)
                    total_col_letter = chr(ord('A') + total_col)
                    season_formula = f'={month_col_letter}{excel_data_row + 1}/${total_col_letter}${excel_data_row + 1}'
                    worksheet.write(season_row, season_start_col + col_idx, season_formula, data_format)
                
                season_rows_list.append(season_row)
                season_row += 1

        # Прогнозные сезонные коэффициенты (усредненные по всем историческим годам)
        if season_rows_list:
            # Для каждого года, требующего прогноза, создаем строку сезонных коэффициентов
            forecast_season_rows = {}
            
            for forecast_year in self.forecast_years:
                forecast_season_row = season_row
                forecast_row_idx = df_copy[df_copy['Год'] == forecast_year].index[0]
                excel_data_row = data_start_row + forecast_row_idx
                
                worksheet.write(forecast_season_row, season_start_col, int(forecast_year), forecast_year_format)
                
                for col_idx in range(1, len(df_copy.columns)):
                    month_name = df_copy.columns[col_idx]
                    current_value = df_copy.iloc[forecast_row_idx, col_idx]
                    month_col_letter = chr(ord('B') + col_idx)
                    
                    # Проверяем, есть ли фактические данные для этого месяца
                    if pd.notna(current_value):
                        # Для месяцев с фактическими данными - рассчитываем коэффициент как отношение к итогу
                        total_col_letter = chr(ord('A') + total_col)
                        season_formula = f'={month_col_letter}{excel_data_row + 1}/${total_col_letter}${excel_data_row + 1}'
                        worksheet.write(forecast_season_row, season_start_col + col_idx, season_formula, data_format)
                    else:
                        # Для прогнозных месяцев - используем среднее по историческим годам
                        first_row = season_start_row + 2
                        last_row = season_rows_list[-1] + 1
                        cell_range = f'{month_col_letter}{first_row}:{month_col_letter}{last_row}'
                        formula = f'=AVERAGE({cell_range})'
                        worksheet.write(forecast_season_row, season_start_col + col_idx, formula, forecast_data_format)
                
                forecast_season_rows[forecast_year] = forecast_season_row
                season_row += 1
            
            # ============= ПРОГНОЗНЫЕ ЗНАЧЕНИЯ =============
            base_col_letter = chr(ord('A') + base_col + 1)
            
            for forecast_year in self.forecast_years:
                forecast_row_idx = df_copy[df_copy['Год'] == forecast_year].index[0]
                forecast_data_row = data_start_row + forecast_row_idx
                forecast_season_row = forecast_season_rows[forecast_year]
                
                # Заполняем прогнозом только те месяцы, которые пустые
                for col_idx in range(1, len(df_copy.columns)):
                    current_value = df_copy.iloc[forecast_row_idx, col_idx]
                    
                    if pd.isna(current_value):
                        # Только пустые ячейки заполняем прогнозом
                        month_col_letter = chr(ord('B') + col_idx)
                        season_coeff_cell = f'{month_col_letter}{forecast_season_row + 1}'
                        formula = f'={season_coeff_cell} * ${base_col_letter}${base_row + 1}'
                        worksheet.write(forecast_data_row, season_start_col + col_idx, formula, forecast_data_format)
                
                # ИТОГО для прогноза (смешанное: факт + прогноз)
                start_letter = chr(ord('B') + 2)
                end_letter = chr(ord('B') + len(df_copy.columns) - 1)
                total_formula = f'=AVERAGE({start_letter}{forecast_data_row + 1}:{end_letter}{forecast_data_row + 1})'
                worksheet.write(forecast_data_row, total_col, total_formula, forecast_data_format)
        
        # ============= ГРАФИК =============
        chart = workbook.add_chart({'type': 'line'})
        
        excel_header_row = header_row + 1
        excel_data_start = data_start_row + 1
        
        col_start = 2
        col_end = 13
        
        start_col_letter = chr(ord('A') + col_start)
        end_col_letter = chr(ord('A') + col_end)
        
        categories = f"='{channel_name}'!${start_col_letter}${excel_header_row}:${end_col_letter}${excel_header_row}"
        series_added = 0
        
        colors = ['#696969', '#FF7F00', '#00C957', '#53868B', '#0000FF']
        
        # Добавляем исторические ряды (годы, не требующие прогноза)
        year_index = 0
        for row_idx in range(len(df_copy)):
            year = df_copy.loc[row_idx, 'Год']
            
            if year not in self.forecast_years and df_copy.iloc[row_idx, 1:].notna().any():
                excel_row = excel_data_start + row_idx
                values = f"='{channel_name}'!${start_col_letter}${excel_row}:${end_col_letter}${excel_row}"
                color = colors[year_index % len(colors)]
                
                chart.add_series({
                    'name': str(year),
                    'categories': categories,
                    'values': values,
                    'marker': {'type': 'circle', 'size': 5, 'fill': {'color': color}, 'border': {'color': color, 'width': 1}},
                    'line': {'width': 2.5, 'color': color},
                })
                series_added += 1
                year_index += 1


        # Добавляем прогнозные ряды (годы, для которых строим прогноз)
        forecast_colors = ['#DC143C', '#BF3EFF']
        
        # Добавляем прогнозные ряды (годы, для которых строим прогноз)
        for forecast_idx, forecast_year in enumerate(self.forecast_years):
            forecast_row_idx = df_copy[df_copy['Год'] == forecast_year].index[0]
            excel_row = excel_data_start + forecast_row_idx
            values = f"='{channel_name}'!${start_col_letter}${excel_row}:${end_col_letter}${excel_row}"

            forecast_color = None
            # Выбираем цвет для прогнозного года
            if forecast_idx == 0:
                forecast_color = forecast_colors[0]
            elif forecast_idx == 1:
                forecast_color = forecast_colors[1]
            
            chart.add_series({
                'name': f'{forecast_year} (прогноз)',
                'categories': categories,
                'values': values,
                'marker': {'type': 'circle', 'size': 6, 'fill': {'color': forecast_color}, 'border': {'color': forecast_color, 'width': 1}},
                'line': {'width': 2.5, 'dash_type': 'dash', 'color': forecast_color},
            })
            series_added += 1
        
        if series_added > 0:
            chart.set_title({'name': f'{channel_name} {bca_name}', 'name_font': {'size': 14, 'bold': True}})
            chart.set_x_axis({'name': 'Месяцы', 'name_font': {'size': 12, 'bold': True}})
            chart.set_y_axis({'name': 'Значение', 'name_font': {'size': 12, 'bold': True}, 'num_format': '0.000'})
            chart.set_legend({'position': 'bottom', 'font': {'size': 10, 'name': 'Arial'}, 'line': {'width': 0}, 'shadow': False})
            
            worksheet.insert_chart(start_row, 18, chart, {'x_scale': 1.66, 'y_scale': 1.6})
            
            gray_format = workbook.add_format({
                'bold': False, 'align': 'center', 'valign': 'vcenter', 'border': 0,
                'bg_color': '#808080', 'font_color': '#FFFFFF'
            })
            
            gray_row = season_row + 8
            for col_idx in range(0, jan_to_year_col + 10):
                worksheet.write(gray_row, col_idx, '', gray_format)
        
        return gray_row + 2
    

    def pipeline(
            self, type_of_grouping: str, 
            company_filter: str, 
            statistics: list
            ):
        """
            Полный пайплайн
        """
        # Шаг 1. Генерация периодов для выгрузки данных
        periods = self.generate_periods()

        # Шаг 2. Выгрузка данных из БД
        results = self.acquistare_data(type_of_grouping, company_filter, statistics, periods) 

        # Шаг 3. Преобразование данных для построения прогноза
        transformed_results = self.form_output_for_forecast(results)

        # Шаг 4. Отбираю только те каналы и БЦА, которые требуется спрогнозировать.
        print('\n')
        print(Color.BOLD + Color.ORANGE + '=== НАЧИНАЮ ОТБОР КАНАЛОВ И БЦА. ПОЖАЛУЙСТА, ПОДОЖДИТЕ ... ===' + Color.END)
        filtered_data = self.filter_data_by_targets(transformed_results)

        # Шаг 5. Построение прогноза и запись в выходной файл
        # создаем ExcelWriter один раз для всех листов
        with pd.ExcelWriter(self.output_file, engine = 'xlsxwriter') as writer:
            workbook = writer.book
            
            for channel_name in filtered_data.keys():
                # Создаем лист для каждого канала
                worksheet = workbook.add_worksheet(channel_name)
                
                workbook.use_zip64()
                
                # Название канала в ячейке A1
                format_a1 = workbook.add_format({
                    'font_name': 'Arial', 'font_size': 12, 'bold': True, 'italic': True,
                    'font_color': '#00008B', 'align': 'center', 'valign': 'vcenter'
                })
                worksheet.write('A1', channel_name, format_a1)
                
                # Создаем отчеты для каждого БЦА
                current_row = 2  # Начинаем с A3 (индекс 2)
                
                for bca_name, df in filtered_data[channel_name].items():
                    #print(f"\nСоздаю отчет для {bca_name} на листе {channel_name}...")
                    current_row = self.create_bca_report(
                        writer, worksheet, df, bca_name, channel_name, 
                        current_row
                    )
                
                # Настройка ширины столбцов для текущего листа
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:B', 10)
                worksheet.set_column('C:O', 12)
                worksheet.set_column('P:P', 12)
                worksheet.set_column('Q:Q', 14)
                worksheet.set_column('R:Z', 15)
            
            print(Color.BOLD + Color.GREEN + f"\n ✅ 🏁 Файл {self.output_file} успешно создан с {len(filtered_data)} листами!" + Color.END)
        
        return filtered_data



class KUSChannelNearestNeighborPredictor:
    """
        Класс для поиска каналов-аналогов с целью формирования рекомендаций 
        для прогнозирования показателей КУС
    """
    def __init__(self, channel_name: str, thematic_channels_guidebook_path: str, forecast_file: str):
        self.channel_name = channel_name
        self.thematic_channels_guidebook_path = thematic_channels_guidebook_path
        self.forecast_file = forecast_file

        print(Color.BOLD + 'Пожалуйста, задайте ' + Color.END + Color.BLUE + f'жанр, целевую аудиторию и географию.' + Color.END)

        print(' ● В качестве ' + Color.BOLD + 'ЖАНРА' + Color.END + ' необходимо задать одно из следующих: ' + \
              Color.DARK_GREEN + 'детские, документалистика, животные, кино, музыка, новости, патриотическое, развлекательное, спорт.' + Color.END)
        
        print(' ● В качестве ' + Color.BOLD + 'ЦЕЛЕВОЙ АУДИТОРИИ' + Color.END + ' необходимо задать одно из следующих: ' + \
              Color.DARK_ORANGE + 'взрослые, дети, женщины, мужчины.' + Color.END)
        print(' ● В качестве ' + Color.BOLD + 'ГЕОГРАФИИ' + Color.END + ' необходимо задать одно из следующих: ' + \
              Color.DODGER_BLUE + 'россия, ссср, зарубежные.' + Color.END)

        print('\n')
        print(Color.MAGENTA + 'Если вы не знаете, что задать в качестве жанра / целевой аудитории / географии, поставьте прочерк "-"' + Color.END)
        print('\n')


        self.GENRE_OPTIONS = {
            1: 'Детское',
            2: 'Документальное',
            3: 'Животные',
            4: 'Здоровье',
            5: 'Кино',
            6: 'Кулинария',
            7: 'Музыка',
            8: 'Новости',
            9: 'Патриотическое',
            10: 'Познавательное',
            11: 'Развлекательное',
            12: 'Сериалы',
            13: 'Спорт',
            14: '-'
        }

        self.AUDIENCE_OPTIONS = {
            1: 'Взрослые', 2: 'Дети', 3: 'Женщины', 4: 'Мужчины', 5: '-'
        }

        self.GEOGRAFIC_OPTIONS = {
            1: 'Россия', 2: 'СССР', 3: 'Зарубежные', 4: '-'
        }
        

        self.thematic_channels_data = pd.read_excel(self.thematic_channels_guidebook_path)

        for column in self.thematic_channels_data.columns[0:6]:
            self.thematic_channels_data[column] = self.thematic_channels_data[column].str.lower()
        

    def option_suggestion(self):
        """
            Генерация вариантов для прогнозирования
        """

        # Генерация Жанра (множественный выбор)
        print('Выберите ' + Color.BOLD + 'ЖАНР' + Color.END + ' из списка:')
        print('Выберите ' + Color.BOLD + 'ЖАНР' + Color.END + ' из списка (введите номера через запятую, если требуется выбрать несколько. Например: 1,3,5):')
        for key, value in self.GENRE_OPTIONS.items():
            print(f'{key}. {value}')

        genre_input = input('Введите номера: ')
        genre_choices = [int(x.strip()) for x in genre_input.split(',')]
        
        # Проверка на наличие опции "-" (все жанры)
        if self.GENRE_OPTIONS[9] in [self.GENRE_OPTIONS[choice] for choice in genre_choices]:
            self.movie_genres = [self.GENRE_OPTIONS[9]]  # Если выбран "-", то только он
        else:
            self.movie_genres = [self.GENRE_OPTIONS[choice] for choice in genre_choices if choice in self.GENRE_OPTIONS]
        print('\n')

        # Генерация Географии
        print('Выберите ' + Color.BOLD + 'ГЕОГРАФИЮ' + Color.END + ' из списка: (введите номера через запятую, если требуется выбрать несколько. Например: 1,3,5):')
        for key, value in self.GEOGRAFIC_OPTIONS.items():
            print(f'{key}. {value}')

        geografic_input = input('Введите номера: ')
        geografic_choices = [int(x.strip()) for x in geografic_input.split(',')]
        
        # Проверка на наличие опции "-" (все географии)
        if self.GEOGRAFIC_OPTIONS[4] in [self.GEOGRAFIC_OPTIONS[choice] for choice in geografic_choices]:
            self.geografics = [self.GEOGRAFIC_OPTIONS[4]]  # Если выбран "-", то только он
        else:
            self.geografics = [self.GEOGRAFIC_OPTIONS[choice] for choice in geografic_choices if choice in self.GEOGRAFIC_OPTIONS]
        print('\n')

        # Генерация Аудитории
        print('Выберите ' + Color.BOLD + 'АУДИТОРИЮ' + Color.END + ' из списка: (введите номера через запятую, если требуется выбрать несколько. Например: 1,3,5):')
        for key, value in self.AUDIENCE_OPTIONS.items():
            print(f'{key}. {value}')

        audience_input = input('Введите номера: ')
        audience_choices = [int(x.strip()) for x in audience_input.split(',')]
        
        # Проверка на наличие опции "-" (все аудитории)
        if self.AUDIENCE_OPTIONS[5] in [self.AUDIENCE_OPTIONS[choice] for choice in audience_choices]:
            self.audiences = [self.AUDIENCE_OPTIONS[5]]  # Если выбран "-", то только он
        else:
            self.audiences = [self.AUDIENCE_OPTIONS[choice] for choice in audience_choices if choice in self.AUDIENCE_OPTIONS]
        print('\n')

        print(Color.VIOLET + f"В качестве ")
        print(f'•  Жанра выбрано:                {", ".join(self.movie_genres)}')
        print(f'•  Географии выбрано:            {", ".join(self.geografics)}')
        print(f'•  Целевой аудитории выбрано:    {", ".join(self.audiences)}' + Color.END)

        #mask = None
#
        #if self.movie_genre == self.GENRE_OPTIONS[9]:
        #    mask = (self.thematic_channels_data['Аудитория'].str.contains(self.audience.lower(), case = False, na = False)) & \
        #           (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False))
        #
        #elif self.audience == self.AUDIENCE_OPTIONS[5]:
        #    mask = (self.thematic_channels_data['Жанр'].str.contains(self.movie_genre.lower(), case = False, na = False)) & \
        #           (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False))
        #
        #elif self.geografic == self.GEOGRAFIC_OPTIONS[4]:
        #    mask = (self.thematic_channels_data['Жанр'].str.contains(self.movie_genre.lower(), case = False, na = False)) & \
        #           (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False))
        #
        #else:
        #    mask = (self.thematic_channels_data['Аудитория'].str.contains(self.audience.lower(), case = False, na = False)) & \
        #           (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False)) & \
        #           (self.thematic_channels_data['Жанр'].str.contains(self.movie_genre.lower(), case = False, na = False))
#
        #result_df = self.thematic_channels_data[mask].reset_index(drop = True)

        # Формирование маски для фильтрации
        mask = None

        # Фильтрация по жанрам (если не выбран "-")
        if self.GENRE_OPTIONS[9] in self.movie_genres:
            # Если выбран "-", пропускаем фильтр по жанру
            pass
        else:
            # Собираем все жанры через OR
            genre_condition = False
            for genre in self.movie_genres:
                genre_condition |= self.thematic_channels_data['Жанр'].str.contains(genre.lower(), case=False, na=False)
            mask = genre_condition

        # Фильтрация по географии (если не выбран "-")
        if self.GEOGRAFIC_OPTIONS[4] in self.geografics:
            # Если выбран "-", пропускаем фильтр по географии
            pass
        else:
            geografic_condition = False
            for geografic in self.geografics:
                geografic_condition |= self.thematic_channels_data['География'].str.contains(geografic.lower(), case=False, na=False)
            
            if mask is None:
                mask = geografic_condition
            else:
                mask &= geografic_condition

        # Фильтрация по аудитории (если не выбрана "-")
        if self.AUDIENCE_OPTIONS[5] not in self.audiences:
            audience_condition = False
            for audience in self.audiences:
                audience_condition |= self.thematic_channels_data['Аудитория'].str.contains(audience.lower(), case=False, na=False)
            
            if mask is None:
                mask = audience_condition
            else:
                mask &= audience_condition

        # Если ничего не выбрано (все "-"), то показываем все
        if mask is None:
            mask = pd.Series([True] * len(self.thematic_channels_data))

        result_df = self.thematic_channels_data[mask].reset_index(drop=True)

        print('\n')
        if len(result_df) == 0:
            print(Color.BOLD + Color.RED + f'Для канала {self.channel_name} не нашлось похожих. Рекомендую в качестве прогноза взять среднее по всему ВК.')
        else:
            print(Color.BOLD + Color.MAROON + f'Для канала {self.channel_name} сгенерировал следующие варианты. Пожалуйста, ознакомьтесь: ' + Color.END)

        return result_df
    

    def select_special_columns(self, df, statistic: str):
        """
            Вспомогательный метод для отбора интересующих столбцов
        """
        months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 
          'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь', 'ГОД']

        # Отбираем колонки с месяцами
        month_cols = [col for col in df.columns if any(m in col for m in months)]

        # Группируем по названию месяца и берём максимальный суффикс
        selected_cols = []
        for month in months:
            # Все колонки для этого месяца
            cols = [col for col in month_cols if col.startswith(month)]
            if cols:
                # Берём ту, у которой самый большой суффикс
                max_col = max(cols, key = lambda x: int(x.split('.')[1]) if '.' in x else 0)
                selected_cols.append(max_col)

        df_new = df[['Канал+ЦА', 'Канал', 'ЦА'] + selected_cols]

        rename_dict = {col: month for col, month in zip(selected_cols, months)}
        df_new.rename(columns = rename_dict, inplace = True)

        df_new = df_new[['Канал+ЦА', 'Канал', 'ЦА', 'ГОД']]

        str_cols = ['Канал+ЦА', 'Канал', 'ЦА']
        for col in str_cols:
            df_new[col] = df_new[col].str.lower()
        
        df_new.rename(columns = {'ГОД': f"Прогноз {statistic}"}, inplace = True)

        # Заменяем строковые варианты пустоты на реальный np.nan
        df_new.replace(['NaN', 'nan', 'None', '', ' '], np.nan, inplace = True)
        # Теперь удаляем строки, где есть хоть один np.nan
        df_new.dropna(inplace = True)

        return df_new

    

    def read_full_forecast_file(self, bca_list: list):
        """
            Метод для чтения фулл-считалки. Считывается лист с КУС
        """
        # Чтение данных из файла с прогнозом КУС
        kus_df = pd.read_excel(self.forecast_file, sheet_name = 'КУС', skiprows = 1)

        # Чтение данных из файла с прогнозом КУЧ
        kuch_df = pd.read_excel(self.forecast_file, sheet_name = 'КУЧ', skiprows = 1)

        for df in [kus_df, kuch_df]:
            df.rename(columns = {
                                    'Unnamed: 0': 'Канал+ЦА', 
                                    'Unnamed: 1': 'Канал', 
                                    'Unnamed: 2': 'ЦА'
                                    }, inplace = True)
        
        forecast_kus = self.select_special_columns(kus_df, 'КУС')
        forecast_kuch = self.select_special_columns(kuch_df, 'КУЧ')

        full_df = pd.merge(forecast_kus, forecast_kuch, on = ['Канал+ЦА', 'Канал', 'ЦА'], how = 'inner')
        
        bca_list_lowered = []
        for bca in bca_list:
            bca_list_lowered.append(bca.lower())
        pattern = '|'.join(bca_list_lowered)

        mask = full_df['ЦА'].str.contains(pattern, case = False, na = False)
        full_df_filtered = full_df[mask].reset_index(drop = True) 
        return full_df_filtered
    

    def neighbor_pipeline(self, bca_list: list):
        """
            Пайплайн для генерации прогноза КУС и КУЧ на основании похожих каналов
        """
        bca_list_lowered = []
        for bca in bca_list:
            bca_list_lowered.append(bca.lower())

        # Генерация наиболее похожих вариантов
        neighbor_df = self.option_suggestion()

        # Чтение прогнозных значений из файла "Данные ВРК full" по КУС и КУЧ
        forecast_df = self.read_full_forecast_file(bca_list)

        variants_df = pd.merge(forecast_df, neighbor_df, on = ['Канал'], how = 'inner')
        variants_df = variants_df[[
            'Канал', 'ЦА', 'Прогноз КУС', 'Прогноз КУЧ'
        ]]

        if len(variants_df) == 0:
            forecast_results = []
            for bca in bca_list_lowered:
                df = forecast_df[forecast_df['ЦА'] == bca].reset_index(drop = True)

                kus_forecast = df['Прогноз КУС'].mean()
                kuch_forecast = df['Прогноз КУЧ'].mean()

                forecast_df = pd.DataFrame(
                    [[self.channel_name, bca, kus_forecast, kuch_forecast]],
                    columns = ['Канал', 'ЦА', 'Прогноз КУС', 'Прогноз КУЧ']
                )
                forecast_results.append(forecast_df)
            
            forecast_result_df = pd.concat(forecast_results).reset_index(drop = True)
            return pd.DataFrame(), {}, forecast_result_df
            
        else:
            results_dict = {}
            forecast_results = []
            for bca in bca_list_lowered:
                df = variants_df[variants_df['ЦА'] == bca].reset_index(drop = True)

                kus_forecast = df['Прогноз КУС'].mean()
                kuch_forecast = df['Прогноз КУЧ'].mean()

                forecast_df = pd.DataFrame(
                    [[self.channel_name, bca, kus_forecast, kuch_forecast]],
                    columns = ['Канал', 'ЦА', 'Прогноз КУС', 'Прогноз КУЧ']
                )
                forecast_results.append(forecast_df)


                df['Прогноз КУС'] = df['Прогноз КУС'].astype(float).round(2)
                df['Прогноз КУЧ'] = df['Прогноз КУЧ'].astype(float).round(2)

                results_dict[bca] = df
            
            for key, value in results_dict.items():
                print(Color.GREEN + f'{key}' + Color.END)  
                print(f'{value.to_string()}')
                print('\n') 

            #print("Все отобранные каналы, которые оказываются наиболее похожими на целевой канал: ")
            neighbor_df_cleand = neighbor_df[['Канал', 'Аудитория', 'Жанр', 'География', 'Холдинг', 'Описание']]

            forecast_result_df = pd.concat(forecast_results).reset_index(drop = True)
            
            return neighbor_df_cleand, results_dict, forecast_result_df
        
        

        