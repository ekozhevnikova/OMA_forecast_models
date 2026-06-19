import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import xlsxwriter
import re

from mediascope_api.mediavortex import catalogs as cwc
cats = cwc.MediaVortexCats()

from OMA_tools.regions.data_extraction.task_builder import BaseDataService
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
            targets: dict, 
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
                targets: dict
                    Словарь из ЦА
                target_data_to_forecast: dict
                    Словарь из Целевых Каналов и БЦА, которые попросили спрогнозировать
                forecast_years: list
                    Список из годов, для которых будем строить прогноз. Это может быть и 1 год, и 2. И, например, остаток текущего года
                output_file: str
                    Название/Путь к выходному файлу в формате .xlsx
        """
        self.start_date = start_date
        self.stop_date = stop_date
        self.targets = targets
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
        return channels_dict_filtered
    

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
        
        self.periods = []
        
        # Генерируем периоды по годам
        current = start
        while current <= stop:
            year_start = current
            year_end = datetime(current.year, 12, 31)
            
            # Если конечная дата раньше конца года, используем stop_date
            if year_end > stop:
                year_end = stop
            
            self.periods.append([ (year_start.strftime('%Y-%m-%d'), year_end.strftime('%Y-%m-%d')) ])
            
            # Переходим к следующему году
            current = datetime(current.year + 1, 1, 1)
        return self.periods
    


    def acquistare_data(
        self, type_of_grouping: str, company_filter: str, 
        time_filter: str, statistics: list
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


        print(Color.BOLD + Color.VIOLET + '=== НАЧИНАЮ ВЫГРУЗКУ ДАННЫХ ===' + Color.END)

        final_results = []

        for date_filter in self.periods:


            # Формируем задачи в формате json
            tasks = BaseDataService._build_timeband_common_params(
                                                            date_filter = date_filter, company_filter = company_filter, 
                                                            basedemo_filter = None, regions_id = None,          # работаем в Федеральной Базе
                                                            targets = self.targets, time_filter = time_filter, 
                                                            statistics = statistics, slices = slices, 
                                                            sortings = sortings, options = self.options,
                                                            location_filter = self.location_filter, weekday_filter = self.weekday_filter,
                                                            daytype_filter = self.daytype_filter, targetdemo_filter = self.targetdemo_filter
                                                        )
            # Отправляем задачи на расчет
            df = BaseDataService._execute_tasks(tasks)
        
            final_results.append(df)
        
        general_result = pd.concat(final_results).reset_index(drop = True)
        general_result['tvCompanyName'] = general_result['tvCompanyName'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))

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
        transformed_dict = {}
        
        for channel, bca_dict in dict_data.items():
            
            by_bca = {}
            for bca, data in bca_dict.items():
                if 'Месяц' not in data.columns:
                    raise ValueError(
                            "Отсутствует столбец с названием 'Месяц'. Я не смогу трансформировать таблицы."
                    )
                    
                data = data.drop(['БЦА', 'Канал'], axis = 1)
                data['Год'] = pd.to_datetime(data['Месяц']).dt.year
                data['month'] = pd.to_datetime(data['Месяц']).dt.month
                data['month'].replace(self.MONTHS, inplace = True)
                
                data_transformed = data.pivot_table(
                    index = 'Год', 
                    columns = 'month', 
                    values = 'Share'
                ).reset_index()
                
                # Переименуем столбцы (уберем имя 'month' и сделаем нормальные названия)
                data_transformed.columns.name = None
                
                # Если нужно отсортировать месяцы в правильном порядке
                month_order = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 
                            'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
                
                data_transformed = data_transformed[['Год'] + month_order]

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
            # Приводим названия БЦА к нижнему регистру, чтобы было легче сопоставлять
            lowercase_bcas_to_keep = [x.lower() for x in bcas_to_keep]

            # Проверяем, есть ли такой канал в исходных данных
            if channel not in all_results:
                print(f" 🚩 WARNING: Канал '{channel}' не найден в исходных данных. Проверьте название. Оно должно совпадать с названием Mediascope.")
                continue

            # Получаем словарь БЦА для этого канала
            channel_data = all_results[channel]
            # Приводим ключи к нижнему регистру, чтобы проще было совмещать
            lower_channel_data = {k.lower(): v for k, v in channel_data.items()}

            # Отбираем только нужные БЦА
            filtered_channel_data = {}
            for bca in lowercase_bcas_to_keep:
                if bca in lower_channel_data:
                    filtered_channel_data[bca] = lower_channel_data[bca]
                else:
                    print(f" ⏭️ Для канала '{channel}' БЦА '{bca}' не просят спрогнозировать! Я не буду добавлять её в отчёт.")
            
            # Добавляем канал в результат, только если есть хотя бы одна БЦА
            if filtered_channel_data:
                filtered_data[channel] = filtered_channel_data

        print(Color.ROYAL_BLUE + f"🚀 Итог: отобрано {len(filtered_data)} каналов из {len(all_results)}" + Color.END)
    
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
                worksheet.write(forecast_season_row, season_start_col, int(forecast_year), forecast_year_format)
                
                for col_idx in range(1, len(df_copy.columns)):
                    month_col_letter = chr(ord('B') + col_idx)
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
            company_filter: str, time_filter: str, 
            statistics: list
            ):
        """
            Полный пайплайн
        """
        # Шаг 1. Генерация периодов для выгрузки данных
        self.periods = self.generate_periods()

        # Шаг 2. Выгрузка данных из БД
        results = self.acquistare_data(type_of_grouping, company_filter, time_filter, statistics) 

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
            1: 'Детские',
            2: 'Документалистика',
            3: 'Животные',
            4: 'Кино',
            5: 'Музыка',
            6: 'Патриотическое',
            7: 'Развлекательное',
            8: 'Спорт',
            9: '-'
        }

        self.AUDIENCE_OPTIONS = {
            1: 'Взрослые', 2: 'Дети', 3: 'Женщины', 4: 'Мужчины', 5: '-'
        }

        self.GEOGRAFIC_OPTIONS = {
            1: 'Россия', 2: 'СССР', 3: 'Зарубежные', 4: '-'
        }
        

        self.thematic_channels_data = pd.read_excel('Каналы Тематическое ТВ.xlsx')

        for column in self.thematic_channels_data.columns[0:6]:
            self.thematic_channels_data[column] = self.thematic_channels_data[column].str.lower()
        

    def option_suggestion(self):
        """
            Генерация вариантов для прогнозирования
        """

        # Генерация Жанра
        print('Выберите ' + Color.BOLD + 'ЖАНР' + Color.END + ' из списка:')
        for key, value in self.GENRE_OPTIONS.items():
            print(f'{key}. {value}')

        genre_choice = int(input('Введите номер: '))
        self.movie_genre = self.GENRE_OPTIONS[genre_choice]
        print('\n')

        # Генерация Географии
        print('Выберите ' + Color.BOLD + 'ГЕОГРАФИЮ' + Color.END + ' из списка:')
        for key, value in self.GEOGRAFIC_OPTIONS.items():
            print(f'{key}. {value}')

        geografic_choice = int(input('Введите номер: '))
        self.geografic = self.GEOGRAFIC_OPTIONS[geografic_choice]
        print('\n')

        # Генерация Аудитории
        print('Выберите ' + Color.BOLD + 'АУДИТОРИЮ' + Color.END + ' из списка:')
        for key, value in self.AUDIENCE_OPTIONS.items():
            print(f'{key}. {value}')

        audience_choice = int(input('Введите номер: '))
        self.audience = self.AUDIENCE_OPTIONS[audience_choice]
        print('\n')

        print(Color.VIOLET + f"В качестве ")
        print(f'•  Жанра выбрано:                {self.movie_genre}')
        print(f'•  Географии выбрано:            {self.geografic}')
        print(f'•  Целевой аудитории выбрано:    {self.audience}' + Color.END)

        mask = None

        if self.movie_genre == self.GENRE_OPTIONS[9]:
            mask = (self.thematic_channels_data['Аудитория'].str.contains(self.audience.lower(), case = False, na = False)) & \
                   (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False))
        
        elif self.audience == self.AUDIENCE_OPTIONS[5]:
            mask = (self.thematic_channels_data['Жанр'].str.contains(self.movie_genre.lower(), case = False, na = False)) & \
                   (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False))
        
        elif self.geografic == self.GEOGRAFIC_OPTIONS[4]:
            mask = (self.thematic_channels_data['Жанр'].str.contains(self.movie_genre.lower(), case = False, na = False)) & \
                   (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False))
        
        else:
            mask = (self.thematic_channels_data['Аудитория'].str.contains(self.audience.lower(), case = False, na = False)) & \
                   (self.thematic_channels_data['География'].str.contains(self.geografic.lower(), case = False, na = False)) & \
                   (self.thematic_channels_data['Жанр'].str.contains(self.movie_genre.lower(), case = False, na = False))

        result_df = self.thematic_channels_data[mask].reset_index(drop = True)

        print('\n')
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
                max_col = max(cols, key=lambda x: int(x.split('.')[1]) if '.' in x else 0)
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

        results_dict = {}
        for bca in bca_list_lowered:
            df = variants_df[variants_df['ЦА'] == bca].reset_index(drop = True)
            df['Прогноз КУС'] = df['Прогноз КУС'].astype(float).round(2)
            df['Прогноз КУЧ'] = df['Прогноз КУЧ'].astype(float).round(2)
            results_dict[bca] = df
        
        for key, value in results_dict.items():
            print(Color.GREEN + f'{key}' + Color.END)  
            print(f'{value.to_string()}')
            print('\n') 

        #print("Все отобранные каналы, которые оказываются наиболее похожими на целевой канал: ")
        neighbor_df_cleand = neighbor_df[['Канал', 'Аудитория', 'Жанр', 'География', 'Холдинг', 'Описание']]
        
        return neighbor_df_cleand, results_dict
        
        

        