import time
import os
import numpy as np
import xlsxwriter
from openpyxl import load_workbook
import pandas as pd
import subprocess
from datetime import datetime
import gc

from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time

from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.regions.data_extraction.task_builder import *
from OMA_tools.thematic.new_channels import *


class ThematicShare:
    """
        Класс для выгрузки статистик Тематического ТВ
    """
    def __init__(self, date_filter: list, statistics: list, channels_guide_file: str):
        self.date_filter = date_filter
        self.statistics = statistics
        self.channels_guide_file = channels_guide_file

        self.TIME_FILTER = 'timeBand1 >= 60000 AND timeBand1 < 260000' # 06:00:00 - 26:00:00
        self.CHILDREN_TIME_FILTER = 'timeBand1 >= 60000 AND timeBand1 < 220000' # 06:00:00 - 22:00:00

        self.SLICES = ['tvCompanyName'] #Разбиваем по телекомпаниям

        # Задаем условия сортировки: телекомпания (от а до я)
        self.SORTINGS = {"tvCompanyName":"ASC"}

        # Задаем опции расчета
        self.OPTIONS = {
            "kitId": 4, #TV Index Plus All Russia
            "totalType": "TotalChannels", #Расчет Share от Total Channels. 
            #Для расчета от Измеряемого Тематического поменять на: TotalChannelsThem
            "useNbd": False #Расчет накопленного охвата без nbd коррекции
        }

        # Задаем переменные, которые NAN
        self.WEEKDAY_FILTER = None #фильтр на дни недели
        self.DAYTYPE_FILTER = None #фильтр на тип дня
        self.BASEDEMO_FILTER = None #ЦА
        self.TARGETDEMO_FILTER = None #доп фильтр на ЦА для расчета Affinity Index
        self.LOCATION_FILTER = None #место просмотра, если None => Дом, Дача
        self.DAYTYPE_FILTER = None


        # Словарь с целевыми аудиториями: ключ - название переменной (target), значение - ее синтаксис (syntax)
        self.TARGETS = {
            'ВСЕ 25-49': 'age >= 25 AND age <= 49',
            'М 25-49': 'age >= 25 AND age <= 49 AND sex = 1',
            'Ж 25-49': 'age >= 25 AND age <= 49 AND sex = 2'
        }

        self.channels_guide_full = pd.read_excel(self.channels_guide_file)

        # Генерируем компании для выгрузки
        channels_id = np.array(self.channels_guide_full['ID канала Mediascope']).tolist()
        data_id = list(map(lambda x: str(x), channels_id))
        self.company_filter = f'tvCompanyId IN ({", ".join(data_id)})'

        need_columns = ['Канал', 'Название канала в VIMB', 'ЕРК', 'ЖРК', 'МРК', 'ДРК']
        self.channels_guide = self.channels_guide_full[need_columns]
    

    def make_api_calculation(
        self, 
        children_basedemo_filter = 'age >= 4 and age <= 40',

        ):
        """
            Метод для выгрузки данных из Базы Данных. Отдельно выгружаются старшие аудитории и отдельнао детская, тк у нее другой слот.
        """
        # Берём первую дату из первого кортежа
        date_str = self.date_filter[0][0]

        # Преобразуем в объект datetime
        dt = datetime.strptime(date_str, '%Y-%m-%d')

        # Форматируем как "Месяц Год" на русском
        months_ru = {
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
        }

        new_column_name = f"{months_ru[dt.month]} {dt.year}"


        # Формируем задачи в формате json
        tasks = BaseDataService._build_timeband_common_params(
                                                        date_filter = self.date_filter, company_filter = self.company_filter, 
                                                        basedemo_filter = None, regions_id = None,          # работаем в Федеральной Базе
                                                        targets = self.TARGETS, time_filter = self.TIME_FILTER, 
                                                        statistics = self.statistics, slices = self.SLICES, 
                                                        sortings = self.SORTINGS, options = self.OPTIONS,
                                                        location_filter = self.LOCATION_FILTER, weekday_filter = self.WEEKDAY_FILTER,
                                                        daytype_filter = self.DAYTYPE_FILTER, targetdemo_filter = self.TARGETDEMO_FILTER
                                                    )
        # Отправляем задачи на расчет
        df = BaseDataService._execute_tasks(tasks)

        # ВЫГРУЗКА ДАННЫХ ДЛЯ ДЕТСКОЙ АУДИТОРИИ, У КОТОРОЙ СВОЙ ВКУС
        child_tasks = BaseDataService._build_timeband_common_params(
                                                        date_filter = self.date_filter, company_filter = self.company_filter, 
                                                        basedemo_filter = children_basedemo_filter, regions_id = None,          # работаем в Федеральной Базе
                                                        targets = None, time_filter = self.CHILDREN_TIME_FILTER, 
                                                        statistics = self.statistics, slices = self.SLICES, 
                                                        sortings = self.SORTINGS, options = self.OPTIONS,
                                                        location_filter = self.LOCATION_FILTER, weekday_filter = self.WEEKDAY_FILTER,
                                                        daytype_filter = self.DAYTYPE_FILTER, targetdemo_filter = self.TARGETDEMO_FILTER
                                                    )
        # Отправляем задачи на расчет
        child_df = BaseDataService._execute_tasks(child_tasks)
        child_df['prj_name'] = child_df['prj_name'].replace('Total. Ind', 'ВСЕ 4-40')

        result_df = pd.concat([df, child_df])
        result_df['tvCompanyName'] = result_df['tvCompanyName'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))
        data_output = result_df.rename(columns = {'prj_name': 'ЦА', 'tvCompanyName': 'Канал'})
        data_output.rename(columns = {f"{self.statistics[0]}": new_column_name}, inplace = True)

        # Итоговый словарь с выгруженными данными
        erk = data_output[data_output['ЦА'] == 'ВСЕ 25-49'].reset_index(drop = True)
        women = data_output[data_output['ЦА'] == 'Ж 25-49'].reset_index(drop = True)
        men = data_output[data_output['ЦА'] == 'М 25-49'].reset_index(drop = True)
        children = data_output[data_output['ЦА'] == 'ВСЕ 4-40'].reset_index(drop = True)

        result_output = {
            'ЕРК': erk[['Канал', new_column_name]] if len(erk) != 0 else pd.DataFrame(),
            'ЖРК': women[['Канал', new_column_name]] if len(women) != 0 else pd.DataFrame(),
            'МРК': men[['Канал', new_column_name]] if len(men) != 0 else pd.DataFrame(),
            'ДРК': children[['Канал', new_column_name]] if len(children) != 0 else pd.DataFrame()
        }


        dict_data = {}
        for vk in result_output.keys():
            if len(vk) > 0:
                guide_df = self.channels_guide[self.channels_guide[vk] == 1].reset_index(drop = True)
                guide_df = guide_df[['Канал', 'Название канала в VIMB', f'{vk}']]

                df = result_output[vk]

                merged_df = pd.merge(guide_df, df, on = 'Канал', how = 'left')
                merged_df = merged_df.fillna(0)

                merged_df = merged_df.drop(['Канал', f'{vk}'], axis = 1)
                merged_df.rename(columns = {'Название канала в VIMB': 'Канал'}, inplace = True)
                merged_df = merged_df[['Канал', new_column_name]]

                # Каналы, которые не нашлись
                not_found = guide_df[~guide_df['Канал'].isin(df['Канал'])]['Канал'].tolist()

                print(f'Всего каналов в {vk} равно {len(guide_df)}. Найдено соответствие {len(merged_df)} каналам в {vk}. ')
                #print(f'Не найдено каналов: {len(not_found)}')
                if not_found:
                    print('Список не найденных каналов:')
                    for ch in not_found:
                        print(f'  - {ch}')
                
                dict_data[vk] = merged_df
            
            else:
                dict_data[vk] = pd.DataFrame()

        return dict_data
    

    def generate_periods(self, start_date: str = '2021-01-01'):
        """
            Метод для генерации периодов, начиная с 2021 г. по месяцам
        """
        # ШАГ 1. Генерация периодов для выгрузки исторических данных для новых каналов
        # Конечная дата
        end_date = self.date_filter[0][1]

        # Преобразуем в datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Генерируем периоды по месяцам
        periods = []

        current = start
        while current <= end:
            # Начало периода
            period_start = current.strftime('%Y-%m-%d')
            
            # Конец периода - последний день текущего месяца
            # Переходим к следующему месяцу и вычитаем 1 день
            if current.month == 12:
                next_month = current.replace(year = current.year + 1, month = 1, day = 1)
            else:
                next_month = current.replace(month = current.month + 1, day = 1)
            
            period_end = (next_month - pd.Timedelta(days = 1)).strftime('%Y-%m-%d')
            
            # Если конечная дата периода выходит за границу end_date, обрезаем
            if period_end > end_date:
                period_end = end_date
            
            periods.append( [(period_start, period_end)] )
            
            # Переходим к следующему месяцу
            current = next_month

        return periods
    
    
    @staticmethod
    def to_month_year(date_str):
        """
            Метод для преобразования
        """
        dt = pd.to_datetime(date_str)
        months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
        return f"{months[dt.month-1]} {dt.year}"
    


    def get_new_channels_data(self, new_channels: set, VK_problem: str):
        """
            Метод для выгрузки данных по новому/новым каналу/каналам.
        """
        new_channels_list = list(new_channels)
        # Создаем паттерн для поиска
        pattern = '|'.join(new_channels_list)  

        # Ищем каналы, содержащие любое из этих слов
        mask = self.channels_guide_full['Название канала в VIMB'].str.contains(pattern, case = False, na = False)
        target_channels = self.channels_guide_full[mask]
        # Получаем ID новых каналов из Справочника
        target_channels = target_channels[['Название канала в VIMB', 'ID канала Mediascope']].reset_index(drop = True)
        # Создаем параметр company_filter
        company_filter = f"tvCompanyId IN ({', '.join(map(str, target_channels['ID канала Mediascope'].tolist()))})"

        # Генерируем год для прогнозирования (параметр, который требуется в классе для выгрузки новых каналов)
        from datetime import date
        today = date.today()
        future_year = today.year + 1

        # Генерируем каналы и список ЦА, которые будем выгружать
        target_dict = {}
        BCA = None

        if VK_problem == 'ЕРК':
            BCA = 'ВСЕ 25-49'
            for ch in new_channels:
                target_dict[ch] = ['Все 25-49']

        elif VK_problem == 'ЖРК':
            BCA = 'Ж 25-49'
            for ch in new_channels:
                target_dict[ch] = ['Ж 25-49']
        
        elif VK_problem == 'МРК':
            BCA = 'М 25-49'
            for ch in new_channels:
                target_dict[ch] = ['М 25-49']
        
        elif VK_problem == 'ДРК':
            BCA = 'ВСЕ 4-40'
            for ch in new_channels:
                target_dict[ch] = ['Все 4-40']


        # Получаем начальную и конечную дату
        start_date = '2021-01-01'  
        end_date = self.date_filter[0][1]    

        chiter = ForecastNewChannels(
                    start_date = start_date, stop_date = end_date, 
                    target_data_to_forecast = target_dict, forecast_years = [future_year], 
                    output_file = 'test.xlsx'
        )
        periods = chiter.generate_periods()
        result_dict = chiter.acquistare_data('by months', company_filter, self.statistics, periods)


        channels_result = []

        for channel, data_dict in result_dict.items():
            output_df = data_dict[BCA]

            # Создаем колонку с месяцем и годом
            output_df['Месяц_Год'] = output_df['Месяц'].apply(ThematicShare.to_month_year)

            # Преобразуем в datetime
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            # Получаем диапазон лет
            years_range = range(start.year, end.year + 1)

            # Pivot + fillna
            result = output_df.pivot_table(
                index = 'Канал',
                columns = 'Месяц_Год',
                values = 'Share',
                aggfunc = 'first'
            ).fillna(0).reset_index()

            # Генерируем все месяцы с 2021 по 2026
            all_months = []
            for year in years_range:
                for month in range(1, 13):
                    if year == end.year and month > 6:  # до июня 2026
                        break
                    months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                            'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
                    all_months.append(f"{months[month-1]} {year}")

            # Добавляем недостающие колонки
            for month in all_months:
                if month not in result.columns:
                    result[month] = 0.0

            # Сортируем колонки
            result = result[['Канал'] + all_months]

            channels_result.append(result)

        result_df = pd.concat(channels_result).reset_index(drop = True)
        return result_df
         

    def update_historical_data(self, output_dict: dict, historical_data_file_full: str, historical_data_file_current_year: str):
        """
            Метод для обновления файла с фактическими данными

            Параметры:
            ----------
                output_dict: dict
                    Словарь с выгрузкой из БД Mediascope
                historical_data_file_full: str
                    Путь к файлу с историческими данными с полной историей, начиная с 2021 г
                historical_data_file_current_year: str
                    Путь к файлу с историческими данными по текущему году

        """
        # Проверяем, существует ли файл
        if not os.path.exists(historical_data_file_current_year):
            print(f"📁 Файл {historical_data_file_current_year} не найден. Будет создан новый файл с текущими данными")
            return output_dict
            
        
        data_dict = File(historical_data_file_current_year).from_file(0)
        hist_dict = Dict_Operations(data_dict).replace_keys_in_dict(['ЕРК', 'ЖРК', 'МРК', 'ДРК'])

        hist_updated = {}
        hist_updated_full = {}

        for vk, hist_df in hist_dict.items():
            table = output_dict[vk]

            # Проверяем, что не добавилось новых каналов
            target_guide = self.channels_guide[['Канал', 'Название канала в VIMB', vk]]
            target_guide = target_guide[target_guide[vk] == 1].reset_index(drop = True)
            target_guide_cleaned = target_guide.drop(['Канал'], axis = 1)
            # Удаляем столбец с ВК
            guide_df = target_guide.drop([vk], axis = 1)

            new_channels = set(target_guide_cleaned['Название канала в VIMB']) - set(hist_df['Канал'])  # Новые каналы (есть в справочнике, нет в истории)

            if len(new_channels) > 0:
                print(Color.BOLD + Color.GREEN + "Найдены следующие новые каналы. Необходимо выгрузить для них историю." + Color.END)
                for ch in new_channels:
                    print(f"  - {ch}")

                ##################################### НОВЫЙ КУСОК #####################################
                 
                print('Выгружаю данные по новым каналам. Пожалуйста, подождите ...')
                new_channels_data = self.get_new_channels_data(new_channels, vk)
                new_channels_data_cut = new_channels_data.iloc[:, :-1]
                columns_order = list(new_channels_data_cut.columns[1:])

                new_channels_updated = pd.merge(new_channels_data_cut, guide_df, on = 'Канал', how = 'inner')
                new_channels_updated = new_channels_updated.drop(['Канал'], axis = 1)

                #new_channels_updated = new_channels_updated[['Название канала в VIMB'] + columns_order]
                new_channels_updated.rename(columns = {'Название канала в VIMB': 'Канал'}, inplace = True)
                new_channels_updated = new_channels_updated[['Канал'] + columns_order]

                hist_data_full = pd.read_excel(historical_data_file_full, sheet_name = vk)
                data_new = pd.concat([hist_data_full, new_channels_updated])
                data_new = data_new.sort_values(by = 'Канал').reset_index(drop = True)

                try:
                    with pd.ExcelWriter(historical_data_file_full, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
                        data_new.to_excel(writer, sheet_name = vk, index = False)
                except FileNotFoundError:
                    # Если файла нет, создаем новый
                    with pd.ExcelWriter(historical_data_file_full, engine = 'openpyxl') as writer:
                        data_new.to_excel(writer, sheet_name = vk, index = False)
                
                # Отбираем текущий год
                current_year = datetime.now().year
                # Паттерн для поиска колонок с текущим годом
                pattern = re.compile(rf'.*{current_year}$')
                # Отбираем колонки
                columns_to_keep = ['Канал'] + [col for col in new_channels_data.columns if pattern.match(col)]
                df_current_year = new_channels_data[columns_to_keep]
                df_current_year_cut = df_current_year.iloc[:, :-1]
                columns_order_curr_year = list(df_current_year_cut.columns[1:])
               
                df_current_year_updated = pd.merge(df_current_year_cut, guide_df, on = 'Канал', how = 'inner')
                df_current_year_updated = df_current_year_updated.drop(['Канал'], axis = 1)

                df_current_year_updated.rename(columns = {'Название канала в VIMB': 'Канал'}, inplace = True)
                df_current_year_updated = df_current_year_updated[['Канал'] + columns_order_curr_year]

                hist_data_current_year = pd.read_excel(historical_data_file_current_year, sheet_name = vk)
                data_new_current_year = pd.concat([hist_data_current_year, df_current_year_updated])
                data_new_current_year = data_new_current_year.sort_values(by = 'Канал').reset_index(drop = True)

                try:
                    with pd.ExcelWriter(historical_data_file_current_year, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
                        data_new_current_year.to_excel(writer, sheet_name = vk, index = False)
                except FileNotFoundError:
                    # Если файла нет, создаем новый
                    with pd.ExcelWriter(historical_data_file_current_year, engine = 'openpyxl') as writer:
                        data_new_current_year.to_excel(writer, sheet_name = vk, index = False)

                ##################################### КОНЕЦ НОВОГО КУСКА ##################################### 
                print(Color.BOLD + Color.VIOLET + 'Выгрузил исторические данные для новых каналов! Продолжаю обновление.' + Color.END)
            
            # Снова считываем данные из файла с историческими данными
            hist_df_curr_year_new = pd.read_excel(historical_data_file_current_year, sheet_name = vk)
            hist_df_full_hist_new = pd.read_excel(historical_data_file_full, sheet_name = vk)

            # Находим каналы, которые есть в hist_df, но отсутствуют в table
            missing_channels = set(hist_df_curr_year_new['Канал']) - set(table['Канал'])
            
            if missing_channels:
                raise ValueError(Color.RED + f"Для {vk} отсутствуют каналы в выгрузке: {missing_channels}" + Color.END)
                
            hist_df_updated = hist_df_curr_year_new.merge(table, on = 'Канал', how = 'inner')
            hist_df_updated_sorted = hist_df_updated.sort_values(by = 'Канал').reset_index(drop = True)
            hist_updated[vk] = hist_df_updated_sorted

            hist_df_full_updated = hist_df_full_hist_new.merge(table, on = 'Канал', how = 'inner')
            hist_df_full_updated_sorted = hist_df_full_updated.sort_values(by = 'Канал').reset_index(drop = True)
            hist_updated_full[vk] = hist_df_full_updated_sorted
        
        return hist_updated, hist_updated_full
    

    @staticmethod
    def kill_excel_processes():
        """Закрывает все процессы Excel"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.call(['taskkill', '/F', '/IM', 'EXCEL.EXE'], 
                            stdout = subprocess.DEVNULL, 
                            stderr = subprocess.DEVNULL)
                print("🔫 Процессы Excel завершены")
                time.sleep(1)  # Даем время на закрытие
        except:
            pass
    

    @staticmethod
    def save_with_xlsxwriter(dfs_dict: dict, filename = 'output.xlsx'):
        # 1. СОЗДАЕМ ФАЙЛ С УНИКАЛЬНЫМ ИМЕНЕМ (никогда не будет конфликтов)
        base, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{base}_temp_{timestamp}{ext}"
        
        #print(f"📁 Создаю временный файл: {temp_filename}")
        
        try:
            workbook = xlsxwriter.Workbook(temp_filename, {
            'nan_inf_to_errors': True,
            'constant_memory': True
            })
            
            for sheet_name, df in dfs_dict.items():
                # Очищаем данные
                df_clean = df.copy()
                df_clean = df_clean.where(pd.notnull(df_clean), None)
                df_clean = df_clean.replace([np.inf, -np.inf], None)
                
                # Создаем лист
                worksheet = workbook.add_worksheet(sheet_name[:31])
                
                # ============ ФОРМАТЫ ============
                # Формат для заголовков
                header_fmt = workbook.add_format({
                    'bold': True,
                    'align': 'center',
                    'valign': 'vcenter',
                    'bg_color': '#D9E1F2',
                    'border': 1,
                    'border_color': '#4472C4',
                    'font_size': 11,
                    'font_name': 'Arial'
                })
                
                # Формат для первого столбца (каналы)
                left_fmt = workbook.add_format({
                    'align': 'left',
                    'valign': 'vcenter',
                    'font_size': 10,
                    'font_name': 'Arial'
                })
                
                # Формат для чисел
                center_fmt = workbook.add_format({
                    'align': 'center',
                    'valign': 'vcenter',
                    'num_format': '0.000000',
                    'font_size': 10,
                    'font_name': 'Arial'
                })
                
                # Формат для нулевых значений
                zero_fmt = workbook.add_format({
                    'bg_color': '#FFBDBD',
                    'font_color': '#8B0000',
                    'bold': False,
                    'align': 'center',
                    'valign': 'vcenter',
                    'num_format': '0.000000',
                    'font_size': 10,
                    'font_name': 'Arial'
                })
                
                # ============ ЗАПИСЬ ДАННЫХ ============
                
                # 1. Пишем заголовки (строка 0)
                for col_idx, col_name in enumerate(df_clean.columns):
                    worksheet.write(0, col_idx, str(col_name), header_fmt)
                
                # 2. Пишем данные (начиная со строки 1)
                for row_idx, row in enumerate(df_clean.values, start=1):
                    for col_idx, value in enumerate(row):
                        # Пропускаем None
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            continue
                        
                        # Выбираем формат
                        if col_idx == 0:
                            fmt = left_fmt
                        else:
                            fmt = center_fmt
                        
                        # Записываем
                        try:
                            if col_idx == 0:
                                worksheet.write(row_idx, col_idx, str(value), fmt)
                            else:
                                worksheet.write(row_idx, col_idx, float(value), fmt)
                        except:
                            worksheet.write(row_idx, col_idx, str(value), fmt)
                
                # ============ УСЛОВНОЕ ФОРМАТИРОВАНИЕ ============
                n_rows = len(df_clean)
                n_cols = len(df_clean.columns)
                
                if n_rows > 0 and n_cols > 1:
                    worksheet.conditional_format(
                        1, 1,          # start_row, start_col (с первой строки данных, со второго столбца)
                        n_rows, n_cols - 1,  # end_row, end_col
                        {
                            'type': 'cell',
                            'criteria': '==',
                            'value': 0,
                            'format': zero_fmt
                        }
                    )
                
                # ============ ЗАКРЕПЛЕНИЕ ============
                worksheet.freeze_panes(1, 1)  # Закрепляем первую строку и первый столбец
                
                # ============ ШИРИНА КОЛОНОК ============
                for col_idx, col_name in enumerate(df_clean.columns):
                    # Собираем все значения для определения максимальной длины
                    values = [str(col_name)]
                    for row in df_clean.values:
                        val = row[col_idx]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            values.append(str(val))
                    
                    max_len = max(len(v) for v in values) if values else 10
                    worksheet.set_column(col_idx, col_idx, min(max_len + 2, 50))
            
            # ============ ЗАКРЫВАЕМ WORKBOOK ============
            workbook.close()
            workbook = None
            
            # Освобождаем память
            gc.collect()
            time.sleep(0.5)
            
            #print(f"✅ Временный файл создан: {temp_filename}")
            
            # ============ ПЫТАЕМСЯ ПЕРЕИМЕНОВАТЬ ============
            try:
                # Если оригинальный файл существует, пробуем удалить
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                        #print(f"🗑️ Старый файл удален")
                    except PermissionError:
                        # Не можем удалить - сохраняем с новым именем
                        new_filename = f"{base}_{timestamp}{ext}"
                        os.rename(temp_filename, new_filename)
                        print(f"✅ Файл сохранен как: {new_filename}")
                        print(f"ℹ️  Старый файл {filename} был занят")
                        return new_filename
                
                # Переименовываем временный файл
                os.rename(temp_filename, filename)
                print(f"✅ Файл сохранен как: {filename}")
                return filename
                
            except Exception as e:
                # Если не можем переименовать - оставляем временный файл
                print(f"⚠️ Не удалось переименовать: {e}")
                print(f"✅ Файл доступен как: {temp_filename}")
                return temp_filename
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            # Чистим временный файл
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass
            raise
    

    def thematic_share_pipeline(self, historical_data_full_hist: str, historical_data_curr_year: str):
        """
            Пайплайн для выгрузки данных
        """
        # ШАГ 1. Выгрузка данных из БД
        output_dict = self.make_api_calculation()

        # ШАГ 2. Обновление таблицы с историческими данными
        hist_updated, hist_updated_full = self.update_historical_data(output_dict, historical_data_full_hist, historical_data_curr_year)

        # ШАГ 3. Сохранение свежих данных в файл
        ThematicShare.save_with_xlsxwriter(hist_updated, historical_data_curr_year)

        # ШАГ 4. Сохранение свежих исторических данных в файл с форматироварием
        #data_dict = File(historical_data_full_hist).from_file(0)
        #hist_full_dict = Dict_Operations(data_dict).replace_keys_in_dict(['ЕРК', 'ЖРК', 'МРК', 'ДРК'])
        ThematicShare.save_with_xlsxwriter(hist_updated_full, historical_data_full_hist)

        return output_dict, hist_updated




class Share_Thematic:
    def __init__(self, dataframe_erk_drk, dataframe_mrk, dataframe_grk):
        self.dataframe_erk_drk = dataframe_erk_drk
        self.dataframe_mrk = dataframe_mrk
        self.dataframe_grk = dataframe_grk
        self.channels = {
            'ERK_DRK': 'ЕРК_ДРК',
            'MRK': 'МРК',
            'GRK': 'ЖРК'
        }
        
    @staticmethod
    def __get_output(dataframe):
        '''
        Static Private Method to get output DataFrame in new view from API
        '''
        dataframe['tvCompanyName'] = dataframe['tvCompanyName'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))
        data_output = dataframe.rename(columns = {'prj_name': 'ЦА', 'tvCompanyName': 'Канал'})
        #tmp = pd.DataFrame.copy(data_output[data_output['Канал'] == 'BRIDGE CLASSIC'])
        data_output.replace({'VIJU HISTORY': 'VIASAT HISTORY', 
                             'VIJU NATURE': 'VIASAT NATURE', 
                             'VIJU TV1000 НОВЕЛЛА': 'TV 1000 НОВЕЛЛА',
                             'VIJU EXPLORE': 'VIASAT EXPLORE', 
                             'ТВ-21М': 'ТВ21', 
                             'АВТО ПЛЮС ТВ': 'АВТОПЛЮС',
                             'БОБЁР': 'БОБЕР', 
                             'VIJU TV1000': 'TV 1000',
                             'VIJU TV1000 ACTION': 'TV 1000 ACTION', 
                             'VIJU TV1000 РУССКОЕ': 'TV 1000 РУССКОЕ КИНО', 
                             'ЛЯ МИНОР. МОЙ МУЗЫКАЛЬНЫЙ': 'ЛЯ-МИНОР ТВ',
                             'BRIDGE CLASSIC': 'БРИДЖ ТВ CLASSIC', 
                             'BRIDGE HITS': 'БРИДЖ ТВ ХИТ',
                             'BRIDGE РУССКИЙ ХИТ': 'БРИДЖ ТВ РУССКИЙ ХИТ', 
                             'О!': 'О', 
                             'ПОЕХАЛИ!': 'ПОЕХАЛИ',
                             'ПОБЕДА': 'ПОБЕДА', 
                             'BRIDGE': 'БРИДЖ ТВ', 
                             'RU.TV': 'РУ ТВ'},
                             inplace = True)
        #data_output = pd.concat([tmp, data_output], ignore_index=True)
        #data_output.replace('BRIDGE', 'БРИДЖ ТВ', inplace = True)
        data_output.insert(loc = 2, column = 'Channel', value = data_output['Канал'] + ' ' + data_output['ЦА'])
        return data_output

    def __get_share(self, dataframe, channel_name: str, is_dropna: bool = True):
        '''
        Private Method to get Share statistic by using VLOOKUP and merging DataFrame API with each Thematic Channel
        dataframe - DataFRame in new view from API, which you received in previous function
        '''
        if channel_name == self.channels['ERK_DRK']:
            data = pd.merge(self.dataframe_erk_drk, dataframe, on = 'Channel', how = 'left')
            data = data.reindex(self.dataframe_erk_drk.index)
        elif channel_name == self.channels['MRK']:
            data = pd.merge(self.dataframe_mrk ,dataframe, on = 'Channel', how = 'left')
            data = data.reindex(self.dataframe_mrk.index)
        elif channel_name == self.channels['GRK']:
            data = pd.merge(self.dataframe_grk, dataframe, on = 'Channel', how = 'left')
            data = data.reindex(self.dataframe_grk.index)
        else:  
            print('This channel name does not exist' + channel_name)
        data = data.drop(['ЦА', 'Канал'], axis = 1)
        
        if is_dropna:
            data = data.dropna()
        return data

    def __to_file(self, filepath, channel_data: str, sheet_name: str):
        '''
        Private Method which helps you to write your received data to Excel file 
        '''
        with pd.ExcelWriter(filepath, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
            channel_data.to_excel(writer, sheet_name = sheet_name)
        #writer.save()

    def get_data(self, dataframe, filepath, is_dropna = True):
        '''
        This function helps you to get final result for each thematic channel
        dataframe - is the output DataFrame in the new view from API
        '''
        data_output = Share_Thematic.__get_output(dataframe)
        #data_output.to_excel('data_output.xlsx')        
        
        res = {}
        for key, value in self.channels.items():
            res[key] = self.__get_share(data_output, value, is_dropna)
            self.__to_file(filepath, channel_data = res[key], sheet_name = value)
        return data_output, res