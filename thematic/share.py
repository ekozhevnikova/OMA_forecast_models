import time
import os
import numpy as np
import xlsxwriter
import pandas as pd
import subprocess
from datetime import datetime
import gc

from OMA_tools.io_data.operations import File, Table, Dict_Operations
from OMA_tools.regions.data_extraction.task_builder import *


class ThematicShare:
    """
        Класс для выгрузки статистик Тематического ТВ
    """
    def __init__(self, date_filter: list, statistics: list, channels_id_file: str, channels_guide_file: str):
        self.date_filter = date_filter
        self.statistics = statistics
        self.channels_id_file = channels_id_file
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
            'ВСЕ 25-49':'age >= 25 AND age <= 49',
            'М 25-49':'age >= 25 AND age <= 49 AND sex = 1',
            'Ж 25-49':'age >= 25 AND age <= 49 AND sex = 2'
        }

        self.channels_guide = pd.read_excel(self.channels_guide_file)
        need_columns = ['Канал', 'Название канала в VIMB', 'ЕРК', 'ЖРК', 'МРК', 'ДРК']
        self.channels_guide = self.channels_guide[need_columns]


        data = pd.read_excel(self.channels_id_file)
        data_ = np.array(data['ID']).tolist()
        data_id = list(map(lambda x: str(x), data_))
        self.company_filter = f'tvCompanyId IN ({", ".join(data_id)})'
    

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
                #df = pd.read_excel(f'{driver_N}ИСТОРИЧЕСКИЕ ДАННЫЕ/Исторические доли.xlsx', sheet_name = vk)
                #months_order = list(df.columns[1:])

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
    

    def update_historical_data(self, output_dict, historical_data_file: str):
        """
            Метод для обновления файла с фактическими данными
        """
        # Проверяем, существует ли файл
        if not os.path.exists(historical_data_file):
            print(f"📁 Файл {historical_data_file} не найден. Будет создан новый файл с текущими данными")
            return output_dict
            
        
        data_dict = File(historical_data_file).from_file(0)
        hist_dict = Dict_Operations(data_dict).replace_keys_in_dict(['ЕРК', 'ЖРК', 'МРК', 'ДРК'])

        hist_updated = {}

        for vk, hist_df in hist_dict.items():
            table = output_dict[vk]
            
            # Находим каналы, которые есть в hist_df, но отсутствуют в table
            missing_channels = set(hist_df['Канал']) - set(table['Канал'])
            
            if missing_channels:
                print(f"Для {vk} отсутствуют каналы в table: {missing_channels}")
                
            hist_df_new = hist_df.merge(table, on = 'Канал', how = 'inner')

            hist_df_new_sorted = hist_df_new.sort_values(by = 'Канал').reset_index(drop = True)
            
            hist_updated[vk] = hist_df_new_sorted
        
        return hist_updated
    

    @staticmethod
    def kill_excel_processes():
        """Закрывает все процессы Excel"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.call(['taskkill', '/F', '/IM', 'EXCEL.EXE'], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
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
        
        print(f"📁 Создаю временный файл: {temp_filename}")
        
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
            
            print(f"✅ Временный файл создан: {temp_filename}")
            
            # ============ ПЫТАЕМСЯ ПЕРЕИМЕНОВАТЬ ============
            try:
                # Если оригинальный файл существует, пробуем удалить
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                        print(f"🗑️ Старый файл удален")
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
    

    def thematic_share_pipeline(self, historical_data_file: str):
        """
            Пайплайн для выгрузки данных
        """
        # ШАГ 1. Выгрузка данных из БД
        output_dict = self.make_api_calculation()

        # ШАГ 2. Обновление таблицы с историческими данными
        hist_updated = self.update_historical_data(output_dict, historical_data_file)

        # ШАГ 3. Сохранение свежих данных в файл
        ThematicShare.save_with_xlsxwriter(hist_updated, historical_data_file)

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