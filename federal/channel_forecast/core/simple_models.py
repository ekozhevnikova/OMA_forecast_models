import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from dateutil.relativedelta import relativedelta
from OMA_tools.io_data.operations import Dates_Operations
from OMA_tools.io_data.time_series import TimeSeriesTransformer
from sklearn.preprocessing import LabelEncoder
import json


class PrimitiveModel:
    """
        Класс для прогнозирования долей телепрограмм примитивным методом.
    """
    def __init__(self, start_date: str, start_of_current_year: str):
        """
            Атрибуты класса
                start_date: str: Дата, начиная с которой начинаем построение прогноза на будущие периоды
                start_of_current_year: str: Дата начала года, начиная с которого строим исторический DataFrame
        """
        self.start_date = start_date
        self.start_of_current_year = start_of_current_year


    @staticmethod
    def build_russian_holidays(holidays_path: str):
        """
            Генератор праздников РФ
            Args:
                holidays_path: путь к файлу json, в котором перечислены все празднику согласно производственному календарю
            Returns: 
                holidays['working_saturdays']: список из рабочих суббот
                all_holidays: список праздников за исключением рабочих суббот
        """
        with open(holidays_path, 'r', encoding = 'utf-8') as file:
            holidays = json.load(file)

        all_holidays = []
        for year in list(holidays.keys())[1:]:
            all_holidays.extend(holidays[year])
        return holidays['working_saturdays'], all_holidays


    @staticmethod
    def round_time_to_str_advanced(time_obj: str):
        """
            Округляет время до ближайших 10 минут.
            Примеры:
                14:15 -> 14:20
                08:44 -> 08:40
                15:32 -> 15:30
                22:55 -> 23:00
            Args:
                time_obj: str: слот в формате данных строка.
        """
        # Преобразуем строку в объект времени
        if isinstance(time_obj, str):
            try:
                # Для формата 'HH:MM:SS'
                time_obj = datetime.strptime(time_obj, '%H:%M:%S').time()
            except ValueError:
                try:
                    # Для формата 'HH:MM'
                    time_obj = datetime.strptime(time_obj, '%H:%M').time()
                except ValueError:
                    raise ValueError(f"Неверный формат времени: {time_obj}")
        
        total_minutes = time_obj.hour * 60 + time_obj.minute
        remainder = total_minutes % 10
        
        if remainder < 5:
            # Округляем вниз
            rounded_minutes = total_minutes - remainder
        else:
            # Округляем вверх
            rounded_minutes = total_minutes + (10 - remainder)
        
        # Обработка перехода через полночь
        hours = (rounded_minutes // 60) % 24
        minutes = rounded_minutes % 60
        
        return f'{hours:02d}:{minutes:02d}:00'
    

    @staticmethod
    def get_day_type(date, holidays: list, worling_saturdays: list):
        date_str = datetime.strftime(date, '%Y-%m-%d')
        
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        weekday = date_obj.weekday()
        
        if date_str in worling_saturdays:
            return 'Будни'
            
        elif date_str in holidays:
            return 'Выходной'
            
        elif weekday < 5:
            return 'Будни'
            
        else:
            return 'Выходной'
        

    @staticmethod
    def create_mask(data_source, params, slot, day, day_type):
        """
            Создать маску для фильтрации данных
        """
        mask = data_source['Share'] != ''
        
        param_mapping = {
            'slot': ('Время выхода', slot),
            'day': ('День недели', day), 
            'day_type': ('Тип дня', day_type)
        }
        
        for param in params:
            if param not in param_mapping:
                continue
                
            col_name, value = param_mapping[param]
            if value is None:
                continue
                
            mask &= (data_source[col_name] == value)
        
        return mask


    @staticmethod
    def find_shares_in_hierarchy(
                        slot, 
                        day, 
                        day_type, 
                        search_hierarchy: dict, 
                        data_sources: dict, 
                        source_priority = ['last_4_weeks', 'current_year', 'history']
                        ):
        """
            Поиск данных по иерархии с возможностью настройки порядка источников
        """
        for level_name in search_hierarchy.keys():
            level_config = search_hierarchy[level_name]
            
            for source_name in source_priority:
                if source_name in level_config:
                    params = level_config[source_name]
                    data_source = data_sources[source_name]
                    
                    mask = PrimitiveModel.create_mask(data_source, params, slot, day, day_type)
                    shares_data = data_source[mask]['Share']
                    
                    if len(shares_data) > 0:
                        return shares_data
        
        return pd.Series(dtype = float)


    @staticmethod
    def get_clean_mean(series):
        """
            Обработать выбросы и вернуть среднее значение
        """
        if len(series) == 0:
            return 0.0
        clean_shares = TimeSeriesTransformer(series).replace_outliers_with_median()
        return np.mean(clean_shares)


    @staticmethod
    def calculate_share(df, day, day_type, slot):
        """
            Расчет доли
        """
        data = df.copy()
        
        mask = (data['День недели'] == day) & \
            (data['Тип дня'] == day_type) & \
            (data['Время выхода'] == slot)
        
        shares_data = data[mask]['Share']

        # 1. По дню недели и типу дня
        if len(shares_data) == 0:
            mask = (data['День недели'] == day) & (data['Тип дня'] == day_type)
            shares_data = data[mask]['Share']

        # 2. По дню недели и времени выхода
        if len(shares_data) == 0:
            mask = (data['День недели'] == day) & (data['Время выхода'] == slot)
            shares_data = data[mask]['Share']


        # 3. По типу дня и времени выхода
        if len(shares_data) == 0:
            mask = (data['Тип дня'] == day_type) & (data['Время выхода'] == slot)
            shares_data = data[mask]['Share']

        # 4. По дню недели
        if len(shares_data) == 0:
            mask = (data['День недели'] == day)
            shares_data = data[mask]['Share']
            
        # 5. По типу дня
        if len(shares_data) == 0:
            mask = (data['Тип дня'] == day_type)
            shares_data = data[mask]['Share']

        # 5. По времени выхода
        if len(shares_data) == 0:
            mask = (data['Время выхода'] == slot)
            shares_data = data[mask]['Share']

        # 6. Весь датасет
        if len(shares_data) == 0:
            shares_data = data['Share']


        if len(shares_data) > 5:
            clean_shares = TimeSeriesTransformer(shares_data).replace_outliers_with_median()
            share_mean = np.mean(clean_shares)
        else:
            share_mean = np.mean(shares_data)
            
        return share_mean
    
    # Округление слотов
    #for i in range(len(time_slot_columns)):
    #    df[time_slot_columns[i]] = df[time_slot_columns[i]].apply(PrimitiveModel.round_time_to_str_advanced)

    def forecast_big_programs(self, big_programs_dict: dict, russian_holidays: str, search_hierarchy) -> pd.DataFrame:
        """
            Метод для прогнозирования долей на крупных программах с богатой историей.

            Args:
                big_programs: dict: словарь из крупных программ, в котором ключ: название программы, значение: DataFrame с историей.
                russian_holidays: str: Путь к файлу .json, в котором хранится информации о праздниках РФ, начиная с 2021 г.

            Returns:
                pd.DataFrame: DataFrame с прогнозом для всех крупных программ.
        """
        # Генерация праздников на основе json файла
        work_saturdays, all_holidays = PrimitiveModel.build_russian_holidays(russian_holidays)

        # Список для хранения прогнозов
        forecast_big = []

        for big_program in list(big_programs_dict.keys()):
            df = big_programs_dict[big_program]

            # 1. Определение типа дня
            df['Тип дня'] = df['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))
            # 0 - Будни, 1 - Выходные
            df['Тип дня'] = LabelEncoder().fit_transform(df['Тип дня'])
            tmp_df = df[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'День недели', 'Тип дня', 'Share']]
            
            # 2. Отбор исторических данных, начиная с 2021 г
            historical_data = tmp_df[tmp_df['Дата'] < self.start_date].reset_index(drop = True).copy()
            
            # 3. Отбор только текущего года
            current_year = tmp_df[tmp_df['Дата'] < self.start_of_current_year].reset_index(drop = True).copy()
            
            # 4. Отбор последних 4х недель, исходя из максимальной даты в фактических данных
            #start_date_ = pd.to_datetime(start_date)
            four_weeks_ago = historical_data['Дата'].max() - pd.Timedelta(weeks = 4)
            last_4_weeks = historical_data[historical_data['Дата'] > four_weeks_ago].reset_index(drop = True).copy()
            
            # 4. Данные для прогнозирования (начиная с даты начала прогноза)
            future_data = tmp_df[tmp_df['Дата'] >= self.start_date].reset_index(drop = True).copy()
            
            data_sources = {
                        'last_4_weeks': last_4_weeks,
                        'current_year': current_year,
                        'history': historical_data
                    }
            # Основной цикл для Прогнозирования Доли
            slots_unique = future_data['Время выхода'].unique()
            days_of_week_unique = future_data['День недели'].unique()
            day_types_unique = future_data['Тип дня'].unique()
            
            for slot in slots_unique:
                for day in days_of_week_unique:
                    for day_type in day_types_unique:
                        # Проверяем, существует ли такая комбинация в future_data
                        mask = (future_data['Время выхода'] == slot) & \
                            (future_data['День недели'] == day) & \
                            (future_data['Тип дня'] == day_type)
                        
                        if not mask.any():
                            continue  # пропускаем несуществующие комбинации
                        
                        # Ищем данные по иерархии
                        shares_data = PrimitiveModel.find_shares_in_hierarchy(slot, day, day_type, search_hierarchy, data_sources)
                        share_mean = PrimitiveModel.get_clean_mean(shares_data)
                        
                        # Заполняем Share для всех строк с этой комбинацией
                        future_data.loc[mask, 'Share'] = share_mean
            forecast_big.append(future_data)
            
        return pd.concat(forecast_big).reset_index(drop = True)
    

    def forecast_small_programs(self, small_programs_dict: dict, russian_holidays: str) -> pd.DataFrame:
        """
            Метод для прогнозирования долей на мелких программ.

            Args:
                small_programs_dict: dict: словарь из мелких программ, в котором ключ: название программы, значение: DataFrame с историей.
                russian_holidays: str: Путь к файлу .json, в котором хранится информации о праздниках РФ, начиная с 2021 г.

            Returns:
                pd.DataFrame: DataFrame с прогнозом для всех мелких программ.
        """
        # Генерация праздников на основе json файла
        work_saturdays, all_holidays = PrimitiveModel.build_russian_holidays(russian_holidays)

        # Словарь для хранения результатов
        res = {}

        for small_program in list(small_programs_dict.keys()):
            df = small_programs_dict[small_program]

            # 1. Определение типа дня
            df['Тип дня'] = df['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))
            # 0 - Будни, 1 - Выходные
            df['Тип дня'] = LabelEncoder().fit_transform(df['Тип дня'])
            tmp_df = df[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'День недели', 'Тип дня', 'Share']]
            
            share_mean = None
            
            # 1. Отбор исторических данных, начиная с 2021 г
            historical_data = tmp_df[tmp_df['Дата'] < self.start_date].reset_index(drop = True).copy()
            
            # 2. Отбор текущего года
            current_year = df[(df['Дата'] >= self.start_of_current_year) & (df['Дата'] < self.start_date)].reset_index(drop = True).copy()
            
            # 3. Отбираем данные, начиная с предыдущего года
            start_of_previous_year = (datetime.strptime(self.start_of_current_year, '%Y-%m-%d') 
                                - relativedelta(years = 1)).strftime('%Y-%m-%d')
            prev_year = df[(df['Дата'] >= start_of_previous_year) & (df['Дата'] < self.start_of_current_year)].reset_index(drop = True).copy()
            
            # Если есть данные по прошлому году
            if len(prev_year) != 0:
                prev_to_current_year = pd.concat([prev_year, current_year]).reset_index(drop = True)
            # Если не нашлось данных по прошлому году
            else:
                prev_to_current_year = pd.DataFrame()
            
            # 4. Данные для прогнозирования (начиная с даты начала прогноза)
            future_data = tmp_df[tmp_df['Дата'] >= self.start_date].reset_index(drop = True).copy()
            
            # Группируем future_data по уникальным комбинациям
            combinations = future_data[['Время выхода', 'День недели', 'Тип дня']].drop_duplicates()
            
            # Если история очень короткая
            if len(historical_data) < 5:
                # Считаем среднее между последними двумя значениями
                if len(current_year) != 0:
                    if len(current_year) > 2:
                        last_two_values = historical_data['Share'].iloc[-2:]
                        average_last_two = last_two_values.mean()
                        future_data['Share'] = future_data['Share'].replace('', average_last_two)
                    else:
                        share_mean = np.mean(current_year['Share'])
                        # Заполняем Share для всех строк с этой комбинацией
                        future_data['Share'] = future_data['Share'].replace('', share_mean)
                else:
                    share_mean = np.mean(historical_data['Share'])
                    # Заполняем Share для всех строк с этой комбинацией
                    future_data['Share'] = future_data['Share'].replace('', share_mean)
                    
            else:
                # Считаем по предыдущему году
                if len(current_year) != 0:
                    
                    data = current_year[current_year['Share'] != '']
                    
                    for idx, row in combinations.iterrows():
                        slot = row['Время выхода']
                        day = row['День недели']
                        day_type = row['Тип дня']
                        
                        share_mean = PrimitiveModel.calculate_share(data, day, day_type, slot)
                        
                        # Создаем маску для конкретной комбинации
                        mask = (future_data['Время выхода'] == slot) & \
                            (future_data['День недели'] == day) & \
                            (future_data['Тип дня'] == day_type)
                        
                        future_data.loc[mask, 'Share'] = share_mean
                                
                elif len(prev_to_current_year) != 0:
                    data = prev_to_current_year[prev_to_current_year['Share'] != '']
                    
                    for idx, row in combinations.iterrows():
                        slot = row['Время выхода']
                        day = row['День недели']
                        day_type = row['Тип дня']
                        
                        share_mean = PrimitiveModel.calculate_share(data, day, day_type, slot)
                        
                        # Создаем маску для конкретной комбинации
                        mask = (future_data['Время выхода'] == slot) & \
                            (future_data['День недели'] == day) & \
                            (future_data['Тип дня'] == day_type)
                        
                        future_data.loc[mask, 'Share'] = share_mean
            res[small_program] = future_data

        return pd.concat(res.values(), ignore_index = True)
    

    def forecast_new_programs(self, new_programs: pd.DataFrame, vimb: pd.DataFrame, plmrs: pd.DataFrame, russian_holidays: str):
        """
            Метод для прогнозирования долей на новых программ.

            Args:
                small_programs_dict: dict: словарь из мелких программ, в котором ключ: название программы, значение: DataFrame с историей.
                russian_holidays: str: Путь к файлу .json, в котором хранится информации о праздниках РФ, начиная с 2021 г.

            Returns:
                pd.DataFrame: DataFrame с прогнозом для всех мелких программ.
        """
        day_type_mapping = {'Будни': 0, 'Выходной': 1}

        # Генерация праздников на основе json файла
        work_saturdays, all_holidays = PrimitiveModel.build_russian_holidays(russian_holidays)

        plmrs['Дата'] = pd.to_datetime(plmrs['Дата'])
        #not_found = result[result['Программа Palomars'] == 0].reset_index()

        not_found_programs = list(new_programs['Программа VIMB'])

        res_not_found = {}
        for new_pr in not_found_programs:
            # 1. Отбор по программе
            df = vimb.copy()
            df['Flag'] = vimb['program_name'].str.contains(new_pr)  
            analysis = df.loc[(df['Flag'] == True)]
            
            tmp_df = analysis[['Дата', 'program_name', 'Время выхода', 'Время окончания', 'День недели']].reset_index(drop = True)
            tmp_df.rename(columns = {'program_name': 'Название программы'}, inplace = True)
            
            tmp_df['Дата'] = pd.to_datetime(tmp_df['Дата'])
            
            # 2. Определение типа дня
            tmp_df['Тип дня'] = tmp_df['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))
            # 0 - Будни, 1 - Выходные
            tmp_df['Тип дня'] = tmp_df['Тип дня'].map(day_type_mapping)
            
            # 3. Создание столбца с Share
            tmp_df['Share'] = ''
            
            # 4. Отбор исторических даных, начиная с 2021
            historical_data = plmrs[plmrs['Дата'] < self.start_date].reset_index(drop = True).copy()
            
            # Определение типа дня
            historical_data['Тип дня'] = historical_data['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))
            # 0 - Будни, 1 - Выходные
            historical_data['Тип дня'] = historical_data['Тип дня'].map(day_type_mapping)
            
            # 5. Отбор последних 4х недель, исходя из максимальной даты в фактических данных
            four_weeks_ago = historical_data['Дата'].max() - pd.Timedelta(weeks = 4)
            last_4_weeks = historical_data[historical_data['Дата'] > four_weeks_ago].reset_index(drop = True).copy()
            
            # Определение типа дня
            last_4_weeks['Тип дня'] = last_4_weeks['Дата'].apply(lambda x: PrimitiveModel.get_day_type(x, all_holidays, work_saturdays))
            # 0 - Будни, 1 - Выходные
            last_4_weeks['Тип дня'] = last_4_weeks['Тип дня'].map(day_type_mapping)
            # Конвертация столбца Время выхода в строку
            last_4_weeks['Время выхода'] = last_4_weeks['Время выхода'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notnull(x) else x)
            
            # Конвертация столбца Время выхода в строку
            historical_data['Время выхода'] = historical_data['Время выхода'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notnull(x) else x)

            future_data = tmp_df.copy()

            # Группируем future_data по уникальным комбинациям
            combinations = future_data[['Время выхода', 'День недели', 'Тип дня']].drop_duplicates()
            for idx, row in combinations.iterrows():
                slot = row['Время выхода']
                day = row['День недели']
                day_type = row['Тип дня']
                
                share_mean = PrimitiveModel.calculate_share(last_4_weeks, day, day_type, slot)
                
                # Создаем маску для конкретной комбинации
                mask = (future_data['Время выхода'] == slot) & \
                    (future_data['День недели'] == day) & \
                    (future_data['Тип дня'] == day_type)
                
                future_data.loc[mask, 'Share'] = share_mean

            res_not_found[new_pr] = future_data

        return pd.concat(res_not_found.values(), ignore_index = True)
    


class TVShareCalculator:
    """
        Класс для расчета долей в конкретных слотах через вес слота и процент длительности программы в часе
    """
    def __init__(self, start_date: pd.Timestamp, total_tv_auedience: pd.DataFrame, n: int = 4):
        """
            Атрибуты класса
            Args:
                start_date: pd.Timestamp: дата, начиная с которой будем строить прогноз
                df: pd.DataFrame: таблица с посчитанной прогнозной долей, которая нуждается в дальнейшем пересчёте
                auedience: pd.DataFrame: таблица с прогнозными значениями Total TV Auedience по слотам за последние n недель.
                n: количество недель, которое берется для анализа. По умолчанию 4
        """
        self.start_date = start_date
        self.total_tv_auedience = total_tv_auedience
        self.n = n
        self.auedience_forecast = None
        self.df = None
    

    @staticmethod
    def get_hour(dt, param: str):
        """
            Метод для генерации часа. 
            Возможные опции: начало текущего часа, конец текущего часа, начало следующего часа, начало предыдущего часа
        """
        # Вариант 1: Начало текущего часа
        if param == 'start_of_current_hour':
            return dt.replace(minute = 0, second = 0, microsecond = 0)

        # Вариант 2: Конец текущего часа
        elif param == 'end_of_current_hour':
            return dt.replace(minute = 59, second = 59, microsecond = 0)
        
        # Вариант 3: Старт следующего часа
        elif param == 'start_next_hour':
            return dt.replace(minute = 0, second = 0, microsecond = 0) + timedelta(hours = 1)
        
        # Вариант 4: Старт прошлого часа
        elif param == 'start_previous_hour':
            return dt.replace(minute = 0, second = 0, microsecond = 0) + timedelta(hours = -1)
    
        
    @staticmethod
    def get_previous_hour(dt):
        """
            Вспомогательная функция для вычисления ближайшего предыдущего часа
        """
        prev_hour = dt.replace(minute = 0, second = 0, microsecond = 0)
        return prev_hour


    def _total_tv_auedience_predict(self) -> pd.DataFrame:
        """
            Метод для прогнозирования Total TV Auedience по слотам на будущие периоды.
            В качестве прогноза берется среднее за последние n недель для каждого слота.
            Args:
                start_date: дата, начиная с которой мы начинаем строить прогноз.
                total_tv_auedience: pd.DataFrame с историческими данными по Auedience.
                n: количество последних недель, которые мы берем с расчет. По умолчанию n = 4.
            Returns:
                forecast: DataFrame с прогнозными значениями Total TV Auedience
        """
        dates = Dates_Operations.get_last_4_weeks(self.start_date, self.n)
        # Выделение последних n недель
        last_weeks_auedience = self.total_tv_auedience[(self.total_tv_auedience['Date'].astype(str).isin(dates))].reset_index(drop = True)
        # Вычисляем средние значения по слотам за последние 4 недели
        self.auedience_forecast = last_weeks_auedience.groupby('TimeSlot')['TTVRtg000'].mean().reset_index()
        return self.auedience_forecast
    

    def calculate_slot_weight(self) -> pd.DataFrame:
        """
            Метод для расчета веса слотов для конкретного дня.
            Args:
                df: входной датафрейм с Total TV Auedience
            Returns:
                pd.DataFrame: DataFrame с весами слотов
        """
        self.auedience_forecast = self._total_tv_auedience_predict()
        summ_audience = np.sum(list(self.auedience_forecast['TTVRtg000']))
        self.auedience_forecast['Slot_weight'] = self.auedience_forecast['TTVRtg000'] / summ_audience
        
        self.auedience_forecast['TimeSlot_dt'] = pd.to_datetime(self.auedience_forecast['TimeSlot'])
        self.auedience_forecast['hour_start'] = self.auedience_forecast['TimeSlot_dt'].dt.hour
        return self.auedience_forecast[['TimeSlot', 'Slot_weight', 'hour_start']].reset_index(drop = True)


    def calculate_weighted_share(self, data, auedience) -> pd.DataFrame:
        """
            Функция для расчета взвешенной доли. 
            Args:
                df: pd.DataFrame: Датафрейм, в котором есть столбцы Долей (Share), Время выхода, Время окончания, Название программы для какого одного дня.
                auedience: pd.DataFrame: ДатаФрейм с весами слотов, посчитанными через TotalTVAuedience для конкретного дня.
            Returns:
                data: pd.DataFrame: Датафрейм с новой рассчитанной долей
        """
        self.df = data
        # Если время уже в формате времени, преобразуем в строку и затем в datetime
        self.df['Время выхода_dt'] = pd.to_datetime(self.df['Время выхода'].astype(str))
        self.df['Время окончания_dt'] = pd.to_datetime(self.df['Время окончания'].astype(str))
        self.df['hour_start'] = self.df['Время выхода_dt'].dt.hour
        self.df['hour_end'] = self.df['Время окончания_dt'].dt.hour

        # Создаем новые столбцы для результатов
        self.df['Длительность 1'] = pd.Timedelta(0)   # если нет скачка через час
        self.df['Длительность 2'] = pd.Timedelta(0)   # если есть скачок через час
        self.df['Разделение часа'] = False

        # Проходим по всем строкам
        for idx, row in self.df.iterrows():
            start_time = row['Время выхода_dt']
            end_time = row['Время окончания_dt']
            
            # Проверяем, совпадают ли часы
            if start_time.hour == end_time.hour:
                # Если часы совпадают - простая разность
                self.df.at[idx, 'Длительность 1'] = end_time - start_time
                self.df.at[idx, 'Разделение часа'] = False
                
            else:
                # Если часы разные - разделяем на две длительности
                end_hour = TVShareCalculator.get_hour(start_time, 'end_of_current_hour')
                start_of_curr_hour = TVShareCalculator.get_hour(end_time, 'start_of_current_hour')
                
                # Длительность 1: от начала до следующего часа
                duration1 = end_hour - start_time
                
                # Длительность 2: от предыдущего часа до окончания
                duration2 = end_time - start_of_curr_hour
                
                self.df.at[idx, 'Длительность 1'] = duration1
                self.df.at[idx, 'Длительность 2'] = duration2
                self.df.at[idx, 'Разделение часа'] = True

        # Конвертируем в минуты для удобства
        self.df['Длительность 1, мин'] = self.df['Длительность 1'].dt.total_seconds() / 60.0
        self.df['Длительность 2, мин'] = self.df['Длительность 2'].dt.total_seconds() / 60.0

        # Считаем % длительности программы в часе
        self.df['% duration 1'] = self.df['Длительность 1, мин'] / 60.0
        self.df['% duration 2'] = self.df['Длительность 2, мин'] / 60.0

        res = self.df[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 
                'hour_start', 'hour_end', 'Share', 'Длительность 1, мин',
                'Длительность 2, мин', '% duration 1', '% duration 2', 'Разделение часа']]

        # Создаём столбец с новой долей
        res['Share_NEW'] = 0.0


        ########################## Расчет новой доли ####################
        for i in range(len(res)):
            slot_weight_2 = 0.0
            
            start_hour = res.iloc[i]['hour_start']
            end_hour = res.iloc[i]['hour_end']
            share = res.iloc[i]['Share']
            duration_1 = res.iloc[i]['% duration 1']
            
            # Если есть скачок через час
            duration_2 = res.iloc[i]['% duration 2']
            
            slot_weight_1 = auedience.loc[auedience['hour_start'] == start_hour, 'Slot_weight'].iloc[0]

            # Если скачка через час нет, считаем долю в слоте как Share * вес слота * % длительности программы в часе
            res.at[i, 'Share_NEW'] = share * duration_1 * slot_weight_1
            
            # Если есть скачок через час, считаем долю по-другому
            if duration_2 != 0.0:
                
                slot_weight_2 = auedience.loc[auedience['hour_start'] == end_hour, 'Slot_weight'].iloc[0]
                res.at[i, 'Share_NEW'] = share * (duration_1 * slot_weight_1 + duration_2 * slot_weight_2)
                
        data = res[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share_NEW']]
        data.rename(columns = {'Share_NEW': 'Share'}, inplace = True)
        # Расчёт суммарной доли по дню
        share_sum = np.sum(list(data['Share']))
        return data, share_sum
