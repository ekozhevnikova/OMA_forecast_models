import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from OMA_tools.io_data.operations import Dates_Operations
from OMA_tools.io_data.time_series import TimeSeriesTransformer


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
    

    def forecast_big_programs(self, big_programs_dict: dict) -> pd.DataFrame:
        """
            Функция для прогнозирования долей на крупных программах с богатой историей.
            Args:
                big_programs: dict: словарь из крупных программ, в котором ключ: название программы, значение: DataFrame с историей.
                start_date: str: дата старта. Дата, начиная с которой будет строить прогноз на будущие периоды.
                start_of_current_year: str: Дата старта года, от которого будем смотреть историю. Если указали 2024-01-01, 
                                            то будем смотреть с начала 2024 г и далее.
            Returns:
                big_forecast: pd.DataFrame: DataFrame с прогнозом для всех крупных программ.
        """
        time_slot_columns = ['Время выхода', 'Время окончания']
        
        big_programs = list(big_programs_dict.keys())

        # Перевод даты в формат времени
        start_date = pd.to_datetime(self.start_date)

        forecast = {}

        for big_program in big_programs:
            
            # Отбор программы
            df = big_programs_dict[big_program]
            
            # Округление слотов
            for i in range(len(time_slot_columns)):
                df[time_slot_columns[i]] = df[time_slot_columns[i]].apply(PrimitiveModel.round_time_to_str_advanced)
                
            # Отбор текущего года
            df_current_year = df[df['Дата'].astype(str) >= self.start_of_current_year]
            
            # Создаем копию для работы
            data = df_current_year[df_current_year['Share'] == ''].copy().reset_index(drop = True)
            
            # Если нет новых программ для прогноза, переходим к следующей
            if len(data) == 0:
                print(f"Для программы {big_program} нет новых данных для прогноза")
                continue
            
            # Отбираем уникальные дни недели и слоты
            day_of_week = list(set(data['День недели']))
            slots = list(set(data['Время выхода']))
        
            # Моделируем будущие доли на будущие периоды
            for day in day_of_week:
                for slot in slots:
                    # Создаем маску для текущего дня и слота
                    mask = (data['День недели'] == day) & (data['Время выхода'] == slot)
                    rows_to_fill = data[mask]
                    
                    if len(rows_to_fill) == 0:
                        continue
                        
                    # Ищем исторические данные для ТОГО ЖЕ дня недели и слота
                    historical_mask = (df_current_year['День недели'] == day) & \
                                    (df_current_year['Время выхода'] == slot) & \
                                    (df_current_year['Share'] != '')
                    historical_data = df_current_year[historical_mask]
                    
                    # Если истории очень мало
                    if len(historical_data) < 1:
                        print(f"Мало исторических данных, используем последние 4 недели...")
                        
                        # Получаем даты за последние 4 недели
                        dates = Dates_Operations.get_last_4_weeks(start_date)
                        
                        # Ищем исторические данные за последние 4 недели для ЭТОГО ЖЕ дня недели
                        last_4_weeks_data = df_current_year[
                            (df_current_year['Дата'].astype(str).isin(dates)) &
                            (df_current_year['День недели'] == day) &
                            (df_current_year['Время выхода'] == slot) &
                            (df_current_year['Share'] != '')
                        ]
                        
                        print(f"Найдено исторических данных за 4 недели: {len(last_4_weeks_data)}")
                        
                        if len(last_4_weeks_data) > 0:
                            # Обрабатываем выбросы и считаем среднее
                            try:
                                drop_outlinear = TimeSeriesTransformer(last_4_weeks_data['Share']).replace_outliers_with_median()
                                share_mean = np.mean(drop_outlinear)
                                #print(f"Рассчитанное среднее: {share_mean}")
                            except Exception as e:
                                print(f"Ошибка при расчете: {e}")
                                share_mean = 0.0
                        else:
                            # Если вообще нет данных, ищем среднее по всем данным программы
                            print("Нет данных за 4 недели, используем общее среднее по программе")
                            all_shares = df_current_year[
                                (df_current_year['Share'] != '') & 
                                (df_current_year['Share'].notna())
                            ]['Share']
                            if len(all_shares) > 0:
                                share_mean = np.mean(all_shares)
                            else:
                                share_mean = 0.0
                            #print(f"Общее среднее: {share_mean}")
                        
                        # Заполняем данные
                        data.loc[mask, 'Share'] = share_mean
                        
                    else:
                        # Если история богатая
                        drop_outlinear = TimeSeriesTransformer(historical_data['Share']).replace_outliers_with_median()
                        share_mean = np.mean(drop_outlinear)
                        
                        # Заполняем данные
                        data.loc[mask, 'Share'] = share_mean
            
            # Проверяем результат
            unfilled = data[data['Share'] == '']
            if len(unfilled) > 0:
                print(f"ВНИМАНИЕ: Осталось незаполненных строк: {len(unfilled)}")
                print(unfilled[['День недели', 'Время выхода']])
            
            forecast[big_program] = data
        
        # Создание единого датафрейма с прогнозом для крупных программ
        res = []
        for program, df in forecast.items():
            res.append(df)
        big_forecast = pd.concat(res)
        return big_forecast
    

    def forecast_share_not_found_programs(self, new_df: pd.DataFrame, df_hist: pd.DataFrame) -> pd.DataFrame:
        """
        Функция для расчета прогноза доли для программ, для которых не было найдено похожей программы.
        
        Args:
            new_df: pd.DataFrame: DataFrame с новыми программами для прогнозирования
            df_hist: pd.DataFrame: DataFrame с историческими данными
            start_of_current_year: str: Дата старта года, от которого будем смотреть историю.
        
        Returns:
            pd.DataFrame: DataFrame с прогнозом для программ без найденных аналогов.
        """
        # Перевод даты в формат времени
        start_date = pd.to_datetime(self.start_date)

         # Получаем даты за последние 4 недели
        dates = Dates_Operations.get_last_4_weeks(start_date)

        # Создаем копию для работы
        new_df_ = new_df.copy().reset_index(drop = True)
        
        # Если нет программ для прогноза, возвращаем исходный DataFrame
        if len(new_df_) == 0:
            print("Нет программ для прогнозирования")
            return new_df_
        
        # Отбор исторических данных, начиная с текущего года
        df_hist_current = df_hist[df_hist['Дата'].astype(str) >= self.start_of_current_year]

        # Создаем столбец для прогноза, если его нет
        if 'Forecast' not in new_df_.columns:
            new_df_['Forecast'] = None
        
        # Отбираем уникальные дни недели и слоты для прогноза
        days_of_week = list(set(new_df_['День недели']))
        slots = list(set(new_df_['Время выхода']))
        
        
        # Моделируем будущие доли
        for day in days_of_week:
            for slot in slots:
                # Создаем маску для текущего дня и слота
                mask = (new_df_['День недели'] == day) & (new_df_['Время выхода'] == slot)
                rows_to_forecast = new_df_[mask]
                
                if len(rows_to_forecast) == 0:
                    continue
                
                # Ищем исторические данные для ТОГО ЖЕ дня недели и слота
                historical_mask = (df_hist_current['День недели'] == day) & \
                            (df_hist_current['Время выхода'] == slot) & \
                            (df_hist_current['Share'].notna()) & \
                            (df_hist_current['Share'] != '')
            
                historical_data = df_hist_current[historical_mask]
                
                # Если есть исторические данные по тому же слоту
                if len(historical_data) > 0:
                    #print(f"  Найдено исторических данных по слоту: {len(historical_data)}")
                    try:
                        drop_outlinear = TimeSeriesTransformer(historical_data['Share']).replace_outliers_with_median()
                        share_mean = np.mean(drop_outlinear)
                        new_df_.loc[mask, 'Forecast'] = share_mean
                        continue  # Переходим к следующей итерации
                    except Exception as e:
                        print(f"  Ошибка при расчете по историческим данным: {e}")
                
                    # Если не нашлось данных по слоту, используем последние 4 недели
                    print(f"  Не найдено данных по слоту, используем последние 4 недели...")
                
                else:    
                    # Ищем исторические данные за последние 4 недели
                    last_4_weeks_data = df_hist_current[
                        (df_hist_current['Дата'].astype(str).isin(dates)) &
                        (df_hist_current['Share'].notna()) & 
                        (df_hist_current['Share'] != '')
                    ]
                
                    #print(f"  Найдено исторических данных за 4 недели: {len(last_4_weeks_data)}")
                
                    if len(last_4_weeks_data) > 0:
                        try:
                            drop_outlinear = TimeSeriesTransformer(last_4_weeks_data['Share']).replace_outliers_with_median()
                            share_mean = np.mean(drop_outlinear)
                            new_df_.loc[mask, 'Forecast'] = share_mean
                        except Exception as e:
                            print(f"  Ошибка при расчете за 4 недели: {e}")
                            # Если вообще не получилось, используем общее среднее
                            all_shares = df_hist_current[
                                (df_hist_current['Share'].notna()) & 
                                (df_hist_current['Share'] != '')
                            ]['Share']
                            if len(all_shares) > 0:
                                share_mean = np.mean(all_shares)
                                new_df_.loc[mask, 'Forecast'] = share_mean
                            else:
                                new_df_.loc[mask, 'Forecast'] = 0.0
                    else:
                        # Если вообще нет данных, используем общее среднее
                        print("Нет данных за 4 недели, используем общее среднее")
                        all_shares = df_hist_current[
                            (df_hist_current['Share'].notna()) & 
                            (df_hist_current['Share'] != '')
                        ]
                        if len(all_shares) > 0:
                            drop_outlinear = TimeSeriesTransformer(all_shares['Share']).replace_outliers_with_median()
                            share_mean = np.mean(drop_outlinear)
                            new_df_.loc[mask, 'Forecast'] = share_mean
                        else:
                            new_df_.loc[mask, 'Forecast'] = 0.0
        
        # Проверяем результат
        unfilled = new_df_[new_df_['Forecast'].isna()]
        if len(unfilled) > 0:
            print(f"ВНИМАНИЕ: Осталось незаполненных прогнозов: {len(unfilled)}")
            print(unfilled[['День недели', 'Время выхода', 'Дата']])
        else:
            print("Все прогнозы успешно заполнены")
        
        return new_df_
    


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
    def get_next_hour(dt):
        """
            Вспомогательная функция для вычисления ближайшего следующего часа
        """
        next_hour = dt.replace(minute = 0, second = 0, microsecond = 0) + timedelta(hours = 1)
        return next_hour
    
        
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


    def calculate_weighted_share(self, data, slot_weights: dict) -> pd.DataFrame:
        """
            Функция для расчета взвешенной доли. 
            Args:
                df: pd.DataFrame: Датафрейм, в котором есть столбцы Долей (Share), Время выхода, Время окончания, Название программы. 
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
                next_hour = TVShareCalculator.get_next_hour(start_time)
                prev_hour = TVShareCalculator.get_previous_hour(end_time)
                
                # Длительность 1: от начала до следующего часа
                duration1 = next_hour - start_time
                # Длительность 2: от предыдущего часа до окончания
                duration2 = end_time - prev_hour
                
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
        res['Share_NEW'] = ''

        for i in range(len(res)):
            slot_weight_2 = 0.0
            
            start_hour = res.iloc[i]['hour_start']
            end_hour = res.iloc[i]['hour_end']
            share = res.iloc[i]['Share']
            duration_1 = res.iloc[i]['% duration 1']
            # Если есть скачок через час
            duration_2 = res.iloc[i]['% duration 2']
            slot_weight_1 = slot_weights[start_hour]

            # Если скачка через час нет, считаем долю в слоте как Share * вес слота * % длительности программы в часе
            res.at[i, 'Share_NEW'] = share * duration_1 * slot_weight_1
            # Если есть скачок через час, считаем долю по-другому
            if duration_2 != 0.0:
                slot_weight_2 = slot_weights[end_hour]
                res.at[i, 'Share_NEW'] = share * (duration_1 * slot_weight_1 + duration_2 * slot_weight_2)
        data = res[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share_NEW']]
        data.rename(columns = {'Share_NEW': 'Share'}, inplace = True)
        # Расчёт суммарной доли по дню
        share_sum = np.sum(list(data['Share']))
        return data, share_sum
