import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from calendar import monthrange
from functools import reduce

from OMA_tools.federal.requests.func import *
from OMA_tools.io_data.colors import *


import warnings
warnings.filterwarnings('ignore')


class AdvertisingCampaign:
    """
        Класс работает для каждого канала по отдельности
    """
    def __init__(self, channel_name: str, data: pd.DataFrame, month_num: int):
        self.channel_name = channel_name
        self.data = data
        self.month_num = month_num

        self.TARGET_SLOTS = [
            '07:00:00', '09:00:00', '11:00:00',
            '13:00:00', '15:00:00', '17:00:00',
            '19:00:00', '21:00:00', '23:00:00',
        ]


    def prepare_input_data(self):
        """
            Метод для подготовки данных для дальнейшего моделирования
        """
        self.data['hour_start'] = pd.to_datetime(self.data['Время выхода']).dt.hour
        self.data['hour_stop'] = pd.to_datetime(self.data['Время окончания']).dt.hour
        self.data['month'] = pd.to_datetime(self.data['ResearchDate']).dt.month
        
        # Отбираем март
        tmp_df = self.data[self.data['month'] == self.month_num].reset_index(drop = True)
        return tmp_df
    

    def divide_by_minutes(self, df):
        """
            Метод для разбивки событий по минутам.
        """
        # Преобразуем время
        df['start_dt'] = pd.to_datetime(df['ResearchDate'].astype(str) + ' ' + df['Время выхода'])
        df['end_dt'] = pd.to_datetime(df['ResearchDate'].astype(str) + ' ' + df['Время окончания'])

        # Разбиение на минуты (вес НЕ дробится)
        rows = []
        for _, row in df.iterrows():
            current = row['start_dt']
            end = row['end_dt']
            
            while current < end:
                rows.append({
                    'ResearchDate': row['ResearchDate'],
                    'SubjectID': row['SubjectID'],
                    'Channel': row['Channel'],
                    'weight_new': row['weight_new'],  # полный вес на каждую минуту
                    'Media': row['Media'],
                    'minute_start': current,
                    'minute_end': current + timedelta(minutes = 1),
                    'hour': current.hour,
                    'minute': current.minute,
                    'month': row['month']
                })
                current += timedelta(minutes = 1)

        df_minutes = pd.DataFrame(rows)
        df_minutes['Время выхода'] = pd.to_datetime(df_minutes['minute_start']).dt.strftime('%H:%M:%S')
        df_minutes['Время окончания'] = pd.to_datetime(df_minutes['minute_end']).dt.strftime('%H:%M:%S')
        df_minutes = df_minutes[['ResearchDate', 'SubjectID', 'Channel', 'weight_new', 'Media',
            'Время выхода', 'Время окончания']]

        minutely_df = GeneralClass().calculate_program_duration(df_minutes)
        minutely_df['Duration'] = pd.to_timedelta(minutely_df['Продолжительность']).dt.total_seconds()
        minutely_df = minutely_df.drop(['Время выхода_dt', 'Время окончания_dt', 'Продолжительность'], axis = 1)
        # Переводим длительность из секунд в минуты
        minutely_df['dur_min'] = minutely_df['Duration'] / 60.0
        return minutely_df
    

    def select_slots(self, df: pd.DataFrame, mean_weights_df: pd.DataFrame, debug = False):

        selected_slots_list = []
        TVR_slots_list = []
        for date in df['ResearchDate'].unique():
            
            date_data = df[df['ResearchDate'] == date]

            # Разделяем по минутам все события
            minutely = self.divide_by_minutes(date_data)
            
            selected_slots_list_per_date = []
            TVR_slots_list_per_date = []
            for slot in self.TARGET_SLOTS:
                slot_data = minutely[minutely['Время выхода'] == slot]
                
                if len(slot_data) != 0:
                    # Выбранные данные по слоту добавляем в список
                    selected_slots_list_per_date.append(slot_data)
            
                    # Считаем Audience и TVR слота
                    Audience = slot_data['weight_new'].sum()
                    slot_TVR = (Audience / 14482.13) * 100
            
                    result_per_slot = {
                        'Дата': date,
                        'Канал': self.channel_name,
                        'Время выхода': slot,
                        'Audience': Audience,
                        'Universe': 14482.13,
                        'TVR': slot_TVR,
                        'Количество респондентов': len(slot_data)
                    }
                    TVR_slots_list_per_date.append(result_per_slot)
                else:
                    if debug:
                        print(f'Для даты {date} и слота {slot} не найдено данных по респодентам.')
                    result_per_slot = {
                        'Дата': date,
                        'Канал': self.channel_name,
                        'Время выхода': slot,
                        'Audience': 0.0,
                        'Universe': 14482.13,
                        'TVR': 0.0,
                        'Количество респондентов': 0
                    }
                    TVR_slots_list_per_date.append(result_per_slot)
            
            if len(selected_slots_list_per_date) != 0:
                selected_slots_list.append( pd.concat(selected_slots_list_per_date) )
            else:
                selected_slots_list.append( pd.DataFrame() )
            
            if len(TVR_slots_list_per_date) != 0:
                TVR_slots_list.append( pd.DataFrame(TVR_slots_list_per_date) )
            else:
                TVR_slots_list.append( pd.DataFrame() )

        selected_slots_df = pd.concat(selected_slots_list).reset_index(drop = True)
        # Делаем Join с таблицей из средних весов каждого респондента
        results_selected_slots = pd.merge(selected_slots_df, mean_weights_df, on = ['SubjectID'], how = 'inner')
        
        return results_selected_slots, pd.concat(TVR_slots_list).reset_index(drop = True)


    def calculate_average_TVR_daily(self, TVR_by_slots_df: pd.DataFrame):

        result_per_date = {}
        for date in TVR_by_slots_df['Дата'].unique():
            tmp_df = TVR_by_slots_df[TVR_by_slots_df['Дата'] == date]
            TVR_mean = tmp_df['TVR'].sum() / len(self.TARGET_SLOTS)
            result_per_date[date] = TVR_mean
            
        daily_mean_TVR = pd.DataFrame(list(result_per_date.items()), columns = ['Дата', f'TVR средний {self.channel_name}'])
        return daily_mean_TVR
    

class Simulation:
    def __init__(self, target_grp_list, media_name: str, media_dict: dict):
        self.target_grp_list = target_grp_list
        self.media_name = media_name
        self.media_dict = media_dict


        if media_name not in ['Только ТВ', 'Только Радио', 'ТВ+Радио']:
            raise ValueError(f"Выберите медиа из списка: {'Только ТВ', 'Только Радио', 'ТВ+Радио'}.")

        self.FEDERAL_CHANNELS = [
            'ПЕРВЫЙ КАНАЛ', 'РОССИЯ 1', 'НТВ', 'ТНТ', 'СТС', 'РЕН ТВ', 'ПЯТЫЙ КАНАЛ',
            'ДОМАШНИЙ', 'ТВ-3', 'ПЯТНИЦА', 'РОССИЯ 24','КАРУСЕЛЬ', 'ЗВЕЗДА', 'МАТЧ ТВ', 'ТВ ЦЕНТР',
            'СПАС', 'МИР', 'ЧЕ', 'МУЗ ТВ', 'СОЛНЦЕ', 'ТНТ 4', '2X2', 'СТС LOVE', 'Ю', 'СУББОТА'
        ]

        self.RADIO = [
            'МАЯК', 'РАДИО ДЖАЗ', 'ДЕТСКОЕ РАДИО', 'ДОРОЖНОЕ РАДИО', 'РАДИО РЕКОРД', 'MAXIMUM', 'КОММЕРСАНТЪFM', 
            'COMEDY RADIO', 'ROCK FM', 'РАДИО SPUTNIK', 'ЕВРОПА ПЛЮС', 'АВТОРАДИО','РАДИО МОНТЕ-КАРЛО', 'ГОВОРИТ МОСКВА', 
            'КАЛИНА КРАСНАЯ', 'ПЕРВОЕ СПОРТИВНОЕ', 'ОРФЕЙ', 'РАДИО ДАЧА', 'BUSINESS FM', 'СЕРЕБРЯНЫЙ ДОЖДЬ', 'ВОСТОК FM', 
            'РАДИО ROMANTIKA', 'МАРУСЯ FM', 'ТАКСИ FM', 'LIKE FM', 'РАДИО КУЛЬТУРА', 'РАДИО ЭНЕРДЖИ (NRG)', 'МИЛИЦЕЙСКАЯ ВОЛНА',
            'РАДИО КП', 'РАДИО ГОРДОСТЬ', 'РАДИО РБК', 'НАШЕ РАДИО', 'ЖАРА FM', 'РУССКИЙ ХИТ', 'РЕТРО FM', 'РУССКОЕ РАДИО',
            'РАДИО ШОКОЛАД', 'DFM', 'РАДИО РОССИИ', 'РАДИО ШАНСОН', 'STUDIO 21', 'РАДИО МОСКВЫ', 'LOVE RADIO', 'RELAX FM',
            'МОСКВА FM', 'РАДИО 7 НА СЕМИ ХОЛМАХ', 'ХИТ FM', 'НОВОЕ РАДИО', 'РАДИО ЗВЕЗДА','РАДИО ВЕРА', 'ЮМОР FM', 'ВЕСТИ FM'
        ]

        self.BOTH_MEDIA = self.FEDERAL_CHANNELS + self.RADIO

        self.TARGET_SLOTS = [
            '07:00:00', '09:00:00', '11:00:00',
            '13:00:00', '15:00:00', '17:00:00',
            '19:00:00', '21:00:00', '23:00:00',
        ]

        self.data_dict = {
            'Только ТВ': self.FEDERAL_CHANNELS,
            'Только Радио': self.RADIO,
            'ТВ+Радио': self.BOTH_MEDIA
        }
    

    def TVR_calculation(self, resps_weights_mean_df: pd.DataFrame, num_month: int = 3):
        """
            Метод для расчета TVR

            Параметры:
            ----------
                resps_weights_mean_df: pd.DataFrame
                    Таблица со средними весами респондентов
                num_month: int
                    Месяц, для которого будем делать моделирование
        """
        channels_result = {}

        slots_list = []         # список с отобранными слотами и респондентами
        TVR_by_slots_list = []  # список рассчитанными рейтингами слотов для каждого дня
        TVR_daily_list = []     # список с рассчитанными СРЕДНИМИ рейтингами слотов для каждого дня

        for channel, data in self.media_dict.items():
            print(f'========= Расчет канала {channel} =========')
            # Подготавливаем данные для анализа
            adv = AdvertisingCampaign(channel, data, num_month)
            tmp_df = adv.prepare_input_data()

            # Отбор интересующих слотов, а также расчет рейтингов слотов на основании Audience и TVR
            selected_slots_df, slots_TVR_df = adv.select_slots(tmp_df, resps_weights_mean_df)  
            # Расчет средних рейтингов слотов для каждого дня. Результат записывается в DataFrame
            daily_mean_TVR = adv.calculate_average_TVR_daily(slots_TVR_df)

            slots_list.append(selected_slots_df)
            TVR_by_slots_list.append(slots_TVR_df)
            TVR_daily_list.append(daily_mean_TVR)

        selected_slots_general = pd.concat(slots_list).reset_index(drop = True)
        selected_slots_general.rename(columns = {'ResearchDate': 'Дата', 'Channel': 'Канал'}, inplace = True)


        TVR_by_slots_general = pd.concat(TVR_by_slots_list).reset_index(drop = True)
        TVR_mean_general = reduce(
            lambda left, right: pd.merge(left, right, on = 'Дата', how = 'outer'),
            TVR_daily_list
        )

        # Расчет суммарного рейтинга для каждого дня
        TVR_mean_general['ИТОГО'] = TVR_mean_general[TVR_mean_general.columns[1:]].sum(axis = 1)
                
        Summary_TVR = TVR_mean_general[['Дата', 'ИТОГО']]
        # Расчет суммарного СРЕДНЕГО рейтинга за месяц
        TVR_mean_per_month = Summary_TVR['ИТОГО'].sum() / 31.0

        return selected_slots_general, TVR_by_slots_general, TVR_mean_per_month

    
    def round_math(self, value: float, decimals: int = 0):
        """
            Математическое округление числа.
            
            Параметры:
            ----------
                value: float
                    Число для округления
                decimals: int
                    Количество знаков после запятой (по умолчанию 0)
            
            Returns:
            ----------
                Округленное число
        """
        multiplier = 10 ** decimals
        return int(value * multiplier + (0.5 if value >= 0 else -0.5)) / multiplier


    def calculate_number_of_airings(self, TVR_mean: float, debug: bool = True):
        """
            Метод рассчитывает необходимое количество выпусков рекламных роликов в месяц, а также на каждом канале в зависимости от целевого GRP.
        """
        if TVR_mean <= 0:
            raise ValueError(f"TVR_mean должен быть положительным, получено {TVR_mean}")
        
        channels = []
        # Выбор списка каналов в зависимости от типа медиа
        if self.media_name == 'Только ТВ':
            channels = self.FEDERAL_CHANNELS
        elif self.media_name == 'Только Радио':
            channels = self.RADIO
        elif self.media_name == 'ТВ+Радио':
            channels = self.BOTH_MEDIA
        else:
            raise ValueError(f"Неизвестный тип медиа: {self.media_name}")
        
        channels_count = len(channels)
        slots_count = len(self.TARGET_SLOTS)
        max_outputs_per_channel = slots_count * 31

        required_outputs_dict = {}

        for GRP in self.target_grp_list:

            outputs_per_channel = 0.0

            # Расчет количества выходов рекламного ролика в месяц
            total_outputs = self.round_math(GRP / TVR_mean)

            # Расчет количества выпусков на каждом канале
            outputs_per_channel = self.round_math(total_outputs / channels_count)

            # Проверка на превышение максимального количества
            if outputs_per_channel > max_outputs_per_channel:
                if debug:
                    print(f'Для целевого GRP {GRP}: требуемое количество {outputs_per_channel} превышает '
                    f'максимально возможное {max_outputs_per_channel}. Будет использовано {max_outputs_per_channel}')
                outputs_per_channel = max_outputs_per_channel
                required_outputs_dict[GRP] = [int(total_outputs), int(max_outputs_per_channel)]
            
            else:
                required_outputs_dict[GRP] = [int(total_outputs), int(outputs_per_channel)]
            
            if debug:
                print(f'Для целевого GRP {GRP} кол-во выходов в месяц равно {total_outputs} шт., на каждом канале {outputs_per_channel} шт.')
            
        
        required_outputs = pd.DataFrame.from_dict(required_outputs_dict, orient = 'index', columns = ['Кол-во выходов в месяц', 'Кол-во выходов на каждом канале'])
        required_outputs.index.name = 'Целевое GRP'
        required_outputs = required_outputs.reset_index()
        
        return required_outputs
    

    def calculate_GRP_fact(self, time_table_dict: dict, TVR_by_slots: pd.DataFrame):
        """
            Метод для расчета фактического GRP рекламной кампании, основываясь на рейтингах слотов, в которые размещается реклама.

            Параметры:
            ----------
                time_table_dict: dict
                    Словарь с распределением количества выходов по каналам и слотам
                TVR_by_slots: pd.DataFrame
                    Таблица с рейтингами каждого слота по каналам
            
            Returns:
            ----------
                GRP_fact_df: pd.DataFrame
                        Таблица с результатами расчета Фактического GRP кампании. Содержит столбцы: Кол-во выходов на каждом канале, GRP факт
        """
        # Расчет фактического GRP, основываясь на количестве выходов на каждом канале, а также на TVR слотов
        GRP_fact_dict = {}

        for number_of_outputs, time_table in time_table_dict.items():
            if number_of_outputs == 0:
                GRP_fact_dict[number_of_outputs] = 0
            else:
                #time_table_new = time_table.rename(columns = {'Время': 'Время выхода'})
                res_df = pd.merge(time_table, TVR_by_slots, on = ['Дата', 'Время выхода', 'Канал'], how = 'inner')
                GRP_fact_dict[number_of_outputs] = res_df['TVR'].sum()

        GRP_fact_df = pd.DataFrame(list(GRP_fact_dict.items()), columns = ['Кол-во выходов на каждом канале', 'GRP факт'])
        return GRP_fact_df
    

    def cum_reach(self, time_table_dict: dict, selected_slots_df: pd.DataFrame):
        """
            Параметры:
            ----------
                time_table_dict: dict
                    Словарь с распределением количества выходов по каналам и слотам
                selected_slots_df: pd.DataFrame
                    Таблица с выбранными слотами для анализ
            
            Returns:
            ----------
                reach_df: pd.DataFrame
                    Таблица с результатами расчета Reach. Содержит столбцы: Кол-во выходов на каждом канале, Reach
        """
        # Расчет накопленного охвата рекламной компании
        reach_dict = {}

        for number_of_ouputs, time_table in time_table_dict.items():
            if number_of_ouputs == 0:
                reach_dict[number_of_ouputs] = 0.0
            else:
                tmp_df = pd.merge(time_table, selected_slots_df, on = ['Дата', 'Время выхода', 'Канал'], how = 'inner')
                unique_resps = tmp_df.drop_duplicates(subset = ['SubjectID'], keep = 'first')
                reach_dict[number_of_ouputs] = unique_resps['weight_mean'].sum()

        reach_df = pd.DataFrame(list(reach_dict.items()), columns = ['Кол-во выходов на каждом канале', 'Reach'])
        return reach_df


    def simulation_pipeline(self, resps_weights_mean_df: pd.DataFrame, target_date: str = '01.03.2026'):
        """
            Полный пайплайн для моделирования рекламной кампании (Анализ ТВ & Радио)

            Параметры:
            ----------
                resps_weights_mean_df: pd.DataFrame
                    Таблица со средними весами респондентов
                
                target_date: str
                    Произвольная дата месяца для получения количества дней
            
            Returns:
            ----------
                required_outputs_df: pd.DataFrame
                    Таблица с результатами моделирования
        """
        # ШАГ 1. Расчет TVR
        selected_slots_df, TVR_by_slots_df, TVR_mean_per_month = self.TVR_calculation(resps_weights_mean_df)

        # Шаг 2. Расчет необходимого количества выходов роликов
        required_outputs_df = self.calculate_number_of_airings(TVR_mean_per_month)

        # Шаг 3. РАСПРЕДЕЛЕНИЕ РЕКЛАМНЫХ БЛОКОВ ПО ДНЯМ И СЛОТАМ. ПОЛУЧАЕМ РАСПИСАНИЕ (time_table_dict)¶
        break_points = BreakPointGeneration(target_date, self.media_name)
        time_table_dict = break_points.generate_distribution_pipeline( list(required_outputs_df['Кол-во выходов на каждом канале']) )

        # Шаг 4. Расчет фактического GRP, основываясь на количестве выходов на каждом канале, а также на TVR слотов
        GRP_fact_df = self.calculate_GRP_fact(time_table_dict, TVR_by_slots_df)
        required_outputs_df = pd.merge(required_outputs_df, GRP_fact_df, on = ['Кол-во выходов на каждом канале'], how = 'inner')

        # Шаг 5. Расчет накопленного Reach
        reach_df = self.cum_reach(time_table_dict, selected_slots_df)
        required_outputs_df = pd.merge(required_outputs_df, reach_df, on = ['Кол-во выходов на каждом канале'], how = 'inner')

        #required_outputs_df['GRP факт'] = required_outputs_df['GRP факт'].round(2)
        #required_outputs_df['Reach'] = required_outputs_df['Reach'].round(2)

        return required_outputs_df



class BreakPointGeneration:
    """
        Класс для генерации точек дней и слотов.
    """
    def __init__(self, target_date: str, media: str):
        self.target_date = target_date
        self.media = media

        # Парсим дату
        date_obj = pd.to_datetime(target_date, format='%d.%m.%Y')
        self.year = date_obj.year
        self.month = date_obj.month

        # Количество дней в месяце
        num_of_days = monthrange(self.year, self.month)[1]
        # Последовательность дней
        self.dates_list = list(range(1, num_of_days + 1))

        # Создаём список дат марта 2026 в формате "2026-03-01"
        self.dates = [f"{self.year}-{self.month:02d}-{day:02d}" for day in self.dates_list]

        self.TARGET_SLOTS = [
            '07:00:00', '09:00:00', '11:00:00',
            '13:00:00', '15:00:00', '17:00:00',
            '19:00:00', '21:00:00', '23:00:00',
        ]
        self.slots_list = [int(slot[:2]) for slot in self.TARGET_SLOTS]


        self.FEDERAL_CHANNELS = [
            'ПЕРВЫЙ КАНАЛ', 'РОССИЯ 1', 'НТВ', 'ТНТ', 'СТС', 'РЕН ТВ', 'ПЯТЫЙ КАНАЛ',
            'ДОМАШНИЙ', 'ТВ-3', 'ПЯТНИЦА', 'РОССИЯ 24','КАРУСЕЛЬ', 'ЗВЕЗДА', 'МАТЧ ТВ', 'ТВ ЦЕНТР',
            'СПАС', 'МИР', 'ЧЕ', 'МУЗ ТВ', 'СОЛНЦЕ', 'ТНТ 4', '2X2', 'СТС LOVE', 'Ю', 'СУББОТА'
        ]

        self.RADIO = [
            'МАЯК', 'РАДИО ДЖАЗ', 'ДЕТСКОЕ РАДИО', 'ДОРОЖНОЕ РАДИО', 'РАДИО РЕКОРД', 'MAXIMUM', 'КОММЕРСАНТЪFM', 
            'COMEDY RADIO', 'ROCK FM', 'РАДИО SPUTNIK', 'ЕВРОПА ПЛЮС', 'АВТОРАДИО','РАДИО МОНТЕ-КАРЛО', 'ГОВОРИТ МОСКВА', 
            'КАЛИНА КРАСНАЯ', 'ПЕРВОЕ СПОРТИВНОЕ', 'ОРФЕЙ', 'РАДИО ДАЧА', 'BUSINESS FM', 'СЕРЕБРЯНЫЙ ДОЖДЬ', 'ВОСТОК FM', 
            'РАДИО ROMANTIKA', 'МАРУСЯ FM', 'ТАКСИ FM', 'LIKE FM', 'РАДИО КУЛЬТУРА', 'РАДИО ЭНЕРДЖИ (NRG)', 'МИЛИЦЕЙСКАЯ ВОЛНА',
            'РАДИО КП', 'РАДИО ГОРДОСТЬ', 'РАДИО РБК', 'НАШЕ РАДИО', 'ЖАРА FM', 'РУССКИЙ ХИТ', 'РЕТРО FM', 'РУССКОЕ РАДИО',
            'РАДИО ШОКОЛАД', 'DFM', 'РАДИО РОССИИ', 'РАДИО ШАНСОН', 'STUDIO 21', 'РАДИО МОСКВЫ', 'LOVE RADIO', 'RELAX FM',
            'МОСКВА FM', 'РАДИО 7 НА СЕМИ ХОЛМАХ', 'ХИТ FM', 'НОВОЕ РАДИО', 'РАДИО ЗВЕЗДА','РАДИО ВЕРА', 'ЮМОР FM', 'ВЕСТИ FM'
        ]

        self.BOTH_MEDIA = self.FEDERAL_CHANNELS + self.RADIO

        self.media_dict = {
            'Только ТВ': self.FEDERAL_CHANNELS,
            'Только Радио': self.RADIO,
            'ТВ+Радио': self.BOTH_MEDIA
        }

        # Счётчик использованных каналов (для равномерного распределения)
        self.channel_usage = {channel: 0 for channel in self.media_dict[self.media]}


    def get_symmetric_days(self, n_days):
        """
            Возвращает n_days симметрично распределённых дней
        """
        if n_days == 1:
            return [self.dates_list[len(self.dates_list) // 2]]
        
        step = len(self.dates_list) / n_days
        indices = [int(i * step) for i in range(n_days)]
        return [self.dates_list[i] for i in indices]


    def get_symmetric_slots(self, n_slots):
        """
            Возвращает n_slots симметрично распределённых слотов
        """
        if n_slots == 1:
            return [self.slots_list[len(self.slots_list) // 2]]
        
        step = len(self.slots_list) / n_slots
        indices = [int(i * step) for i in range(n_slots)]
        return [self.slots_list[i] for i in indices]
    

    def get_slot_pairs(self, n_pairs):
        """
            Возвращает n_pairs симметричных пар слотов
        """
        pairs = []
        for i in range(n_pairs):
            left = self.slots_list[i % len(self.slots_list)]
            right = self.slots_list[-(i % len(self.slots_list)) - 1]
            pairs.append([left, right])
        return pairs
    

    def get_slot_triplets(self, n_triplets):
        """
            Возвращает n_triplets троек слотов
        """
        triplets = []
        for i in range(n_triplets):
            triplet = [
                self.slots_list[i % len(self.slots_list)],
                self.slots_list[(i + 3) % len(self.slots_list)],
                self.slots_list[(i + 6) % len(self.slots_list)]
            ]
            triplets.append(triplet)
        return triplets
    

    def get_middle_day(self, days, total_days):
        """
            Возвращает день между существующими днями
        """
        sorted_days = sorted(days)
        for i in range(len(sorted_days) - 1):
            gap = sorted_days[i + 1] - sorted_days[i]
            if gap > 1:
                return sorted_days[i] + gap // 2
        return total_days // 2
    

    def get_symmetric_breakpoints(self, target_list: list, max_value: int = None):
        """
            Метод для поиска медиан в последовательности чисел
        """
        # Медианная дата
        median_center = int(np.median(target_list))

        # Определяем максимальное значение
        if max_value is None:
            max_value = max(target_list)
        
        # Первая часть дат ДО медианы
        part_to_median = list(range(1, median_center))
        
        # Поиск медианы в этом куске
        if part_to_median:
            median_left = int(np.median(part_to_median))
        else:
            median_left = 1
        
        # Вторая часть дат ПОСЛЕ медианы
        part_after_median = list(range(median_center + 1, max_value))
        # Поиск медианы в этом куске
        if part_after_median:
            median_right = int(np.median(part_after_median))
        else:
            median_right = max_value

        return  median_center, median_left, median_right
    

    def delete_elements(self, data_list: list, n_to_remove: int):
        """
            Метод рандомно удаляет определенное количество элементов в массиве.
        """
        if n_to_remove > len(data_list):
            raise ValueError(
                    f"Вы хотите удалить элементов больше ({n_to_remove} шт.), чем есть в массиве ({len(data_list)} шт.)."
            )
        to_remove = random.sample(data_list, n_to_remove)
        # Удаляем их
        result = [num for num in data_list if num not in to_remove]
        return result
    

    def distribute_36_outputs(self, total_blocks):
        """
        Специальное распределение для 36 выходов:
        - 12 дней × 3 выхода
        - или 9 дней × 4 выхода
        - с разной интенсивностью
        """
        # Определяем, сколько дней удалить в зависимости от количества блоков
        if total_blocks >= 35:
            days_to_remove = 0

        elif total_blocks >= 30:
            days_to_remove = 3

        elif total_blocks >= 25:
            days_to_remove = 4

        elif total_blocks >= 20:
            days_to_remove = 10

        elif total_blocks >= 15:
            days_to_remove = 15

        elif total_blocks >= 10:
            days_to_remove = 20

        else:
            days_to_remove = 20

        # Удаляем случайные дни
        remaining_days = self.delete_elements(self.dates_list, days_to_remove)
        n_days = len(remaining_days)
        
        # Минимум по 1 блоку на каждый день
        blocks = [1] * n_days
        current_total = sum(blocks)
        
        # Если нужно добавить блоки
        if current_total < total_blocks:
            remaining = total_blocks - current_total
            days_indices = list(range(n_days))
            
            while remaining > 0:
                random.shuffle(days_indices)
                for idx in days_indices:
                    if remaining <= 0:
                        break
                    if blocks[idx] < 9:
                        blocks[idx] += 1
                        remaining -= 1
        
        # Если получилось больше блоков, чем нужно (редко, но бывает)
        current_total = sum(blocks)
        if current_total > total_blocks:
            #print(f"⚠️ Внимание! Получилось {current_total} блоков, а нужно {total_blocks}. Удаляем лишние.")
            
            # Создаём список всех блоков с их позициями
            all_blocks = []
            for idx, day in enumerate(remaining_days):
                for _ in range(blocks[idx]):
                    all_blocks.append((idx, day))
            
            # Перемешиваем и удаляем лишние
            random.shuffle(all_blocks)
            blocks_to_remove = current_total - total_blocks
            to_remove_set = set(all_blocks[:blocks_to_remove])
            
            # Пересчитываем blocks
            new_blocks = [0] * n_days
            for idx, day in all_blocks[blocks_to_remove:]:
                new_blocks[idx] += 1
            
            blocks = new_blocks
            
            # Удаляем дни, где стало 0 блоков
            remaining_days_new = []
            blocks_new = []
            for day, block_count in zip(remaining_days, blocks):
                if block_count > 0:
                    remaining_days_new.append(day)
                    blocks_new.append(block_count)
            
            remaining_days = remaining_days_new
            blocks = blocks_new
            n_days = len(remaining_days)
        
        # Финальная проверка
        final_total = sum(blocks)
        
        if final_total != total_blocks:
            raise ValueError(f"Не удалось распределить блоки: получилось {final_total}, нужно {total_blocks}")
        
        # Словарь для хранения слотов
        result = {}
        
        # Счётчик использования каждого слота (для равномерного распределения)
        slot_usage = {slot: 0 for slot in self.slots_list}
        
        for day, n_slots in zip(remaining_days, blocks):
            # Выбираем n_slots наименее использованных слотов
            available_slots = sorted(self.slots_list, key=lambda x: slot_usage[x])
            selected_slots = sorted(available_slots[:n_slots])
            
            result[day] = selected_slots
            
            # Обновляем счётчики
            for slot in selected_slots:
                slot_usage[slot] += 1
        
        return result



    def distribute_outputs_for_channel(self, outputs_per_channel: int) -> dict:
        """
        Распределение выходов для ОДНОГО канала.
        Возвращает словарь {день: [список_слотов]} для одного канала.
        """
        if outputs_per_channel == 0:
            return {}
        
        total_days = len(self.dates_list)
        median_date, left_median, right_median = self.get_symmetric_breakpoints(self.dates_list)
        
        # 1-3 выхода
        if outputs_per_channel <= 3:
            if outputs_per_channel == 1:
                return {median_date: [self.slots_list[len(self.slots_list) // 2]]}
            elif outputs_per_channel == 2:
                return {left_median: [self.slots_list[0]], 
                        right_median: [self.slots_list[-1]]}
            else:  # 3
                return {
                    left_median: [self.slots_list[0]],
                    median_date: [self.slots_list[len(self.slots_list) // 2]],
                    right_median: [self.slots_list[-1]]
                }
        
        # 4-18 выходов
        elif outputs_per_channel <= 18:
            if outputs_per_channel <= 7:
                # По 1 выходу в день
                days = self.get_symmetric_days(outputs_per_channel)
                slots = self.get_symmetric_slots(outputs_per_channel)
                return {day: [slot] for day, slot in zip(days, slots)}
            else:
                # По 2 выхода в день
                n_days = outputs_per_channel // 2
                days = self.get_symmetric_days(n_days)
                slot_pairs = self.get_slot_pairs(n_days)
                result = {day: slot_pair for day, slot_pair in zip(days, slot_pairs)}
                
                if outputs_per_channel % 2 == 1:
                    extra_day = self.get_middle_day(days, total_days)
                    extra_slot = [self.slots_list[len(self.slots_list) // 2]]
                    result[extra_day] = extra_slot
                
                return result
        
        # 19-36 выходов
        else:
            return self.distribute_36_outputs(outputs_per_channel)
        

    def distribute_all_channels(self, outputs_per_channel: int) -> dict:
        """
        Распределяет выходы для ВСЕХ каналов.
        Каждый канал получает одинаковое количество выходов (outputs_per_channel).
        
        Returns:
        - Словарь {канал: {день: [список_слотов]}}
        """
        all_channels_distribution = {}
        
        for channel in self.media_dict[self.media]:
            # Для каждого канала генерируем своё распределение
            distribution = self.distribute_outputs_for_channel(outputs_per_channel)
            all_channels_distribution[channel] = distribution
        
        return all_channels_distribution
    

    def generate_distribution_pipeline(self, outputs_per_channel_list: list) -> dict:
        """
        Пайплайн для генерации распределения.
        
        Параметры:
        - outputs_per_channel_list: список с количеством выходов НА КАНАЛ для каждого варианта
          Например: [2, 4, 5, 7, 9, 11, 12, 14, 16, 18, 36]
          Это означает, что каждый канал получит 2, 4, 5 и т.д. выходов в месяц
        
        Returns:
        - Словарь датафреймов, где ключ - количество выходов на канал,
          значение - DataFrame с колонками [Дата, Время, Канал]
        """
        time_table_dict = {}
        
        for outputs_per_channel in outputs_per_channel_list:
            # Для каждого варианта получаем распределение по всем каналам
            all_channels_dist = self.distribute_all_channels(outputs_per_channel)
            
            # Собираем все выходы в один список
            all_outputs = []
            
            for channel, distribution in all_channels_dist.items():
                for day, slots in distribution.items():
                    # Конвертируем слоты в строковый формат
                    slots_str = [f"{slot:02d}:00:00" if isinstance(slot, int) else slot 
                                for slot in slots]
                    
                    for slot in slots_str:
                        all_outputs.append({
                            'Дата': f"{self.year}-{self.month:02d}-{day:02d}",
                            'Время выхода': slot,
                            'Канал': channel
                        })
            
            # Создаём DataFrame
            df = pd.DataFrame(all_outputs)
            
            # Сортируем по дате и времени
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['Дата'] + ' ' + df['Время выхода'])
                df = df.sort_values('datetime').drop('datetime', axis = 1)
                df = df.reset_index(drop = True)
            
            time_table_dict[outputs_per_channel] = df
        
        return time_table_dict