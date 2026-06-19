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
        Класс для моделирования рекламной кампании
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
        #tmp_df = tmp_df[tmp_df['hour_start'] >= 7].reset_index(drop = True)
        return tmp_df


    @staticmethod
    def round_math(value, decimals = 0):
        """
            Математическое округление числа.
            
            Параметры:
            - value: число для округления
            - decimals: количество знаков после запятой (по умолчанию 0)
            
            Возвращает:
            - Округленное число
        """
        multiplier = 10 ** decimals
        return int(value * multiplier + (0.5 if value >= 0 else -0.5)) / multiplier
        


    @staticmethod
    def add_one_minute(time_str: str):
        """
            Метод для добавления 1 минуты
        """
        dt = datetime.strptime(time_str, '%H:%M:%S')
        dt = dt + timedelta(minutes = 1)
        return dt.strftime('%H:%M:%S')


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


    @staticmethod
    def find_slot_with_shift(df, target_time, max_shift = 60):
        """
            Метод для поиска слота с учетом сдвига.
            Ищет слот в DataFrame. Если не находит, пробует со сдвигом от 1 до max_shift минут.
            
            Параметры:
            - df: DataFrame с данными
            - target_time: целевое время (строка 'HH:MM:SS')
            - max_shift: максимальный сдвиг в минутах (по умолчанию 10)
            
            Возвращает:
            - (row, shift) или (None, None)
        """
        # Сначала пробуем точное совпадение
        result = df[df['Время выхода'] == target_time]
        if not result.empty:
            return result.iloc[0], 0
        
        # Преобразуем целевое время в datetime.time для удобства
        target_dt = datetime.strptime(target_time, '%H:%M:%S')
        
        # Пробуем сдвиги в обе стороны (по возрастанию абсолютного значения)
        for shift in range(1, max_shift + 1):
            for direction in [1, -1]:
                current_shift = shift * direction
                
                # Вычисляем новое время
                shifted_dt = target_dt + timedelta(minutes=current_shift)
                search_time = shifted_dt.strftime('%H:%M:%S')
                
                result = df[df['Время выхода'] == search_time]
                if not result.empty:
                    return result.iloc[0], current_shift
        
        return None, None


    def select_and_calculate_tvr(self, df, debug=False):
        """
        Метод для отбора нужных слотов и расчета TVR.
        weight_new уже содержит сумму весов респондентов в слоте = Audience
        """
        selected_slots = []
        missing_slots = []
        shifted_slots = []
        
        for i, target in enumerate(self.TARGET_SLOTS):
            # Ищем ВСЕ слоты в исходных данных
            found_rows, shifts = AdvertisingCampaign.find_all_slots_with_shift(df, target)
            
            # Если найдены слоты
            if found_rows:
                # Агрегируем все найденные строки для одного целевого слота
                base_row = self.aggregate_rows(found_rows, shifts, target)
                
                # Добавляем информацию о сдвиге в отчёт
                for row, shift in zip(found_rows, shifts):
                    if shift != 0:
                        shifted_slots.append({
                            'target': target,
                            'found': row['Время выхода'],
                            'shift': shift
                        })
                
                selected_slots.append(base_row)
            else:
                # Слот не найден - создаём строку-заглушку
                if len(selected_slots) > 0:
                    base_row = selected_slots[-1].copy()
                else:
                    base_row = pd.Series(index=df.columns, dtype=object)
                    if len(df) > 0:
                        for col in ['ResearchDate', 'Channel', 'Media']:
                            if col in df.columns:
                                base_row[col] = df.iloc[0][col]
                
                # Устанавливаем оригинальное время = целевое
                base_row['Время выхода'] = target
                base_row['Целевое время'] = target
                base_row['Сдвиг (мин)'] = 0
                base_row['Количество респондентов'] = 0
                base_row['weight_new'] = 0  # Audience = 0
                
                # Вычисляем время окончания
                try:
                    end_time = (pd.to_datetime(target) + timedelta(minutes=1)).time()
                    base_row['Время окончания'] = end_time.strftime('%H:%M:%S')
                except:
                    base_row['Время окончания'] = target
                
                selected_slots.append(base_row)
                missing_slots.append(target)
        
        # Создаём датафрейм с отобранными слотами
        df_selected = pd.DataFrame(selected_slots).reset_index(drop=True)
        df_selected.rename(columns={'Время выхода': 'Время выхода init', 'Целевое время': 'Время выхода'}, inplace=True)
        
        # РАСЧЕТ TVR
        # Группируем по времени выхода и агрегируем аудиторию
        tvr_data = []
        
        for time in df_selected['Время выхода'].unique():
            # Берём все записи для этого временного слота
            slot_data = df_selected[df_selected['Время выхода'] == time]
            
            # Audience = сумма weight_new по всем респондентам в слоте
            audience = np.sum(slot_data['weight_new'])
            
            # Берём время окончания из первой записи
            end_time = slot_data['Время окончания'].iloc[0] if len(slot_data) > 0 else ''
            
            # Количество записей (респондентов) в слоте
            respondents_count = len(slot_data)
            
            tvr_data.append({
                'Время выхода': time,
                'Время окончания': end_time,
                'Audience': audience,
                'Количество респондентов': respondents_count,
                'Universe': 14482.13  # Можно вынести в параметр класса
            })
        
        # Создаём финальный датафрейм
        res_df = pd.DataFrame(tvr_data)
        
        # Рассчитываем TVR
        res_df['TVR'] = (res_df['Audience'] / res_df['Universe']) * 100
        
        # Добавляем информацию о пропущенных слотах
        res_df['Слот найден'] = res_df['Время выхода'].apply(
            lambda x: x not in missing_slots
        )
        
        # Сортируем по времени
        res_df = res_df.sort_values('Время выхода').reset_index(drop=True)
        
        # Переупорядочиваем колонки
        res_df = res_df[['Время выхода', 'Время окончания', 'Audience', 
                        'Количество респондентов', 'Universe', 'TVR', 'Слот найден']]
        
        if debug:
            print("=== ОТОБРАННЫЕ СЛОТЫ ===\n")
            display_cols = ['Время выхода init', 'Время выхода', 'Сдвиг (мин)', 
                            'Количество респондентов', 'weight_new', 'Channel']
            existing_cols = [col for col in display_cols if col in df_selected.columns]
            if existing_cols:
                print(df_selected[existing_cols].to_string(index=False))
            
            print(f"\n=== ИТОГО ===")
            print(f"Всего целевых слотов: {len(self.TARGET_SLOTS)}")
            print(f"Найдено в исходных данных: {len(self.TARGET_SLOTS) - len(missing_slots)}")
            print(f"Найдено со сдвигом: {len(shifted_slots)}")
            print(f"Не найдено (заполнены 0): {len(missing_slots)}")
            
            if shifted_slots:
                print("\n=== СЛОТЫ СО СДВИГОМ ===")
                for s in shifted_slots[:10]:
                    print(f"Искали: {s['target']} → Нашли: {s['found']} (сдвиг {s['shift']} мин)")
            
            if missing_slots:
                print(f"\n=== НЕ НАЙДЕНЫ (заполнены 0) ===")
                print(missing_slots)
            
            print("\n=== TVR РЕЗУЛЬТАТЫ ===\n")
            print(res_df.to_string(index=False))
        
        return df_selected, res_df
    

    def aggregate_rows(self, rows, shifts, target):
        """
        Агрегирует несколько найденных строк для одного целевого слота
        weight_new уже является суммой весов = Audience
        """
        # Берём первую строку как основу (для нечисловых полей)
        base_row = rows[0].copy()
        
        # Суммируем weight_new по всем респондентам в слоте
        if 'weight_new' in base_row.index:
            base_row['weight_new'] = sum(row['weight_new'] for row in rows)
        
        # Добавляем мета-информацию
        base_row['Целевое время'] = target
        base_row['Сдвиг (мин)'] = shifts[0] if shifts else 0
        base_row['Количество респондентов'] = len(rows)
        
        # Вычисляем время окончания
        try:
            end_time = (pd.to_datetime(target) + timedelta(minutes=1)).time()
            base_row['Время окончания'] = end_time.strftime('%H:%M:%S')
        except:
            base_row['Время окончания'] = target
        
        return base_row

    @staticmethod
    def find_all_slots_with_shift(df, target_time, max_shift_minutes=3):
        """
        Находит ВСЕ слоты, соответствующие целевому времени со сдвигом до max_shift_minutes
        Возвращает: (список строк, список сдвигов)
        """
        found_rows = []
        shifts = []
        
        # Сначала ищем точное совпадение (сдвиг 0)
        exact_matches = df[df['Время выхода'] == target_time]
        if not exact_matches.empty:
            for _, row in exact_matches.iterrows():
                found_rows.append(row)
                shifts.append(0)
            return found_rows, shifts  # Если есть точное совпадение, возвращаем только его
        
        # Ищем со сдвигом
        target_dt = pd.to_datetime(target_time, format='%H:%M:%S')
        
        for shift in range(1, max_shift_minutes + 1):
            # Поиск со сдвигом +shift минут
            plus_time = (target_dt + timedelta(minutes=shift)).strftime('%H:%M:%S')
            plus_matches = df[df['Время выхода'] == plus_time]
            if not plus_matches.empty:
                for _, row in plus_matches.iterrows():
                    found_rows.append(row)
                    shifts.append(shift)
            
            # Поиск со сдвигом -shift минут
            minus_time = (target_dt - timedelta(minutes=shift)).strftime('%H:%M:%S')
            minus_matches = df[df['Время выхода'] == minus_time]
            if not minus_matches.empty:
                for _, row in minus_matches.iterrows():
                    found_rows.append(row)
                    shifts.append(-shift)
            
            if found_rows:
                break
        
        return found_rows, shifts


    def make_campaign(self, debug = False):
        """
            Пайплайн для моделирования
        """
        print(Color.BOLD + Color.DODGER_BLUE + f'==== НАЧИНАЮ МОДЕЛИРОВАНИЕ ДЛЯ КАНАЛА {self.channel_name}' + Color.END)


        data_by_dates = {}
        TVR_dict = {}
        minutely_dict = {}

        # Шаг 1. Предварительная подготовка данных
        tmp_df = self.prepare_input_data()

        unique_dates = tmp_df['ResearchDate'].unique()

        for date in unique_dates:
            
            if debug:
                print(Color.GREEN + f"Расчет для {date}" + Color.END)
                print('\n')
            table = tmp_df[tmp_df['ResearchDate'] == date].reset_index(drop = True)

            # Шаг 2. Производим разбивку по 1 минуте всех событий
            minutely_df = self.divide_by_minutes(table)

            # Шаг 3. Отбираем только нужные слоты из всего датафрейма для анализа
            df_selected, res_df = self.select_and_calculate_tvr(minutely_df, debug)

            # Шаг 4. Расчёт TVR через Audience слота и Universe
            #TVR_df = self.calculate_TVR(df_selected)

            mean_tvr = res_df['TVR'].sum() / len(self.TARGET_SLOTS)

            data_by_dates[date] = res_df
            TVR_dict[date] = mean_tvr

            minutely_dict[date] = df_selected
        
        TVR_df = pd.DataFrame(list(TVR_dict.items()), columns = ['Дата', f'TVR средний {self.channel_name}'])

        if debug:
            print('\n')
            print('\n')

        return data_by_dates, TVR_df, minutely_dict
    


class BreakPointGeneration:
    """
        Класс для генерации точек дней и слотов.
    """
    def __init__(self, target_date: str, channels: list):
        self.target_date = target_date
        self.channels = channels

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

        # Счётчик использованных каналов (для равномерного распределения)
        self.channel_usage = {channel: 0 for channel in self.channels}


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
            days_to_remove = 2

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
                    if blocks[idx] < 4:
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
        
        for channel in self.channels:
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
                            'Время': slot,
                            'Канал': channel
                        })
            
            # Создаём DataFrame
            df = pd.DataFrame(all_outputs)
            
            # Сортируем по дате и времени
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['Дата'] + ' ' + df['Время'])
                df = df.sort_values('datetime').drop('datetime', axis = 1)
                df = df.reset_index(drop = True)
            
            time_table_dict[outputs_per_channel] = df
        
        return time_table_dict


class SimulationClass:
    def __init__(self, media_dict: dict, channel_order: list, target_grp_list: list):
        self.media_dict = media_dict
        self.channel_order = channel_order
        self.target_grp_list = target_grp_list

    def choose_slots(self):
        """
            Функция для выбора слотов
        """
        # Константы в начало функции или класса
        DATE_COLUMN = 'Дата'
        CHANNEL_COLUMN = 'Канал'
        ORIGINAL_DATE_COLUMN = 'Date'
        
        # Отбор необходимых слотов для анализа
        results_by_channels = {}
        tvr_results = []
        minutely_results = {}

        
        for channel_name in self.media_dict.keys():
            channel_df = self.media_dict[channel_name]
            campaign = AdvertisingCampaign(channel_name, channel_df, 3)
            data_by_dates, tvr, minutely_dict = campaign.make_campaign(debug = False)
        
            results_by_channels[channel_name] = data_by_dates
            tvr_results.append(tvr)
            minutely_results[channel_name] = minutely_dict

        # Манипуляции с разбитыми на минуты слотами 
        all_dfs = []
        for minutely_by_dates in minutely_results.values():
            all_dfs.extend(minutely_by_dates.values())
        minutely_df = pd.concat(all_dfs)
        minutely_df.rename(columns = {'ResearchDate': 'Дата', 'Channel': 'Канал'}, inplace = True)

        # Формирование таблицы с рассчитанным TVR по выбранным слотам
        # Формируем итоговый DataFrame с рассчитанным TVR
        res_full = []
        for channel, slots_data in results_by_channels.items():
            res = []
            for date, df in slots_data.items():
                df_ = df.copy()
                #df.rename(columns = {'Date': 'Дата'}, inplace = True)
                df_['Дата'] = date
                res.append(df_)
        
            df_res = pd.concat(res)
            df_res['Канал'] = channel
            res_full.append(df_res)
        
        data_result_TVR = pd.concat(res_full).reset_index(drop = True)
        data_result_TVR.rename(columns = {'Date': 'Дата'}, inplace = True)

        return minutely_df, data_result_TVR, tvr_results


    def average_TVR_monthly(self, tvr_results: dict):
        """
            Функция для расчета TVR
        """
        # Расчет суммарного рейтинга на месяц
        TVR_full = reduce(lambda left, right: pd.merge(left, right, on = 'Дата', how = 'outer'), tvr_results)
        TVR_full.columns = TVR_full.columns.str.replace('TVR средний ', '', regex = False)
        TVR_full = TVR_full[['Дата'] + self.channel_order]
        TVR_full['ИТОГО'] = TVR_full[TVR_full.columns[1:]].sum(axis = 1)
        
        Summary_TVR = TVR_full[['Дата', 'ИТОГО']]
        TVR_mean_per_month = Summary_TVR['ИТОГО'].mean()
        print(f'Суммарный рейтинг за период {np.round(TVR_mean_per_month, 3)}')
        return TVR_mean_per_month


    def necessary_outputs_advertising_calculation(
                                self,
                                TVR_mean: float, 
                                debug = False
        ):
        """
            Функция для расчета необходимого количества выходов рекламной кампании на каждом канале в месяц, основываясь на целевом GRP.
        """
        # Расчет необходимого количества выходов рекламной кампании, основываясь на среднем рейтинге и целевом GRP.
        required_outputs_by_grp = {}

        for GRP in self.target_grp_list:
            
            # Общее количество выходов для достижения целевого GRP
            total_outputs = AdvertisingCampaign.round_math(GRP / TVR_mean)
            
            # Количество выходов на канал в месяц
            outputs_per_channel = AdvertisingCampaign.round_math(total_outputs / len(self.channel_order))

            required_outputs_by_grp[GRP] = int(outputs_per_channel)

            if debug:
                print(f'Для целевого GRP = {GRP} количество выходов равно {int(total_outputs)}. ' + \
                    f'На каждом канале по {int(outputs_per_channel)} выхода в месяц.')

        GRP_and_points = pd.DataFrame(list(required_outputs_by_grp.items()), columns = ['Целевое GRP', 'Количество выходов'])

        return required_outputs_by_grp, GRP_and_points


    def make_new_GRP(self, time_table_dict: dict, data_result_TVR: pd.DataFrame):
        """
            Функция для расчета фактических GRP кампании, основываясь на рейтингах выбранных слотов в выбранные дни.
        """
        # Считаем новые GRP, основываясь на количестве выходов и TVR
        new_target_GRP = {}
        for number_of_outputs, df in time_table_dict.items():
            if number_of_outputs == 0:
                new_target_GRP[number_of_outputs] = 0
            else:
                df.rename(columns = {'Время': 'Время выхода'}, inplace = True)
                df_to_calculate_grp = pd.merge(df, data_result_TVR, on = ['Дата', 'Канал', 'Время выхода'], how = 'inner')
                GRP_new = df_to_calculate_grp['TVR'].sum()
                T = df_to_calculate_grp['Количество респондентов'].sum() * 60
                new_target_GRP[number_of_outputs] = [GRP_new, T]
        # Создаем новый DataFrame с новыми GRP
        new_GRP_df = pd.DataFrame(new_target_GRP)
        new_GRP_df.columns = ['Количество выходов', 'GRP new', 'Объём']
        #new_GRP_df = pd.DataFrame(list(new_target_GRP.items()), columns = ['Количество выходов', 'GRP new'])
        return new_GRP_df


    def calculate_reach(
        self,
        time_table_dict: dict, 
        minutely_df: pd.DataFrame, 
        new_GRP_df: pd.DataFrame, 
        GRP_and_points: pd.DataFrame, 
        respondents_weights_mean_df: pd.DataFrame
        ):
        """
            Функция для расчета накопленного Reach
        """
        # Расчет Reach рекламной кампании основываясь на выбранных слотах, днях и каналах
        reach_results = {}
        for target_grp, timetable_df in time_table_dict.items():
        
            if target_grp == 0:
                reach_results[target_grp] = 0
        
            else:
                time_table = timetable_df.copy()
                time_table.rename(columns = {'Время': 'Время выхода'}, inplace = True)
            
                per_reach = pd.merge(time_table, minutely_df, on = ['Дата', 'Канал', 'Время выхода'], how = 'inner')
                per_reach = per_reach[
                            [
                                'Дата', 'Канал', 'SubjectID', 'weight_new', 
                                'Media', 'Время выхода init', 'Время окончания', 'dur_min'
                            ]
                ]
                data_new = pd.merge(per_reach, respondents_weights_mean_df, on = ['SubjectID'], how = 'inner')
        
                unique_respondents = data_new.drop_duplicates(subset = ['SubjectID'], keep = 'first')
                reach_results[target_grp] = unique_respondents['weight_mean'].sum()

        reach_df = pd.DataFrame(list(reach_results.items()), columns = ['Количество выходов', 'Reach'])

        with_new_grp = pd.merge(GRP_and_points, new_GRP_df, on = ['Количество выходов'], how = 'inner')
        final = pd.merge(with_new_grp, reach_df, on = ['Количество выходов'], how = 'inner')
        return final


    def make_simulation(
        self,
        mean_weight_df: pd.DataFrame,
        debug = False,
        target_date: str = '01.03.2026'
        ):
        """
            Функция для расчета охвата рекламной кампании для выбранного медиа
        """
        # ШАГ 1. ОТБОР НЕОБХОДИМЫХ СЛОТОВ ДЛЯ АНАЛИЗА (tvr_results), РАЗБИВКА ВРЕМЕННОГО СТОЛБЦА ПО МИНУТАМ (minutely_df), 
        # А ТАКЖЕ РАСЧЕТ TVR (data_result_TVR)
        minutely_df, data_result_TVR, tvr_results = self.choose_slots()

        # ШАГ 2. РАСЧЕТ СРЕДНЕГО СУММАРНОГО РЕЙТИНГА
        TVR_mean_per_month = self.average_TVR_monthly(tvr_results)

        # ШАГ 3. РАСЧЕТ НЕОБХОДИМОГО КОЛИЧЕСТВА ВЫХОДОВ РЕКЛАМНОЙ КАМПАНИИ, ОСНОВЫВАЯСЬ НА ЦЕЛЕВОМ GRP
        required_outputs_by_grp, GRP_and_points = self.necessary_outputs_advertising_calculation(
                                                        TVR_mean_per_month, debug
                                                )

        # ШАГ 4. РАСПРЕДЕЛЕНИЕ РЕКЛАМНЫХ БЛОКОВ ПО ДНЯМ И СЛОТАМ. ПОЛУЧАЕМ РАСПИСАНИЕ (time_table_dict)
        break_points = BreakPointGeneration(target_date, self.channel_order)
        time_table_dict = break_points.generate_distribution_pipeline(required_outputs_by_grp.values())

        # ШАГ 5. РАСЧЕТ ФАКТИЧЕСКИХ GRP КАМПАНИИ ЧЕРЕЗ РЕЙТИНГИ СЛОТОВ, В КОТОРЫХ ОНИ РАЗМЕСТИЛИСЬ
        new_GRP_df = self.make_new_GRP(time_table_dict, data_result_TVR)

        print(new_GRP_df)

        # ШАГ 6. РАСЧЁТ НАКОПЛЕННОГО REACH КАК СУММА ВЕСОВ УНИКАЛЬНЫХ РЕСПОДЕНТОВ ЗА ВЕСЬ ПЕРИОД
        reach_df = self.calculate_reach(time_table_dict, minutely_df, new_GRP_df, GRP_and_points, mean_weight_df)
        return reach_df

