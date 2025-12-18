import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Callable
from datetime import timedelta, datetime, time
from dateutil.relativedelta import relativedelta
from difflib import SequenceMatcher
import traceback
traceback.print_exc()


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class TVShareCalculator:
    """
    Класс для расчета долей в конкретных слотах через вес слота и процент длительности программы в часе
    """

    def __init__(self, df):
        """
        Атрибуты класса
        Args:
            df: pd.DataFrame: Датафрейм с исходной долей
        """
        self.df = df
        self._validate_data()

    def _validate_data(self) -> None:
        """
        Проверка обязательных колонок в данных.
        """
        required_columns = ["Дата", "Время выхода", "Время окончания"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    @staticmethod
    def calculate_slot_weights(
        total_tv_audience: pd.DataFrame,
        rating_col: str = "TTVRtg000",
        timeslot_col: str = "TimeSlot",
        date_col: str = "Date",
    ) -> pd.DataFrame:
        """
        Рассчитывает веса слотов на основе данных о TotalTVAudience.

        Args:
            total_tv_audience: Датафрейм с аудиторными данными
            rating_col: Название колонки с рейтингом
            timeslot_col: Название колонки с временным слотом
            date_col: Название колонки с датой

        Returns:
            Датафрейм с весами слотов
        """
        df = total_tv_audience.copy()

        df["TimeSlot_dt"] = pd.to_datetime(df[timeslot_col])
        df["hour_start"] = df["TimeSlot_dt"].dt.hour

        # Группировка и расчет весов
        df["daily_total"] = df.groupby(date_col)[rating_col].transform("sum")
        df["Slot_weight"] = df[rating_col] / df["daily_total"]

        # Замена бесконечно малых значений
        df["Slot_weight"] = df["Slot_weight"].fillna(0)

        return df[
            [date_col, timeslot_col, rating_col, "Slot_weight", "hour_start"]
        ].reset_index(drop=True)

    @staticmethod
    def get_hour(dt, param: str):
        """
        Метод для генерации часа.
        Возможные опции: начало текущего часа, конец текущего часа, начало следующего часа, начало предыдущего часа
        """
        # Вариант 1: Начало текущего часа
        if param == "start_of_current_hour":
            return dt.replace(minute=0, second=0, microsecond=0)

        # Вариант 2: Конец текущего часа
        elif param == "end_of_current_hour":
            return dt.replace(minute=59, second=59, microsecond=0)

        # Вариант 3: Старт следующего часа
        elif param == "start_next_hour":
            return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        # Вариант 4: Старт прошлого часа
        elif param == "start_previous_hour":
            return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=-1)


    def round_time(self, time_column: str, minutes=1, method="round"):
        """
        Точное округление времени до минут без использования float.

        Args:

            time_series : pd.Series: Серия со временем в формате 'HH:MM:SS'
            minutes : int: Шаг округления в минутах (1, 5, 10, 15, 30, 60)
            method : str: Метод округления: 'round', 'floor', 'ceil'

        Returns:
            pd.Series: Округленное время
        """
        df = self.df.copy()
        time_series = df[time_column]

        def round_single_time(time_str, minutes_step, method_type):
            # Разбираем время
            if isinstance(time_str, str):
                h, m, s = map(int, time_str.split(":"))
            elif hasattr(time_str, "hour"):  # Если это datetime.time
                h, m, s = time_str.hour, time_str.minute, time_str.second
            else:
                return time_str

            # Если имеем начало часа, например, 05:00:00, то возвращаем в исходном виде
            if m == 0.0 and s == 0.0:
                return f"{h:02d}:00:00"

            # Если время кривое
            else:

                # Общее количество секунд
                total_seconds = h * 3600 + m * 60 + s
                step_seconds = minutes_step * 60

                if method_type == "floor":
                    # Округление вниз
                    rounded_seconds = (total_seconds // step_seconds) * step_seconds
                elif method_type == "ceil":
                    # Округление вверх
                    if total_seconds % step_seconds == 0:
                        rounded_seconds = total_seconds
                    else:
                        rounded_seconds = (
                            (total_seconds // step_seconds) + 1
                        ) * step_seconds
                else:  # 'round' - стандартное округление
                    # Количество секунд от начала интервала
                    remainder = total_seconds % step_seconds

                    # Если остаток >= половины интервала, округляем вверх
                    if remainder >= step_seconds / 2:
                        rounded_seconds = (
                            (total_seconds // step_seconds) + 1
                        ) * step_seconds
                    else:
                        rounded_seconds = (total_seconds // step_seconds) * step_seconds

                # Преобразуем обратно
                new_h = (rounded_seconds // 3600) % 24
                new_m = (rounded_seconds % 3600) // 60

                return f"{new_h:02d}:{new_m:02d}:00"

        # Применяем к каждой строке
        return time_series.apply(lambda x: round_single_time(x, minutes, method))


    def adjust_hour_start(self, end_col="Время окончания", start_col="Время выхода"):
        """
        Заменяет XX:00:00 на XX:59:59, только если это действительно
        начало нового часа в расписании (т.е. следующее время выхода начинается с этого часа).
        """
        df_adj = self.df.copy()

        for i in range(len(df_adj) - 1):
            current_end = df_adj.loc[i, end_col]
            next_start = df_adj.loc[i + 1, start_col]

            # Если текущее окончание - начало часа И следующее время выхода начинается с этого же часа
            if current_end.endswith(":00:00") and next_start.startswith(
                current_end[:2]
            ):
                hours = int(current_end.split(":")[0])

                if hours == 0:
                    new_time = "23:59:59"
                else:
                    new_hour = hours - 1
                    new_time = f"{new_hour:02d}:59:59"

                df_adj.loc[i, end_col] = new_time

        # Обрабатываем последнюю строку отдельно
        if df_adj.loc[df_adj.index[-1], end_col].endswith(":00:00"):
            hours = int(df_adj.loc[df_adj.index[-1], end_col].split(":")[0])
            if hours == 0:
                new_time = "23:59:59"
            else:
                new_hour = hours - 1
                new_time = f"{new_hour:02d}:59:59"
            df_adj.loc[df_adj.index[-1], end_col] = new_time

        return df_adj


    def calculate_hour_jump(self, start_col="Время выхода", end_col="Время окончания"):
        """
        Рассчитывает количество скачков через час для всех программ в DataFrame.

        Args:
            df: DataFrame с колонками времени
            start_col: название колонки с временем начала
            end_col: название колонки с временем окончания

        Returns:
            DataFrame с добавленной колонкой 'Скачки через час'
        """
        result_df = self.df.copy()

        jumps_list = []
        hours_list = []
        durations_list = []

        for idx, row in result_df.iterrows():
            try:
                # Парсим время
                start_time = datetime.strptime(str(row[start_col]), "%H:%M:%S")
                end_time = datetime.strptime(str(row[end_col]), "%H:%M:%S")

                # Корректируем время окончания при переходе через полночь
                if end_time <= start_time:
                    end_time += timedelta(days=1)

                # Рассчитываем количество скачков
                current_time = start_time
                hour_jumps = 0
                hours_covered = []

                while current_time < end_time:
                    current_hour = current_time.hour
                    hours_covered.append(current_hour)

                    # Определяем начало следующего часа
                    next_hour_start = current_time.replace(
                        minute=0, second=0, microsecond=0
                    ) + timedelta(hours=1)

                    # Если следующий час не превышает время окончания, это скачок
                    if next_hour_start < end_time:
                        hour_jumps += 1

                    # Переходим к следующему часу
                    current_time = next_hour_start

                # Длительность в минутах
                duration = (end_time - start_time).total_seconds() / 60.0

                jumps_list.append(hour_jumps)
                hours_list.append(hours_covered)
                durations_list.append(duration)

            except Exception as e:
                print(f"Ошибка в строке {idx}: {e}")
                jumps_list.append(0)
                hours_list.append([])
                durations_list.append(0)

        # Добавляем результаты в DataFrame
        result_df["Скачки через час"] = jumps_list
        result_df["Пройдено часов"] = hours_list
        result_df["Длительность (мин)"] = durations_list

        return result_df


    def calculate_weighted_share(self, auedience) -> pd.DataFrame:
        """
        Функция для расчета взвешенной доли для какого-то конкретного дня.
        Args:
            auedience: pd.DataFrame: ДатаФрейм с весами слотов, посчитанными через TotalTVAuedience для конкретного дня.
        Returns:
            res: pd.DataFrame: Датафрейм с новой рассчитанной долей для какого-то конкретного дня
            share_sum: суммарная доля для какого-то конкретного дня
        """
        # Определяем количество скачков через час в датафрейме
        result_df = self.calculate_hour_jump()

        df = result_df.copy()

        # Создаём столбец с новой долей
        df["Share_weighted"] = 0.0

        # Создаем словарь весов для быстрого доступа
        weight_dict = dict(zip(auedience["hour_start"], auedience["Slot_weight"]))

        for i in range(len(df)):
            num_of_jumps = df.iloc[i]["Пройдено часов"]
            program_start = df.iloc[i]["Время выхода"]
            program_finish = df.iloc[i]["Время окончания"]
            share = df.iloc[i]["Share"]

            start_dt = datetime.strptime(program_start, "%H:%M:%S")
            end_dt = datetime.strptime(program_finish, "%H:%M:%S")

            # Обрабатываем переход через полночь
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            coeffs = []

            # Если скачка нет (программа в пределах одного часа)
            if len(num_of_jumps) == 1:

                # Определение часа старта для подбора веса слота
                hour = num_of_jumps[0]

                # Длительность в минутах
                duration_minutes = (end_dt - start_dt).total_seconds() / 60.0
                # % длительности программы в часе
                percent_duration = duration_minutes / 60.0

                # Получаем вес слота
                slot_weight = weight_dict.get(hour, 1.0)

                # Если скачка через час нет, считаем долю в слоте как Share * вес слота * % длительности программы в часе
                coeffs.append(percent_duration * slot_weight)

            # Если есть скачки (он необязательно должен быть 1)
            else:
                for k in range(len(num_of_jumps)):
                    # Первый скачок
                    if k == 0:
                        # Определяем конец первого часа
                        end_hour = datetime.strptime(
                            f"{num_of_jumps[k]:02d}:59:59", "%H:%M:%S"
                        )
                        # Длительность в минутах
                        duration_minutes = (end_hour - start_dt).total_seconds() / 60.0
                        # % длительности программы в часе
                        percent_duration = duration_minutes / 60.0

                        # Получаем вес слота
                        slot_weight = weight_dict.get(num_of_jumps[0], 1.0)

                        coeffs.append(percent_duration * slot_weight)

                    # Последний скачок
                    elif k == len(num_of_jumps) - 1:
                        # Определяем начало последнего часа
                        start_hour = datetime.strptime(
                            f"{num_of_jumps[k]:02d}:00:00", "%H:%M:%S"
                        )

                        # Если start_hour меньше start_dt (переход через полночь), добавляем день
                        if start_hour < start_dt:
                            start_hour += timedelta(days=1)

                        # Длительность в минутах
                        duration_minutes = (end_dt - start_hour).total_seconds() / 60.0
                        # % длительности программы в часе
                        percent_duration = duration_minutes / 60.0

                        # Получаем вес слота
                        slot_weight = weight_dict.get(num_of_jumps[-1], 1.0)

                        coeffs.append(percent_duration * slot_weight)

                    # Промежуточный скачок
                    else:
                        # Определяем начало часа
                        start_hour = datetime.strptime(
                            f"{num_of_jumps[k]:02d}:00:00", "%H:%M:%S"
                        )

                        # Определяем конец часа
                        end_hour = datetime.strptime(
                            f"{num_of_jumps[k]:02d}:59:59", "%H:%M:%S"
                        )
                        # Длительность в минутах
                        duration_minutes = (
                            end_hour - start_hour
                        ).total_seconds() / 60.0
                        # % длительности программы в часе
                        percent_duration = duration_minutes / 60.0

                        # Получаем вес слота
                        slot_weight = weight_dict.get(num_of_jumps[k], 1.0)

                        coeffs.append(percent_duration * slot_weight)

            if len(coeffs) != 0:
                coefficient = np.sum(coeffs)
                df.at[i, "Share_weighted"] = share * coefficient

        res = df[
            [
                "Дата",
                "Название программы",
                "Время выхода",
                "Время окончания",
                "Share",
                "Share_weighted",
            ]
        ]
        # res.rename(columns = {'Share_NEW': 'Share'}, inplace = True)
        # Расчёт суммарной доли по дню
        share_sum = np.sum(list(res["Share_weighted"]))
        return res, share_sum


class TVScheduleProcessor:
    """
    Класс для подгона сетки Palomars под сетку VIMB из Сводного отчёта
    """

    def __init__(self):
        """
        Atributes:
            vimb_init: pd.DataFrame: исходная сетка ТВ-программ VIMB
            palomars_init: pd.DataFrame: исходная сетка ТВ-программ Mediascope
        """
        self.vimb_init = None
        self.palomars_init = None
        self.stats = {}
    

    @staticmethod
    def calculate_duration_minutes(row):
            try:
                start_time = datetime.strptime(row["Время выхода"], "%H:%M:%S")
                end_time = datetime.strptime(row["Время окончания"], "%H:%M:%S")

                # Если время окончания меньше времени начала (переход через полночь)
                if end_time < start_time:
                    end_time = end_time.replace(day=end_time.day + 1)

                duration_minutes = (
                    end_time - start_time
                ).total_seconds() / 60  # в минутах
                return duration_minutes
            except:
                return 0
            

    def _adjust_end_time(
        self, df: pd.DataFrame, time_col: str = "Время окончания"
    ) -> pd.DataFrame:
        """
        Корректировка времени окончания для обработки границ часов. Если время окончания, например, 05:00:00, то будет сделана замена на 04:59:59.
        Отдельно обрабатывается перескок через полночь.

        Args:
            df: датафрейм, в котором хотим произвести конвертацию времени.
            time_col: str: название колонки, в которой хотим сделать конвертацию. По умолчанию 'Время окончания'.

        Returns:
            Датафрейм df с конвертированными слотами Времени окончания программ.
        """

        def adjust_time(time_str: str) -> str:
            h, m, s = map(int, time_str.split(":"))

            if m == 0 and s == 0:

                # Если полночь
                if h == 0:
                    return "23:59:59"

                return f"{h-1:02d}:59:59"

            return time_str

        df = df.copy()
        df[time_col] = df[time_col].apply(adjust_time)
        return df
    

    @staticmethod
    def split_programs_by_time(
            df, 
            start_minutes_allowed = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],  # разрешенные минуты для начала
            end_minutes_allowed = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],  # разрешенные минуты для окончания
            max_duration_minutes = 60
        ):
        """
            Универсальное разделение по времени начала/окончания.
            
            Параметры:
            - start_minutes_allowed: список разрешенных минут для времени начала
            - end_minutes_allowed: список разрешенных минут для времени окончания  
            - max_duration_minutes: максимальная длительность в минутах
        """
        def parse_time(time_str):
            """
                Парсит время в (часы, минуты, секунды).
            """
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 2:
                return parts[0], parts[1], 0
            return parts[0], parts[1], parts[2]
        
        def check_condition(row):
            h1, m1, s1 = parse_time(row['Время выхода'])
            h2, m2, s2 = parse_time(row['Время окончания'])
            
            # Проверка секунд (должны быть 0)
            if s1 != 0 or s2 != 0:
                return False
            
            # Проверка минут начала
            if m1 not in start_minutes_allowed:
                return False
            
            # Проверка минут окончания
            if m2 not in end_minutes_allowed:
                return False
            
            # Расчет длительности
            start_total = h1 * 60 + m1
            end_total = h2 * 60 + m2
            
            if end_total < start_total:
                end_total += 24 * 60
                
            duration = end_total - start_total
            
            return duration <= max_duration_minutes
        
        mask = df.apply(check_condition, axis=1)
        
        return df[mask].reset_index(drop = True), df[~mask].reset_index(drop = True)
    

    def _add_original_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление колонок с исходным временем, чтобы можно было сопоставить с исходным датафреймом VIMB.
        Args:
            df: датафрейм, в котором хотим добавить всопомгательные столбцы
        Returns:
            pd.DataFrame: датафрейм df с двумя дополнительными колонками с исходными временами слотов
        """
        data = df.copy()
    
        if "Время выхода_исходное" not in data.columns:
            data["Время выхода_исходное"] = data["Время выхода"]
        
        if "Время окончания_исходное" not in data.columns:
            data["Время окончания_исходное"] = data["Время окончания"]
        
        return data
    

    def _remove_found_programs(
            self, source_df: pd.DataFrame, found_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
            Удаление найденных программ из исходных данных

            Args:
                source_df: датафрейм, из которого будем удалять найденные программы
                found_df: датафрейм, с найденными программами

            Returns:
                pd.DataFrame: очищенный датафрейм source_df от найденных программ в датафрейме found_df

        """
        if found_df.empty:
            return source_df

        merge_keys = ["Дата", "Название программы", "Время выхода", "Время окончания"]
        merged = source_df.merge(
            found_df[merge_keys], on = merge_keys, how = "left", indicator = True
        )

        return merged.query('_merge == "left_only"').drop("_merge", axis = 1)
    

    @staticmethod
    def group_broadcasts_by_hours(data):
        """
            Объединяет последовательные трансляции одной программы по часам.
            Объединяет сегменты, которые находятся в одном часу и имеют последовательное время.
            ИСПРАВЛЕНО: Теперь находит все сегменты программы в пределах часа, даже если они прерываются другими программами.

            Args:
                data (pd.DataFrame): Исходный DataFrame с временными сегментами

            Returns:
                pd.DataFrame: DataFrame с консолидированными сегментами по часам
        """
        df = data.copy()

        df["Время выхода"] = pd.to_datetime(df["Время выхода"])
        df["Время окончания"] = pd.to_datetime(df["Время окончания"])

        # Создаем колонку с часом начала для группировки
        df["Час_начала"] = df["Время выхода"].dt.floor("H")

        # Сортируем данные по дате, времени для корректной обработки
        df = df.sort_values(["Дата", "Время выхода"]).reset_index(drop=True)

        rows = []

        # Обрабатываем каждую программу отдельно
        for name in df["Название программы"].unique():
            program_df = df[df["Название программы"] == name].copy()

            # Группируем по дате и часу
            for (date, hour), hour_group in program_df.groupby(["Дата", "Час_начала"]):
                hour_group = hour_group.sort_values("Время выхода").reset_index(
                    drop = True
                )

                # Находим все сегменты этой программы в данном часу
                segments = []
                for _, row in hour_group.iterrows():
                    segments.append(
                        {
                            "start": row["Время выхода"],
                            "end": row["Время окончания"],
                            "share": row["Share"],
                        }
                    )

                if segments:
                    # Сортируем сегменты по времени начала
                    segments.sort(key=lambda x: x["start"])

                    # Объединяем смежные сегменты
                    merged_segments = []
                    current_start = segments[0]["start"]
                    current_end = segments[0]["end"]
                    current_shares = [segments[0]["share"]]

                    for i in range(1, len(segments)):
                        # Если текущий сегмент начинается сразу после окончания предыдущего
                        if segments[i]["start"] == current_end:
                            current_end = segments[i]["end"]
                            current_shares.append(segments[i]["share"])
                        else:
                            # Сохраняем текущую группу и начинаем новую
                            merged_segments.append(
                                {
                                    "start": current_start,
                                    "end": current_end,
                                    "shares": current_shares.copy(),
                                }
                            )
                            current_start = segments[i]["start"]
                            current_end = segments[i]["end"]
                            current_shares = [segments[i]["share"]]

                    # Добавляем последнюю группу
                    merged_segments.append(
                        {
                            "start": current_start,
                            "end": current_end,
                            "shares": current_shares.copy(),
                        }
                    )

                    # Создаем записи для каждой объединенной группы
                    for segment in merged_segments:
                        rows.append(
                            {
                                "Дата": date,
                                "Название программы": name,
                                "Время выхода": segment["start"],
                                "Время окончания": segment["end"],
                                "Share": sum(segment["shares"]),
                                "Количество_отрезков": len(segment["shares"]),
                                "Час_группы": hour.time(),
                            }
                        )

        # Создаем DataFrame
        result_df = pd.DataFrame(rows)

        if len(result_df) > 0:
            # Сортируем результаты
            result_df = result_df.sort_values(
                ["Дата", "Название программы", "Время выхода"]
            ).reset_index(drop=True)

            result_df["Время выхода"] = result_df["Время выхода"].dt.strftime(
                "%H:%M:%S"
            )
            result_df["Время окончания"] = result_df["Время окончания"].dt.strftime(
                "%H:%M:%S"
            )

            # Возвращаем только нужные колонки
            return result_df[
                [
                    "Дата",
                    "Название программы",
                    "Время выхода",
                    "Время окончания",
                    "Share",
                ]
            ]
        else:
            # Если нет данных, возвращаем пустой DataFrame с правильной структурой
            return pd.DataFrame(
                columns = [
                    "Дата",
                    "Название программы",
                    "Время выхода",
                    "Время окончания",
                    "Share"
                ]
            )
        

    def join_broadcasts(self, data, include_share: bool = True):
        """
            Упрощенная версия для объединения трансляций в рамках одного дня.
            Учитывает эфирные сутки (05:00-04:59).
            Объединяет смежные сегменты одной программы.
        """
        if data.empty:
            columns = ["Дата", "Название программы", "Время выхода", "Время окончания"]
            if include_share:
                columns.append("Share")
            return pd.DataFrame(columns=columns)
        
        df = data.copy()
        
        df = self._adjust_end_time(df)
        
        # Функция для сортировки по эфирным суткам
        def broadcast_time_key(time_str):
            h, m, s = map(int, time_str.split(':'))
            return (0 if h >= 5 else 1, h, m, s)
        
        # Преобразование времени в минуты для удобного сравнения
        def time_to_minutes(time_str):
            """Преобразует время в формате HH:MM:SS в минуты с начала эфирных суток"""
            h, m, s = map(int, time_str.split(':'))
            # Для времени до 05:00 добавляем 24 часа
            total_minutes = h * 60 + m + s / 60
            if h < 5:  # Время с 00:00 до 04:59
                total_minutes += 24 * 60  # Добавляем сутки
            return total_minutes
        
        # Подготовка данных
        df["sort_key"] = df["Время выхода"].apply(broadcast_time_key)
        df = df.sort_values(["Дата", "sort_key"])
        df = df.drop("sort_key", axis=1)
        
        # Добавляем колонку с временем в минутах для сравнения
        df["start_minutes"] = df["Время выхода"].apply(time_to_minutes)
        df["end_minutes"] = df["Время окончания"].apply(time_to_minutes)
        
        results = []
        
        # Обработка каждой программы
        for program_name in df["Название программы"].unique():
            program_mask = df["Название программы"] == program_name
            program_data = df[program_mask].copy()
            
            if program_data.empty:
                continue
            
            # Сортировка программы по времени (уже отсортирована)
            program_data = program_data.sort_values("start_minutes")
            
            # Объединение сегментов
            current_group = {
                "Дата": program_data.iloc[0]["Дата"],
                "Название программы": program_name,
                "Время выхода": program_data.iloc[0]["Время выхода"],
                "Время окончания": program_data.iloc[0]["Время окончания"],
                "start_minutes": program_data.iloc[0]["start_minutes"],
                "end_minutes": program_data.iloc[0]["end_minutes"],
            }
            
            if include_share:
                current_group["shares"] = [program_data.iloc[0]["Share"]]
            
            # Обработка остальных записей программы
            for i in range(1, len(program_data)):
                current_row = program_data.iloc[i]
                next_start_minutes = current_row["start_minutes"]
                next_end_minutes = current_row["end_minutes"]
                
                # Проверка на смежность сегментов с учетом разницы в 1 минуту
                time_gap = next_start_minutes - current_group["end_minutes"]
                
                # Ключевое изменение: Не объединяем через границу эфирных суток
                # (кроме специального случая 04:59:59 → 05:00:00)
                prev_end_time = current_group["Время окончания"]
                next_start_time = current_row["Время выхода"]
                
                # Проверяем, не пересекаем ли мы границу эфирных суток
                # (следующий сегмент начинается в новых эфирных сутках, а текущий заканчивается в старых)
                crosses_broadcast_day = (
                    prev_end_time >= "00:00:00" and prev_end_time <= "04:59:59" and
                    next_start_time >= "05:00:00"
                )
                
                # Условия объединения:
                # 1. Нет разрыва (время окончания = время начала следующей) И не пересекаем границу
                # 2. Разрыв в пределах 1 минуты И не пересекаем границу
                # 3. Специальный случай: 04:59:59 → 05:00:00 (это допускается)
                is_adjacent = (
                    (time_gap == 0 and not crosses_broadcast_day) or  # Нет разрыва и не пересекаем границу
                    (0 < time_gap <= 1 and not crosses_broadcast_day) or  # Разрыв не более 1 минуты и не пересекаем границу
                    (prev_end_time == "04:59:59" and next_start_time == "05:00:00")  # Допустимый переход через границу
                )
                
                if is_adjacent:
                    # Объединяем с текущей группой
                    current_group["Время окончания"] = current_row["Время окончания"]
                    current_group["end_minutes"] = next_end_minutes
                    if include_share:
                        current_group["shares"].append(current_row["Share"])
                else:
                    # Сохраняем текущую группу и начинаем новую
                    result_entry = {
                        "Дата": current_group["Дата"],
                        "Название программы": current_group["Название программы"],
                        "Время выхода": current_group["Время выхода"],
                        "Время окончания": current_group["Время окончания"],
                    }
                    
                    if include_share:
                        result_entry["Share"] = sum(current_group["shares"])
                        result_entry["Количество_сегментов"] = len(current_group["shares"])
                    
                    results.append(result_entry)
                    
                    # Новая группа
                    current_group = {
                        "Дата": current_row["Дата"],
                        "Название программы": program_name,
                        "Время выхода": current_row["Время выхода"],
                        "Время окончания": current_row["Время окончания"],
                        "start_minutes": next_start_minutes,
                        "end_minutes": next_end_minutes,
                    }
                    
                    if include_share:
                        current_group["shares"] = [current_row["Share"]]
            
            # Сохраняем последнюю группу
            result_entry = {
                "Дата": current_group["Дата"],
                "Название программы": current_group["Название программы"],
                "Время выхода": current_group["Время выхода"],
                "Время окончания": current_group["Время окончания"],
            }
            
            if include_share:
                result_entry["Share"] = sum(current_group["shares"])
                result_entry["Количество_сегментов"] = len(current_group["shares"])
            
            results.append(result_entry)
        
        # Формирование итогового DataFrame
        if not results:
            columns = ["Дата", "Название программы", "Время выхода", "Время окончания"]
            if include_share:
                columns.append("Share")
            return pd.DataFrame(columns=columns)
        
        result_df = pd.DataFrame(results)
        
        # Сортировка результатов
        result_df["sort_key"] = result_df["Время выхода"].apply(broadcast_time_key)
        result_df = result_df.sort_values("sort_key").drop("sort_key", axis=1)
        
        # Выбор нужных колонок
        columns = ["Дата", "Название программы", "Время выхода", "Время окончания"]
        if include_share:
            columns.append("Share")
        
        return result_df[columns].reset_index(drop = True)



    @staticmethod
    def __get_overlap_duration_seconds(
            start1: datetime, end1: datetime, start2: datetime, end2: datetime
        ) -> float | None:
        overlap_start: datetime = datetime.combine(
            datetime.today(), max(start1, start2)
        )
        overlap_end: datetime = datetime.combine(datetime.today(), min(end1, end2))

        if overlap_start < overlap_end:
            return (overlap_end - overlap_start).total_seconds()
        else:
            return None


    @staticmethod
    def find_time_overlaps(
            program1_start: str,
            program1_end: str,
            program2_start: str,
            program2_end: str,
            min_overlap_ratio: float = 0.8,
        ) -> bool:
        date_format = "%H:%M:%S"

        s1: datetime = datetime.strptime(program1_start, date_format).time()
        f1: datetime = datetime.strptime(program1_end, date_format).time()
        s2: datetime = datetime.strptime(program2_start, date_format).time()
        f2: datetime = datetime.strptime(program2_end, date_format).time()

        s1d: datetime = datetime.combine(datetime.today(), s1)
        f1d: datetime = datetime.combine(datetime.today(), f1)
        s2d: datetime = datetime.combine(datetime.today(), s2)
        f2d: datetime = datetime.combine(datetime.today(), f2)

        overlap: float | None = TVScheduleProcessor.__get_overlap_duration_seconds(
            s1, f1, s2, f2
        )

        if not overlap:
            return False

        return (
            overlap / (min((f1d - s1d).total_seconds(), (f2d - s2d).total_seconds()))
            >= min_overlap_ratio
        )
    

    def _prepare_dataframe(
            self, df: pd.DataFrame, minutes: int, include_share: bool = True, preserve_original: bool = True
        ) -> pd.DataFrame:
        """
            Подготовка датафрейма. Времена слотов округляются до установленных минут.
            Таким образом, датафрейм подготавливается для дальнейшего анализа.
        """
        data = df.copy()

        # Сохраняем исходные значения времени, если требуется
        if preserve_original:
            if "Время выхода_исходное" not in data.columns:
                data["Время выхода_исходное"] = data["Время выхода"]
            
            if "Время окончания_исходное" not in data.columns:
                data["Время окончания_исходное"] = data["Время окончания"]

        # Округление времени
        calculator = TVShareCalculator(data)
        data["Время выхода"] = calculator.round_time("Время выхода", minutes)
        data["Время окончания"] = calculator.round_time("Время окончания", minutes)

        # Выбор колонок
        columns = [
            "Дата",
            "Название программы",
            "Время выхода",
            "Время окончания",
        ]

        # Добавляем исходные времена, если они есть
        if preserve_original:
            columns.extend(["Время выхода_исходное", "Время окончания_исходное"])

        if include_share and "Share" in data.columns:
            columns.insert(4, "Share")

        result = data[columns].copy()
        result["Дата"] = pd.to_datetime(result["Дата"]).dt.strftime("%Y-%m-%d")

        return result


    def _process_schedule_step(
            self,
            palomars: pd.DataFrame,
            vimb: pd.DataFrame,
            minutes: int
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
            Обработка одного шага сопоставления с округлением времени
        """

        try:
            # Сохраняем оригинальный Pal для возврата индексов
            pal_original = palomars.copy().reset_index(drop = True)
            pal_original['__pal_original_index'] = pal_original.index
            
            # Добавляем исходные колонки к VIMB если их нет
            vimb = self._add_original_time_columns(vimb)
            
            # Добавляем индексы
            vimb = vimb.reset_index(drop = True)
            vimb['__vimb_index'] = vimb.index

            # Подготовка данных
            pal_processed = self._prepare_dataframe(palomars, minutes, include_share = True)
            
            # Добавляем индекс Pal к обработанным данным
            pal_processed = pal_processed.reset_index(drop = True)
            pal_processed['__pal_processed_index'] = pal_processed.index

            # Корректировка времени окончания
            pal_processed = self._adjust_end_time(pal_processed, "Время окончания")
            vimb_processed = self._adjust_end_time(vimb, "Время окончания")

            # Конвертируем даты для сравнения
            vimb_processed = self.convert_data_column(vimb_processed)
            pal_processed = self.convert_data_column(pal_processed)

            # Поиск совпадений
            merge_keys = ["Дата", "Название программы", "Время выхода", "Время окончания"]
            
            matches = pd.merge(
                vimb_processed, 
                pal_processed, 
                on = merge_keys, 
                how = "inner"
            )
            
            if not matches.empty:
                # Добавляем информацию об индексах
                matches = matches[[
                    "Дата", "Название программы", 
                    "Время выхода", "Время окончания", 
                    "Share", "__vimb_index", "__pal_processed_index"
                ]]
                
                # Находим оригинальные индексы Pal
                # Создаем mapping между processed и original индексами
                pal_index_mapping = dict(zip(
                    pal_processed['__pal_processed_index'],
                    range(len(pal_processed))
                ))
                
                matches['__pal_original_index'] = matches['__pal_processed_index'].map(
                    lambda x: pal_index_mapping.get(x, -1)
                )

            # Удаление найденных
            vimb_remaining = self._remove_found_programs(
                vimb_processed.drop(columns = ['__vimb_index']), 
                matches.drop(columns = ['__vimb_index', '__pal_processed_index', '__pal_original_index'], errors='ignore')
            ).reset_index(drop = True)
            
            pal_remaining = self._remove_found_programs(
                pal_processed.drop(columns = ['__pal_processed_index']), 
                matches.drop(columns = ['__vimb_index', '__pal_processed_index', '__pal_original_index'], errors='ignore')
            ).reset_index(drop = True)

            return matches, vimb_remaining, pal_remaining, pal_original
            
        except Exception as e:
            print(f"ERROR в _process_schedule_step: {str(e)}")
            return pd.DataFrame(), vimb, palomars, palomars


    def _extract_core_info(
            self, df: pd.DataFrame, include_share: bool = True
        ) -> pd.DataFrame:
        """
            Извлечение основной информации с переименованием колонок
        """
        # Проверяем наличие колонок с исходным временем
        has_original_start = "Время выхода_исходное" in df.columns
        has_original_end = "Время окончания_исходное" in df.columns
        
        if has_original_start and has_original_end:
            if include_share and "Share" in df.columns:
                columns = [
                    'Дата',
                    'Название программы', 
                    'Время выхода_исходное',
                    'Время окончания_исходное', 
                    'Share'
                ]
            else:
                columns = [
                    'Дата', 
                    'Название программы',
                    'Время выхода_исходное', 
                    'Время окончания_исходное',
                ]

            result = df[columns].copy()
            result.rename(
                columns={
                    'Время выхода_исходное': 'Время выхода',
                    'Время окончания_исходное': 'Время окончания'
                },
                inplace=True
            )

            return result
        else:
            # Если исходных колонок нет, возвращаем обычные
            columns = [
                'Дата', 
                'Название программы',
                'Время выхода', 
                'Время окончания',
            ]
            
            if include_share and "Share" in df.columns:
                columns.insert(3, 'Share')
            
            return df[columns].copy()
    

    def convert_data_column(self, df):
        """
            Всопомгательный метод для конвертации столбца с названием 'Дата'
        """
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Дата'] = df["Дата"].dt.strftime("%Y-%m-%d")
        return df
        

    def broadcasts_overlaping_OLD(self, df, vimb):
        """
            Метод для сопоставления программ методом перекрытия двух длительностей.
        """
        vimb_ = self._adjust_end_time(vimb, 'Время окончания')

        df['overlap'] = ''
        df['Время выхода VIMB'] = ''
        df['Время окончания VIMB'] = ''

        # Преобразуем время в timedelta для более точного сравнения
        def time_to_minutes(time_str):
            h, m, s = map(int, time_str.split(':'))
            return h * 60 + m
        
        # Через расчет процента перекрытия понимаем, нужная это программа или нет
        for i in range(len(df)):
            name = df.iloc[i]['Название программы']
            start_palomars = df.iloc[i]['Время выхода']
            end_palomars = df.iloc[i]['Время окончания']
            
            # Конвертируем в минуты
            start_p_minutes = time_to_minutes(start_palomars)
            end_p_minutes = time_to_minutes(end_palomars)
            
            in_vimb = vimb_[vimb_['Название программы'] == name].reset_index(drop=True)

            if len(in_vimb) != 0:
                for j in range(len(in_vimb)):
                    start_vimb = in_vimb.iloc[j]['Время выхода']
                    end_vimb = in_vimb.iloc[j]['Время окончания']
                    
                    # Конвертируем в минуты
                    start_v_minutes = time_to_minutes(start_vimb)
                    end_v_minutes = time_to_minutes(end_vimb)
                    
                    # Более гибкое условие: проверяем перекрытие по времени
                    # Вместо жесткой проверки часов, проверяем фактическое перекрытие
                    
                    # 1. Проверяем, что программы перекрываются хотя бы на 15 минут
                    overlap_start = max(start_p_minutes, start_v_minutes)
                    overlap_end = min(end_p_minutes, end_v_minutes)
                    overlap_duration = overlap_end - overlap_start
                    
                    # 2. Проверяем перекрытие через find_time_overlaps
                    has_overlap = TVScheduleProcessor.find_time_overlaps(
                        start_palomars, end_palomars, start_vimb, end_vimb
                    )
                    
                    # Условие: либо перекрытие > 15 минут, либо find_time_overlaps возвращает True
                    if overlap_duration > 15 or has_overlap:
                        df.at[i, "overlap"] = True
                        df.at[i, "Время выхода VIMB"] = start_vimb
                        df.at[i, "Время окончания VIMB"] = end_vimb
                        break  # Нашли совпадение, выходим из внутреннего цикла
                    else:
                        continue
            else:
                continue

        with_overlap = df[df["overlap"] == True].reset_index(drop=True)
        
        if not with_overlap.empty:
            res_overlap = with_overlap[
                [
                    "Дата",
                    "Название программы",
                    "Время выхода VIMB",
                    "Время окончания VIMB",
                    "Share",
                ]
            ]
            res_overlap = res_overlap.rename(
                columns={
                    "Время выхода VIMB": "Время выхода",
                    "Время окончания VIMB": "Время окончания",
                }
            )
            return res_overlap
        else:
            return pd.DataFrame()


    def broadcasts_overlaping(
            self, df, vimb, check_both_sides: bool = True, only_hour_programs: bool = False
        ):
        """
            Метод для сопоставления программ методом перекрытия двух длительностей.
        """
        vimb_ = self._adjust_end_time(vimb, "Время окончания")

        df = df.copy()
        df["overlap"] = False
        df["Время выхода VIMB"] = ""
        df["Время окончания VIMB"] = ""
        df["overlap_ratio"] = 0.0
        df["time_start_diff"] = 999999
        df["exact_match_score"] = 0  # Новый показатель точности совпадения

        date_format = "%H:%M:%S"

        for i in range(len(df)):
            name = df.iloc[i]["Название программы"]
            start_p_str = df.iloc[i]["Время выхода"]
            end_p_str = df.iloc[i]["Время окончания"]
            date_val = df.iloc[i]["Дата"]

            if (
                pd.isna(name)
                or pd.isna(start_p_str)
                or pd.isna(end_p_str)
                or pd.isna(date_val)
            ):
                continue

            s1 = datetime.strptime(start_p_str, date_format).time()
            f1 = datetime.strptime(end_p_str, date_format).time()
            s1d = datetime.combine(datetime.today(), s1)
            f1d = datetime.combine(datetime.today(), f1)

            if f1d < s1d:
                f1d = f1d.replace(day=f1d.day + 1)

            p_duration = (f1d - s1d).total_seconds()

            in_vimb = vimb_[vimb_["Название программы"] == name].reset_index(drop=True)

            if len(in_vimb) == 0:
                continue

            best_match_info = None
            best_exact_match_score = -1  # Ищем максимальный score точности

            for j in range(len(in_vimb)):
                start_v_str = in_vimb.iloc[j]["Время выхода"]
                end_v_str = in_vimb.iloc[j]["Время окончания"]
                v_date = in_vimb.iloc[j]["Дата"]

                if v_date != date_val:
                    continue

                s2 = datetime.strptime(start_v_str, date_format).time()
                f2 = datetime.strptime(end_v_str, date_format).time()
                s2d = datetime.combine(datetime.today(), s2)
                f2d = datetime.combine(datetime.today(), f2)

                if f2d < s2d:
                    f2d = f2d.replace(day=f2d.day + 1)

                v_duration = (f2d - s2d).total_seconds()

                # НОВАЯ ЛОГИКА: если only_hour_programs=True, проверяем длительность
                if only_hour_programs:
                    # Проверяем длительность программы в VIMB
                    v_duration_minutes = v_duration / 60
                    if not (50 <= v_duration_minutes <= 60):
                        continue  # Пропускаем программы не в диапазоне 50-60 минут

                    # Также проверяем длительность программы в df (опционально)
                    p_duration_minutes = p_duration / 60
                    if not (50 <= p_duration_minutes <= 60):
                        continue  # Пропускаем программы не в диапазоне 50-60 минут

                # Старая проверка длительности
                if not only_hour_programs and p_duration <= 3600 and v_duration > 3600:
                    continue

                overlap_seconds = TVScheduleProcessor.__get_overlap_duration_seconds(
                    s1, f1, s2, f2
                )

                if overlap_seconds:
                    # Проверяем перекрытие с обеих сторон
                    overlap_ratio_p = (
                        overlap_seconds / p_duration if p_duration > 0 else 0
                    )
                    overlap_ratio_v = (
                        overlap_seconds / v_duration if v_duration > 0 else 0
                    )

                    if check_both_sides:
                        if overlap_ratio_p >= 0.8 and overlap_ratio_v >= 0.8:
                            # Вычисляем точность совпадения:
                            # 1. Совпадение времени начала (чем ближе, тем лучше)
                            time_start_diff = abs((s1d - s2d).total_seconds())
                            # 2. Совпадение времени окончания (чем ближе, тем лучше)
                            time_end_diff = abs((f1d - f2d).total_seconds())
                            # 3. Общее перекрытие (чем больше, тем лучше)

                            # Считаем общий score точности:
                            # Базовый score = сумма overlap_ratio
                            base_score = overlap_ratio_p + overlap_ratio_v

                            # Штраф за разницу во времени
                            time_penalty = (
                                time_start_diff + time_end_diff
                            ) / 3600  # в часах

                            # Финальный score: чем больше, тем лучше
                            exact_match_score = base_score * 100 - time_penalty

                            # Выбираем совпадение с максимальным score
                            if exact_match_score > best_exact_match_score:
                                best_exact_match_score = exact_match_score
                                best_match_info = {
                                    "start_v": start_v_str,
                                    "end_v": end_v_str,
                                    "overlap_ratio": overlap_ratio_v,
                                    "time_start_diff": time_start_diff,
                                    "exact_match_score": exact_match_score,
                                }
                    else:
                        if overlap_ratio_v >= 0.8:
                            time_start_diff = abs((s1d - s2d).total_seconds())
                            time_end_diff = abs((f1d - f2d).total_seconds())

                            exact_match_score = (
                                overlap_ratio_v * 100
                                - (time_start_diff + time_end_diff) / 3600
                            )

                            if exact_match_score > best_exact_match_score:
                                best_exact_match_score = exact_match_score
                                best_match_info = {
                                    "start_v": start_v_str,
                                    "end_v": end_v_str,
                                    "overlap_ratio": overlap_ratio_v,
                                    "time_start_diff": time_start_diff,
                                    "exact_match_score": exact_match_score,
                                }

            if best_match_info:
                df.at[i, "overlap"] = True
                df.at[i, "Время выхода VIMB"] = best_match_info["start_v"]
                df.at[i, "Время окончания VIMB"] = best_match_info["end_v"]
                df.at[i, "overlap_ratio"] = best_match_info["overlap_ratio"]
                df.at[i, "time_start_diff"] = best_match_info["time_start_diff"]
                df.at[i, "exact_match_score"] = best_match_info["exact_match_score"]

        # Группируем по интервалам VIMB, чтобы каждый интервал VIMB использовался только один раз
        result_records = []
        used_vimb_intervals = set()

        # Сортируем по качеству совпадения (лучшие сначала)
        df_matched = df[df["overlap"] == True].copy()
        df_matched = df_matched.sort_values(
            ["exact_match_score", "time_start_diff"], ascending=[False, True]
        )

        for _, row in df_matched.iterrows():
            vimb_interval = (
                row["Дата"],
                row["Название программы"],
                row["Время выхода VIMB"],
                row["Время окончания VIMB"],
            )

            if vimb_interval not in used_vimb_intervals:
                result_records.append(
                    {
                        "Дата": row["Дата"],
                        "Название программы": row["Название программы"],
                        "Время выхода": row["Время выхода VIMB"],
                        "Время окончания": row["Время окончания VIMB"],
                        "Share": row["Share"],
                        "time_start_diff": row["time_start_diff"],
                        "overlap_ratio": row["overlap_ratio"],
                        "exact_match_score": row["exact_match_score"],
                    }
                )
                used_vimb_intervals.add(vimb_interval)

        if result_records:
            result_df = pd.DataFrame(result_records)
            # Сортируем по времени
            result_df = result_df.sort_values(["Дата", "Время выхода"])
            return result_df[
                [
                    "Дата",
                    "Название программы",
                    "Время выхода",
                    "Время окончания",
                    "Share",
                ]
            ]

        return pd.DataFrame(
            columns=[
                "Дата",
                "Название программы",
                "Время выхода",
                "Время окончания",
                "Share",
            ]
        )
    
    
    def find_matches(
            self, 
            vimb_init: pd.DataFrame, 
            plmrs_init: pd.DataFrame, 
            all_matches: list, 
            minutes_palomars: int,
            minutes_vimb = None,
            round_vimb: bool = False):
        """
            Вспомогательный метод для поиска совпадающих программ по простому merge путем округления слотов до определенного количества минут.
            Args:
                vimb_init: исходный датафрейм с сеткой VIMB, для которого будем искать совпадения.
                plmrs_init: исходный датафрейм с сеткой Mediascope, который будем использовать для поиска совпадений.
                all_matches: список с наденными программами.
                minutes_palomars: int: количество минут для округления времени сетки Mediascope.
                minutes_vimb: int: количество минут для округления времени сетки VIMB.
            Returns: 
                all_matches: list: обновленный список с найденными программами.
                found_programs: pd.DataFrame: полный датафрейм с найденными программами.
                vimb_clean_not_found: pd.DataFrame: оставшийся датафрейм с программами VIMB, для которых не удалось найти совпадения.
        """
        try:
            vimb_original = vimb_init.copy()
            pal_original = plmrs_init.copy()
            
            # Сохраняем исходные времена VIMB
            vimb_original['Время выхода_исходное'] = vimb_original['Время выхода']
            vimb_original['Время окончания_исходное'] = vimb_original['Время окончания']
            
            # Сохраняем индексы Pal
            pal_original = pal_original.reset_index(drop=True)
            pal_original['__pal_original_index'] = pal_original.index
            
            # Подготавливаем VIMB для поиска
            if round_vimb and minutes_vimb is not None:
                vimb_for_search = self._prepare_dataframe(
                    vimb_original, 
                    minutes_vimb, 
                    include_share=False,
                    preserve_original=True
                )
            else:
                vimb_for_search = self._add_original_time_columns(vimb_original)
            
            # Выполняем поиск
            matches, vimb_remaining, pal_remaining, pal_original_with_indices = self._process_schedule_step(
                pal_original, vimb_for_search, minutes_palomars
            )
            
            # Создаем НОВЫЙ список для текущих найденных программ
            current_found_matches = []
            used_pal_programs = pd.DataFrame()
            unused_pal_programs = pd.DataFrame()
            
            # ИНИЦИАЛИЗИРУЕМ список найденных индексов VIMB
            found_vimb_indices = set()
            
            if not matches.empty:
                # 1. Находим индексы VIMB программ из matches
                if '__vimb_index' in matches.columns:
                    found_vimb_indices = set(matches['__vimb_index'].dropna().astype(int).unique().tolist())
                
                # 2. Находим индексы Pal программ из matches
                pal_indices = []
                if '__pal_original_index' in matches.columns:
                    pal_indices = matches['__pal_original_index'].dropna().astype(int).unique().tolist()
                
                # 3. Получаем найденные программы VIMB по индексам
                found_vimb_programs = []
                for idx in found_vimb_indices:
                    if 0 <= idx < len(vimb_original):
                        prog = vimb_original.iloc[idx].copy()
                        
                        # Добавляем Share из matches
                        if 'Share' in matches.columns:
                            # Ищем совпадение по индексу
                            match_rows = matches[matches['__vimb_index'] == idx]
                            if not match_rows.empty:
                                prog['Share'] = match_rows.iloc[0]['Share']
                        
                        found_vimb_programs.append(prog)
                
                # 4. Использованные программы Pal
                if pal_indices:
                    used_pal_programs = pal_original_with_indices[
                        pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    ].copy()
                    used_pal_programs = used_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # 5. Неиспользованные программы Pal
                if not pal_original_with_indices.empty:
                    unused_mask = ~pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    unused_pal_programs = pal_original_with_indices[unused_mask].copy()
                    unused_pal_programs = unused_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # Добавляем найденные программы VIMB в ТЕКУЩИЕ matches
                if found_vimb_programs:
                    found_df = pd.DataFrame(found_vimb_programs)
                    
                    # Убираем лишние колонки и оставляем только нужные
                    columns_to_keep = ['Дата', 'Название программы']
                    
                    # Используем исходные времена
                    if 'Время выхода_исходное' in found_df.columns:
                        found_df['Время выхода'] = found_df['Время выхода_исходное']
                    if 'Время окончания_исходное' in found_df.columns:
                        found_df['Время окончания'] = found_df['Время окончания_исходное']
                    
                    columns_to_keep.extend(['Время выхода', 'Время окончания'])
                    
                    if 'Share' in found_df.columns:
                        columns_to_keep.append('Share')
                    
                    # Удаляем временные колонки
                    found_df = found_df[columns_to_keep].copy()
                    current_found_matches.append(found_df)
            
            # ПРОСТАЯ И НАДЕЖНАЯ ФИЛЬТРАЦИЯ VIMB ПО ИНДЕКСАМ
            # Оставляем только те программы, индексы которых НЕ в found_vimb_indices
            vimb_clean_indices = []
            for idx in range(len(vimb_original)):
                if idx not in found_vimb_indices:
                    vimb_clean_indices.append(idx)
            
            if vimb_clean_indices:
                vimb_clean = vimb_original.iloc[vimb_clean_indices].copy()
                # Оставляем только оригинальные колонки VIMB
                vimb_clean = vimb_clean[['Дата', 'Название программы',
                                        'Время выхода_исходное', 'Время окончания_исходное']].copy()
                vimb_clean = vimb_clean.rename(columns={
                    'Время выхода_исходное': 'Время выхода',
                    'Время окончания_исходное': 'Время окончания'
                })
            else:
                vimb_clean = pd.DataFrame(columns=['Дата', 'Название программы', 'Время выхода', 'Время окончания'])
            
            # Объединяем ранее найденные программы с текущими
            current_all_matches = all_matches.copy()
            if current_found_matches:
                current_all_matches.extend(current_found_matches)
            
            # Создаем финальный датафрейм найденных программ
            found_programs = pd.DataFrame()
            if current_all_matches:
                found_programs = pd.concat(current_all_matches, ignore_index=True)
                
                # Удаляем дубликаты в финальном результате
                if not found_programs.empty:
                    found_programs = found_programs.drop_duplicates(
                        subset=['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                    ).reset_index(drop=True)
            
            # ОЧИСТКА ФИНАЛЬНОГО ВЫВОДА: Убираем дублирующиеся колонки и оставляем только нужные
            if not found_programs.empty:
                # Определяем нужные колонки в правильном порядке
                final_columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                if 'Share' in found_programs.columns:
                    final_columns.append('Share')
                
                # Оставляем только нужные колонки
                found_programs = found_programs[final_columns].copy()
                
                # Удаляем возможные дубликаты колонок (если вдруг остались)
                found_programs = found_programs.loc[:, ~found_programs.columns.duplicated()]
            
            # Дебаг информация
            #print(f"\n=== ДЕБАГ ИНФОРМАЦИЯ ===")
            #print(f"Всего программ в VIMB: {len(vimb_original)}")
            #print(f"Найдено индексов в этом вызове: {len(found_vimb_indices)}")
            #print(f"Осталось программ VIMB: {len(vimb_clean)}")
            #print(f"Найдено программ в этом вызове: {len(current_found_matches[0]) if current_found_matches else 0}")
            
            if current_found_matches and len(current_found_matches[0]) > 0:
                #print("\nПримеры найденных программ (первые 3):")
                # Показываем только нужные колонки
                sample_df = current_found_matches[0].head(3)
                if 'Время выхода_исходное' in sample_df.columns:
                    sample_df = sample_df.drop(columns=['Время выхода_исходное'], errors='ignore')
                if 'Время окончания_исходное' in sample_df.columns:
                    sample_df = sample_df.drop(columns=['Время окончания_исходное'], errors='ignore')
                #print(sample_df)
                
            #if not vimb_clean.empty:
                #print("\nПримеры оставшихся программ VIMB (первые 3):")
                #print(vimb_clean.head(3))
            
            #print("\nКолонки в финальном found_programs:")
            #print(found_programs.columns.tolist())
            #print("=== КОНЕЦ ДЕБАГА ===\n")
            
            return current_all_matches, found_programs, vimb_clean, used_pal_programs, unused_pal_programs
            
        except Exception as e:
            #print(f"ERROR в find_matches_simple: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return all_matches, pd.DataFrame(), vimb_init.copy(), pd.DataFrame(), plmrs_init.copy()
            


    def filter_hour_programs(
        self, 
        vimb_init: pd.DataFrame, 
        plmrs_init: pd.DataFrame, 
        all_matches: list, 
        minutes_palomars: int):
        """
            Фильтрует часовые программы (40-60 минут) и находит совпадения с VIMB.
        """
        try:
            vimb_original = vimb_init.copy()
            pal_original = plmrs_init.copy()
            
            # Сохраняем исходные времена VIMB
            vimb_original['Время выхода_исходное'] = vimb_original['Время выхода']
            vimb_original['Время окончания_исходное'] = vimb_original['Время окончания']
            
            # Сохраняем индексы Pal
            pal_original = pal_original.reset_index(drop=True)
            pal_original['__pal_original_index'] = pal_original.index
            
            # 1. Подготовка часовых программ из Palomars
            pal_round = self._prepare_dataframe(pal_original.copy(), minutes_palomars, True)
            hour_programs = TVScheduleProcessor.group_broadcasts_by_hours(pal_round.copy())
            
            # 2. Фильтрация по длительности 40-60 минут
            hour_programs['Длительность_минуты'] = hour_programs.apply(
                TVScheduleProcessor.calculate_duration_minutes, axis=1
            )
            hour_filtered = hour_programs[
                (hour_programs['Длительность_минуты'] >= 40) &
                (hour_programs['Длительность_минуты'] <= 60)
            ].drop(columns=['Длительность_минуты'])
            
            # Если нет часовых программ для поиска
            if hour_filtered.empty:
                vimb_clean = vimb_init[['Дата', 'Название программы', 
                                        'Время выхода', 'Время окончания']].copy()
                found_programs = pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()
                return all_matches, found_programs, vimb_clean, pd.DataFrame(), pd.DataFrame()
            
            # 3. Подготавливаем VIMB для поиска (с округлением как у Palomars)
            vimb_for_search = self._prepare_dataframe(
                vimb_original, 
                minutes_palomars, 
                include_share=False,
                preserve_original=True
            )
            
            # 4. Выполняем поиск совпадений с часовыми программами
            matches, vimb_remaining, pal_remaining, pal_original_with_indices = self._process_schedule_step(
                hour_filtered, vimb_for_search, minutes_palomars
            )
            
            # Создаем список для текущих найденных программ
            current_found_matches = []
            used_pal_programs = pd.DataFrame()
            unused_pal_programs = pd.DataFrame()
            
            # ИНИЦИАЛИЗИРУЕМ список найденных индексов VIMB
            found_vimb_indices = set()
            
            if not matches.empty:
                # 1. Находим индексы VIMB программ из matches
                if '__vimb_index' in matches.columns:
                    found_vimb_indices = set(matches['__vimb_index'].dropna().astype(int).unique().tolist())
                
                # 2. Находим индексы Pal программ из matches
                pal_indices = []
                if '__pal_original_index' in matches.columns:
                    pal_indices = matches['__pal_original_index'].dropna().astype(int).unique().tolist()
                
                # 3. Получаем найденные программы VIMB по индексам
                found_vimb_programs = []
                for idx in found_vimb_indices:
                    if 0 <= idx < len(vimb_original):
                        prog = vimb_original.iloc[idx].copy()
                        
                        # Добавляем Share из matches
                        if 'Share' in matches.columns:
                            # Ищем совпадение по индексу
                            match_rows = matches[matches['__vimb_index'] == idx]
                            if not match_rows.empty:
                                prog['Share'] = match_rows.iloc[0]['Share']
                        
                        found_vimb_programs.append(prog)
                
                # 4. Использованные программы Pal
                if pal_indices:
                    used_pal_programs = pal_original_with_indices[
                        pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    ].copy()
                    used_pal_programs = used_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # 5. Неиспользованные программы Pal
                if not pal_original_with_indices.empty:
                    unused_mask = ~pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    unused_pal_programs = pal_original_with_indices[unused_mask].copy()
                    unused_pal_programs = unused_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # 6. Добавляем найденные программы VIMB в ТЕКУЩИЕ matches
                if found_vimb_programs:
                    found_df = pd.DataFrame(found_vimb_programs)
                    
                    # Убираем лишние колонки и оставляем только нужные
                    columns_to_keep = ['Дата', 'Название программы']
                    
                    # Используем исходные времена
                    if 'Время выхода_исходное' in found_df.columns:
                        found_df['Время выхода'] = found_df['Время выхода_исходное']
                    if 'Время окончания_исходное' in found_df.columns:
                        found_df['Время окончания'] = found_df['Время окончания_исходное']
                    
                    columns_to_keep.extend(['Время выхода', 'Время окончания'])
                    
                    if 'Share' in found_df.columns:
                        columns_to_keep.append('Share')
                    
                    # Удаляем временные колонки
                    found_df = found_df[columns_to_keep].copy()
                    current_found_matches.append(found_df)
            
            # 5. ПРОСТАЯ ФИЛЬТРАЦИЯ VIMB ПО ИНДЕКСАМ
            # Оставляем только те программы, индексы которых НЕ в found_vimb_indices
            vimb_clean_indices = []
            for idx in range(len(vimb_original)):
                if idx not in found_vimb_indices:
                    vimb_clean_indices.append(idx)
            
            if vimb_clean_indices:
                vimb_clean = vimb_original.iloc[vimb_clean_indices].copy()
                # Оставляем только оригинальные колонки VIMB
                vimb_clean = vimb_clean[['Дата', 'Название программы',
                                        'Время выхода_исходное', 'Время окончания_исходное']].copy()
                vimb_clean = vimb_clean.rename(columns={
                    'Время выхода_исходное': 'Время выхода',
                    'Время окончания_исходное': 'Время окончания'
                })
            else:
                vimb_clean = pd.DataFrame(columns=['Дата', 'Название программы', 'Время выхода', 'Время окончания'])
            
            # 6. Объединяем ранее найденные программы с текущими
            current_all_matches = all_matches.copy()
            if current_found_matches:
                current_all_matches.extend(current_found_matches)
            
            # 7. Создаем финальный датафрейм найденных программ
            found_programs = pd.DataFrame()
            if current_all_matches:
                found_programs = pd.concat(current_all_matches, ignore_index=True)
                
                # Удаляем дубликаты в финальном результате
                if not found_programs.empty:
                    found_programs = found_programs.drop_duplicates(
                        subset=['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                    ).reset_index(drop=True)
            
            # 8. ОЧИСТКА ФИНАЛЬНОГО ВЫВОДА
            if not found_programs.empty:
                # Определяем нужные колонки в правильном порядке
                final_columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                if 'Share' in found_programs.columns:
                    final_columns.append('Share')
                
                # Оставляем только нужные колонки
                found_programs = found_programs[final_columns].copy()
                
                # Удаляем возможные дубликаты колонок
                found_programs = found_programs.loc[:, ~found_programs.columns.duplicated()]
            
            return current_all_matches, found_programs, vimb_clean, used_pal_programs, unused_pal_programs
            
        except Exception as e:
            print(f"ERROR в filter_hour_programs: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            vimb_clean = vimb_init[['Дата', 'Название программы',
                                    'Время выхода', 'Время окончания']].copy()
            found_programs = pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()
            return all_matches, found_programs, vimb_clean, pd.DataFrame(), pd.DataFrame()


    def filter_long_programs(
            self, 
            vimb_init: pd.DataFrame, 
            plmrs_init: pd.DataFrame, 
            all_matches: list, 
            minutes_palomars: int,
            minutes_vimb: int,
            round_vimb: bool = False):
        """
        Фильтрует длинные программы (более 1 часа) и ищет совпадения с VIMB.
        """
        try:
            vimb_original = vimb_init.copy()
            pal_original = plmrs_init.copy()
            
            # Сохраняем исходные времена VIMB
            vimb_original['Время выхода_исходное'] = vimb_original['Время выхода']
            vimb_original['Время окончания_исходное'] = vimb_original['Время окончания']
            
            # Сохраняем индексы Pal
            pal_original = pal_original.reset_index(drop=True)
            pal_original['__pal_original_index'] = pal_original.index
            
            # 1. Подготавливаем Palomars (длинные программы)
            pal_round = self._prepare_dataframe(pal_original.copy(), minutes_palomars, True)
            long_programs = TVScheduleProcessor.join_broadcasts(pal_round)
            
            if long_programs.empty:
                vimb_clean = vimb_init[['Дата', 'Название программы', 
                                        'Время выхода', 'Время окончания']].copy()
                found_programs = pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()
                return all_matches, found_programs, vimb_clean, pd.DataFrame(), pd.DataFrame()
            
            # 2. Подготавливаем VIMB для поиска
            if round_vimb and minutes_vimb is not None:
                vimb_for_search = self._prepare_dataframe(
                    vimb_original, 
                    minutes_vimb, 
                    include_share=False,
                    preserve_original=True
                )
            else:
                vimb_for_search = self._add_original_time_columns(vimb_original)
            
            # 3. Выполняем поиск совпадений с длинными программами
            matches, vimb_remaining, pal_remaining, pal_original_with_indices = self._process_schedule_step(
                long_programs, vimb_for_search, minutes_palomars
            )
            
            # Создаем список для текущих найденных программ
            current_found_matches = []
            used_pal_programs = pd.DataFrame()
            unused_pal_programs = pd.DataFrame()
            
            # ИНИЦИАЛИЗИРУЕМ список найденных индексов VIMB
            found_vimb_indices = set()
            
            if not matches.empty:
                # 1. Находим индексы VIMB программ из matches
                if '__vimb_index' in matches.columns:
                    found_vimb_indices = set(matches['__vimb_index'].dropna().astype(int).unique().tolist())
                
                # 2. Находим индексы Pal программ из matches
                pal_indices = []
                if '__pal_original_index' in matches.columns:
                    pal_indices = matches['__pal_original_index'].dropna().astype(int).unique().tolist()
                
                # 3. Получаем найденные программы VIMB по индексам
                found_vimb_programs = []
                for idx in found_vimb_indices:
                    if 0 <= idx < len(vimb_original):
                        prog = vimb_original.iloc[idx].copy()
                        
                        # Добавляем Share из matches
                        if 'Share' in matches.columns:
                            # Ищем совпадение по индексу
                            match_rows = matches[matches['__vimb_index'] == idx]
                            if not match_rows.empty:
                                prog['Share'] = match_rows.iloc[0]['Share']
                        
                        found_vimb_programs.append(prog)
                
                # 4. Использованные программы Pal
                if pal_indices:
                    used_pal_programs = pal_original_with_indices[
                        pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    ].copy()
                    used_pal_programs = used_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # 5. Неиспользованные программы Pal
                if not pal_original_with_indices.empty:
                    unused_mask = ~pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    unused_pal_programs = pal_original_with_indices[unused_mask].copy()
                    unused_pal_programs = unused_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # 6. Добавляем найденные программы VIMB в ТЕКУЩИЕ matches
                if found_vimb_programs:
                    found_df = pd.DataFrame(found_vimb_programs)
                    
                    # Убираем лишние колонки и оставляем только нужные
                    columns_to_keep = ['Дата', 'Название программы']
                    
                    # Используем исходные времена
                    if 'Время выхода_исходное' in found_df.columns:
                        found_df['Время выхода'] = found_df['Время выхода_исходное']
                    if 'Время окончания_исходное' in found_df.columns:
                        found_df['Время окончания'] = found_df['Время окончания_исходное']
                    
                    columns_to_keep.extend(['Время выхода', 'Время окончания'])
                    
                    if 'Share' in found_df.columns:
                        columns_to_keep.append('Share')
                    
                    # Удаляем временные колонки
                    found_df = found_df[columns_to_keep].copy()
                    current_found_matches.append(found_df)
            
            # 5. ПРОСТАЯ ФИЛЬТРАЦИЯ VIMB ПО ИНДЕКСАМ
            # Оставляем только те программы, индексы которых НЕ в found_vimb_indices
            vimb_clean_indices = []
            for idx in range(len(vimb_original)):
                if idx not in found_vimb_indices:
                    vimb_clean_indices.append(idx)
            
            if vimb_clean_indices:
                vimb_clean = vimb_original.iloc[vimb_clean_indices].copy()
                # Оставляем только оригинальные колонки VIMB
                vimb_clean = vimb_clean[['Дата', 'Название программы',
                                        'Время выхода_исходное', 'Время окончания_исходное']].copy()
                vimb_clean = vimb_clean.rename(columns={
                    'Время выхода_исходное': 'Время выхода',
                    'Время окончания_исходное': 'Время окончания'
                })
            else:
                vimb_clean = pd.DataFrame(columns=['Дата', 'Название программы', 'Время выхода', 'Время окончания'])
            
            # 6. Объединяем ранее найденные программы с текущими
            current_all_matches = all_matches.copy()
            if current_found_matches:
                current_all_matches.extend(current_found_matches)
            
            # 7. Создаем финальный датафрейм найденных программ
            found_programs = pd.DataFrame()
            if current_all_matches:
                found_programs = pd.concat(current_all_matches, ignore_index=True)
                
                # Удаляем дубликаты в финальном результате
                if not found_programs.empty:
                    found_programs = found_programs.drop_duplicates(
                        subset=['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                    ).reset_index(drop=True)
            
            # 8. ОЧИСТКА ФИНАЛЬНОГО ВЫВОДА
            if not found_programs.empty:
                # Определяем нужные колонки в правильном порядке
                final_columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                if 'Share' in found_programs.columns:
                    final_columns.append('Share')
                
                # Оставляем только нужные колонки
                found_programs = found_programs[final_columns].copy()
                
                # Удаляем возможные дубликаты колонок
                found_programs = found_programs.loc[:, ~found_programs.columns.duplicated()]
            
            return current_all_matches, found_programs, vimb_clean, used_pal_programs, unused_pal_programs
            
        except Exception as e:
            print(f"ERROR в filter_long_programs: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            vimb_clean = vimb_init[['Дата', 'Название программы',
                                    'Время выхода', 'Время окончания']].copy()
            found_programs = pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()
            return all_matches, found_programs, vimb_clean, pd.DataFrame(), pd.DataFrame()


    def filter_long_overlap(
            self, 
            vimb_init: pd.DataFrame, 
            plmrs_init: pd.DataFrame, 
            all_matches: list, 
            minutes_palomars: int,
            minutes_vimb: int,
            round_vimb: bool = False):
        """
        Фильтрует длинные программы (более 1 часа) и ищет совпадения с VIMB с помощью Overlapping.
        """
        try:
            vimb_original = vimb_init.copy()
            pal_original = plmrs_init.copy()
            
            # Сохраняем исходные времена VIMB
            vimb_original['Время выхода_исходное'] = vimb_original['Время выхода']
            vimb_original['Время окончания_исходное'] = vimb_original['Время окончания']
            
            # Сохраняем индексы Pal
            pal_original = pal_original.reset_index(drop=True)
            pal_original['__pal_original_index'] = pal_original.index
            
            # 1. Подготавливаем Palomars (длинные программы)
            pal_round = self._prepare_dataframe(pal_original.copy(), minutes_palomars, True)
            
            # Заменяем значения начиная со второго
            for i in range(1, len(pal_round)):
                pal_round.loc[i, "Время выхода"] = pal_round.loc[i - 1, "Время окончания"]
            
            long_programs = TVScheduleProcessor.join_broadcasts(pal_round)
            
            if long_programs.empty:
                vimb_clean = vimb_init[['Дата', 'Название программы', 
                                        'Время выхода', 'Время окончания']].copy()
                found_programs = pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()
                return all_matches, found_programs, vimb_clean, pd.DataFrame(), pd.DataFrame()
            
            # 2. Подготавливаем VIMB для поиска
            if round_vimb and minutes_vimb is not None:
                vimb_for_search = self._prepare_dataframe(
                    vimb_original, 
                    minutes_vimb, 
                    include_share=False,
                    preserve_original=True
                )
            else:
                vimb_for_search = self._add_original_time_columns(vimb_original)
            
            # 3. Ищем overlapping совпадения
            vimb_for_search_adj = self._adjust_end_time(vimb_for_search.copy())
            long_programs_adj = self._adjust_end_time(long_programs.copy())
            
            # Используем метод broadcasts_overlaping_OLD для поиска пересечений
            res_overlap = self.broadcasts_overlaping_OLD(long_programs_adj, vimb_for_search_adj)
            print(long_programs_adj)
            
            if res_overlap.empty:
                vimb_clean = vimb_original[['Дата', 'Название программы',
                                            'Время выхода_исходное', 'Время окончания_исходное']].copy()
                vimb_clean = vimb_clean.rename(columns={
                    'Время выхода_исходное': 'Время выхода',
                    'Время окончания_исходное': 'Время окончания'
                })
                found_programs = pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()
                return all_matches, found_programs, vimb_clean, pd.DataFrame(), pd.DataFrame()
            
            # Форматируем дату
            res_overlap["Дата"] = pd.to_datetime(res_overlap["Дата"])
            res_overlap["Дата"] = res_overlap["Дата"].dt.strftime("%Y-%m-%d")
            
            # 4. Используем _process_schedule_step для обработки найденных overlapping программ
            matches, vimb_remaining, pal_remaining, pal_original_with_indices = self._process_schedule_step(
                res_overlap, vimb_for_search, minutes_palomars
            )
            
            # Создаем список для текущих найденных программ
            current_found_matches = []
            used_pal_programs = pd.DataFrame()
            unused_pal_programs = pd.DataFrame()
            
            # ИНИЦИАЛИЗИРУЕМ список найденных индексов VIMB
            found_vimb_indices = set()
            
            if not matches.empty:
                # 1. Находим индексы VIMB программ из matches
                if '__vimb_index' in matches.columns:
                    found_vimb_indices = set(matches['__vimb_index'].dropna().astype(int).unique().tolist())
                
                # 2. Находим индексы Pal программ из matches
                pal_indices = []
                if '__pal_original_index' in matches.columns:
                    pal_indices = matches['__pal_original_index'].dropna().astype(int).unique().tolist()
                
                # 3. Получаем найденные программы VIMB по индексам
                found_vimb_programs = []
                for idx in found_vimb_indices:
                    if 0 <= idx < len(vimb_original):
                        prog = vimb_original.iloc[idx].copy()
                        
                        # Добавляем Share из matches
                        if 'Share' in matches.columns:
                            # Ищем совпадение по индексу
                            match_rows = matches[matches['__vimb_index'] == idx]
                            if not match_rows.empty:
                                prog['Share'] = match_rows.iloc[0]['Share']
                        
                        found_vimb_programs.append(prog)
                
                # 4. Использованные программы Pal
                if pal_indices:
                    used_pal_programs = pal_original_with_indices[
                        pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    ].copy()
                    used_pal_programs = used_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # 5. Неиспользованные программы Pal
                if not pal_original_with_indices.empty:
                    unused_mask = ~pal_original_with_indices['__pal_original_index'].isin(pal_indices)
                    unused_pal_programs = pal_original_with_indices[unused_mask].copy()
                    unused_pal_programs = unused_pal_programs.drop(columns=['__pal_original_index'], errors='ignore')
                
                # 6. Добавляем найденные программы VIMB в ТЕКУЩИЕ matches
                if found_vimb_programs:
                    found_df = pd.DataFrame(found_vimb_programs)
                    
                    # Убираем лишние колонки и оставляем только нужные
                    columns_to_keep = ['Дата', 'Название программы']
                    
                    # Используем исходные времена
                    if 'Время выхода_исходное' in found_df.columns:
                        found_df['Время выхода'] = found_df['Время выхода_исходное']
                    if 'Время окончания_исходное' in found_df.columns:
                        found_df['Время окончания'] = found_df['Время окончания_исходное']
                    
                    columns_to_keep.extend(['Время выхода', 'Время окончания'])
                    
                    if 'Share' in found_df.columns:
                        columns_to_keep.append('Share')
                    
                    # Удаляем временные колонки
                    found_df = found_df[columns_to_keep].copy()
                    current_found_matches.append(found_df)
            
            # 5. ПРОСТАЯ ФИЛЬТРАЦИЯ VIMB ПО ИНДЕКСАМ
            # Оставляем только те программы, индексы которых НЕ в found_vimb_indices
            vimb_clean_indices = []
            for idx in range(len(vimb_original)):
                if idx not in found_vimb_indices:
                    vimb_clean_indices.append(idx)
            
            if vimb_clean_indices:
                vimb_clean = vimb_original.iloc[vimb_clean_indices].copy()
                # Оставляем только оригинальные колонки VIMB
                vimb_clean = vimb_clean[['Дата', 'Название программы',
                                        'Время выхода_исходное', 'Время окончания_исходное']].copy()
                vimb_clean = vimb_clean.rename(columns={
                    'Время выхода_исходное': 'Время выхода',
                    'Время окончания_исходное': 'Время окончания'
                })
            else:
                vimb_clean = pd.DataFrame(columns=['Дата', 'Название программы', 'Время выхода', 'Время окончания'])
            
            # 6. Объединяем ранее найденные программы с текущими
            current_all_matches = all_matches.copy()
            if current_found_matches:
                current_all_matches.extend(current_found_matches)
            
            # 7. Создаем финальный датафрейм найденных программ
            found_programs = pd.DataFrame()
            if current_all_matches:
                found_programs = pd.concat(current_all_matches, ignore_index=True)
                
                # Удаляем дубликаты в финальном результате
                if not found_programs.empty:
                    found_programs = found_programs.drop_duplicates(
                        subset=['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                    ).reset_index(drop=True)
            
            # 8. ОЧИСТКА ФИНАЛЬНОГО ВЫВОДА
            if not found_programs.empty:
                # Определяем нужные колонки в правильном порядке
                final_columns = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                if 'Share' in found_programs.columns:
                    final_columns.append('Share')
                
                # Оставляем только нужные колонки
                found_programs = found_programs[final_columns].copy()
                
                # Удаляем возможные дубликаты колонок
                found_programs = found_programs.loc[:, ~found_programs.columns.duplicated()]
            
            return current_all_matches, found_programs, vimb_clean, used_pal_programs, unused_pal_programs
            
        except Exception as e:
            print(f"ERROR в filter_long_overlap: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            vimb_clean = vimb_init[['Дата', 'Название программы',
                                    'Время выхода', 'Время окончания']].copy()
            found_programs = pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()
            return all_matches, found_programs, vimb_clean, pd.DataFrame(), pd.DataFrame()

    

    def print_comments(self, found_programs, clean_vimb):
        """
            Метод для печати комментариев по найденным и не найденным программам
        """
        if len(found_programs) != 0:
            print(
                color.BOLD + color.GREEN
                + f"НАЙДЕННЫЕ ПРОГРАММЫ: {len(found_programs)}"
                + color.END,
                sep = '\n', end = '\n'
            )
            print(found_programs)
        print('=' * 120)

        if len(clean_vimb) != 0:
            print(
                color.BOLD + color.BLUE
                + f"ВИМБ, который осталось найти: {len(clean_vimb)}"
                + color.END,
                sep = '\n', end = '\n',
            )
            print(clean_vimb)
        
        if len(clean_vimb) == 0:
            print(color.BOLD + color.BLUE + 'Весь ВИМБ сопоставили' + color.END, sep = '\n', end = '\n')
        
        if len(found_programs) == 0:
            print(color.BOLD + color.BLUE + 'Не нашли ни одной программы' + color.END, sep = '\n', end = '\n')

        print('=' * 120)

    

    def match_small_programs(self, vimb: pd.DataFrame, palomars: pd.DataFrame, matches: list):
        """
            Функция для поиска обычных (недлинных) программ.
            Всегда использует исходный Palomars для поиска.
            
            Args:
                vimb: исходный датафрейм VIMB
                palomars: исходный датафрейм Palomars
                matches: список уже найденных программ
            
            Returns:
                current_found_programs: найденные программы в виде датафрейма
                current_vimb: оставшийся VIMB
                palomars_used: использованные программы Palomars
                palomars_unused: неиспользованные программы Palomars
        """
        
        minutes_to_round = [5, 10]
    
        current_matches = matches.copy()
        current_vimb = vimb.copy()
        current_found_programs = pd.DataFrame()
        
        # Разные настройки для разных методов
        matching_settings = [
            {'minutes_vimb': 5, 'round_vimb': False},
            {'minutes_vimb': 5, 'round_vimb': True},
            {'minutes_vimb': 10, 'round_vimb': False},
            {'minutes_vimb': 10, 'round_vimb': True},
        ]
        
        # Для отслеживания использованных программ Palomars
        all_used_palomars = []
        all_unused_palomars = []
        
        # Переменные для хранения результатов между итерациями
        iteration_results = []
        
        for setting in matching_settings:
            for minute in minutes_to_round:
                print(f"\nfind_matches: Pal={minute} мин, VIMB={setting['minutes_vimb']}, round={setting['round_vimb']}")
                
                # Сохраняем состояние перед вызовом find_matches
                vimb_before = current_vimb.copy()
                
                # Вызываем find_matches
                new_matches, found_programs, remaining_vimb, used_plmrs, unused_plmrs = self.find_matches(
                    vimb_init = current_vimb,
                    plmrs_init = palomars,  # Всегда используем исходный Palomars
                    all_matches = current_matches,
                    minutes_palomars = minute,
                    minutes_vimb = setting['minutes_vimb'],
                    round_vimb = setting['round_vimb']
                )
                
                # Проверяем, были ли найдены новые программы
                if not found_programs.empty:
                    # Обновляем списки
                    current_matches = new_matches
                    current_found_programs = pd.concat([current_found_programs, found_programs], ignore_index=True)
                    current_vimb = remaining_vimb  # Это ключевое изменение!
                    
                    # Собираем использованные программы Palomars
                    if not used_plmrs.empty:
                        all_used_palomars.append(used_plmrs)
                    
                    if not unused_plmrs.empty:
                        all_unused_palomars.append(unused_plmrs)
                    
                    # Сохраняем результат итерации для отладки
                    iteration_results.append({
                        'setting': setting,
                        'minute': minute,
                        'found': len(found_programs),
                        'remaining': len(current_vimb)
                    })
                
                # Проверяем, очистился ли весь VIMB
                if len(current_vimb) == 0:
                    print("Весь VIMB очищен, завершаем поиск")
                    break
            
            # Выходим из внешнего цикла, если VIMB очищен
            if len(current_vimb) == 0:
                break
        
        # Удаляем дубликаты в найденных программах
        if not current_found_programs.empty:
            current_found_programs = current_found_programs.drop_duplicates(
                subset=['Дата', 'Название программы', 'Время выхода', 'Время окончания']
            ).reset_index(drop=True)
        
        # Формируем итоговые использованные программы Palomars
        palomars_used = pd.DataFrame()
        if all_used_palomars:
            palomars_used = pd.concat(all_used_palomars, ignore_index=True)
            
            # Удаляем дубликаты
            if not palomars_used.empty:
                palomars_used = palomars_used.drop_duplicates(
                    subset=['Дата', 'Название программы', 'Время выхода', 'Время окончания']
                ).reset_index(drop=True)
        
        # Формируем неиспользованные программы Palomars
        palomars_unused = pd.DataFrame()
        if all_unused_palomars:
            # Начинаем с первого unused
            palomars_unused = all_unused_palomars[0].copy()
            
            # Находим пересечение всех unused датафреймов
            for i in range(1, len(all_unused_palomars)):
                if not palomars_unused.empty and not all_unused_palomars[i].empty:
                    # Находим общие программы
                    current_keys = set()
                    for _, row in palomars_unused.iterrows():
                        current_keys.add((
                            str(row['Дата']),
                            str(row['Название программы']),
                            str(row['Время выхода']),
                            str(row['Время окончания'])
                        ))
                    
                    next_unused = all_unused_palomars[i]
                    common_rows = []
                    for _, row in next_unused.iterrows():
                        row_key = (
                            str(row['Дата']),
                            str(row['Название программы']),
                            str(row['Время выхода']),
                            str(row['Время окончания'])
                        )
                        if row_key in current_keys:
                            common_rows.append(row)
                    
                    if common_rows:
                        palomars_unused = pd.DataFrame(common_rows)
                    else:
                        palomars_unused = pd.DataFrame()
                        break

        self.print_comments(current_found_programs, current_vimb)
        
        result = {
                    'found': current_found_programs.reset_index(drop = True),
                    'vimb_remain': current_vimb.reset_index(drop = True), 
                    'used_palomars': palomars_used.reset_index(drop = True),
                    'palomars_remain': palomars_unused.reset_index(drop = True)
        }
        return result



    def process_schedules(
                self, vimb: pd.DataFrame, palomars: pd.DataFrame, verbose: bool = True
            ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
        """
            Основной метод обработки телепрограмм

            Args:
                vimb: pd.DataFrame: датафрейм с сеткой ВИМБ
                palomars: pd.DataFrame: датафрейм с сеткой Mediascope
                verbose: bool:
            Returns:
                Tuple: (результирующий DataFrame, статистика, найденные программы, оставшиеся программы)
        """
        # Сохраняем исходную суммарную долю, чтобы можно было проверить корректность после обработки
        share_sum_init = np.sum(list(palomars['Share']))

        # Инициализация
        self.vimb_init = vimb.copy()
        self.palomars_init = palomars.copy()

        if verbose:
            print(f'Исходные данные: VIMB = {len(self.vimb_init)}, Palomars = {len(self.palomars_init)}')

        # Список для хранения найденных программ
        all_matches = []

        ############################ ШАГ 1: Обработка мелких программ, которые легко сджойнить. Округляем Palomars до 5 мин ############################
        if verbose:
            print(
                color.BOLD + color.RED
                + 'ШАГ 1: Обработка мелких программ, которые легко сджойнить'
                + color.END, sep = '\n', end = '\n'
            )

        all_matches, found_programs_after_1, vimb_clean_after_1 = self.find_mathes(vimb, palomars, all_matches, 5, None)

        if verbose:
            self.print_comments(found_programs_after_1, vimb_clean_after_1)

        ############################ ШАГ 2: Обработка длинных программ с разбивкой по часам (округление 10 минут) ############################
        if verbose:
            print(
                color.BOLD + color.RED
                + 'ШАГ 2: Обработка длинных программ с разбивкой по часам (округление 10 минут)'
                + color.END, sep = '\n', end = '\n',
            )

        all_matches, found_programs_after_2, vimb_clean_after_2 = self.filter_hour_programs(vimb_clean_after_1, palomars, all_matches, 5)
        print(len(found_programs_after_1), len(found_programs_after_2))

        if verbose:
            self.print_comments(found_programs_after_2, vimb_clean_after_2)

        ############################ ШАГ 3: Обработка мелких программ, которые легко сджойнить. Округляем Palomars до 10 мин ############################
        if verbose:
            print(
                color.BOLD + color.RED
                + 'ШАГ 3: Обработка мелких программ, которые легко сджойнить'
                + color.END, sep = '\n', end = '\n'
            )

        # Округлям VIMB до 10 минут и Palomars до 10 минут
        all_matches, found_programs_after_3, vimb_clean_after_3 = self.find_mathes(vimb_clean_after_2, palomars, all_matches, 10, 10, True)
        print(len(found_programs_after_2), len(found_programs_after_3))
    

        if verbose:
            self.print_comments(found_programs_after_3, vimb_clean_after_3)

        ############################ ШАГ 4: Обработка длинных программ (округление 10 минут) ############################
        if verbose:
            print(
                color.BOLD + color.RED
                + 'ШАГ 4: Обработка длинных программ (округление 10 минут)'
                + color.END, sep = '\n', end = '\n'
            )

        all_matches, found_programs_after_4, vimb_clean_after_4 = self.filter_long_programs(
                                                                                vimb_clean_after_3, 
                                                                                palomars, 
                                                                                all_matches, 
                                                                                5, None, False)
        print(len(found_programs_after_3), len(found_programs_after_4))

        if len(found_programs_after_3) == len(found_programs_after_4):
            all_matches, found_programs_after_4, vimb_clean_after_4 = self.filter_long_programs(
                                                                                vimb_clean_after_3, 
                                                                                palomars, 
                                                                                all_matches, 
                                                                                5, 5, True)
            

        if verbose:
            self.print_comments(found_programs_after_4, vimb_clean_after_4)

        ############################ ШАГ 5: Обработка длинных программ с overlaping ###########################
        if verbose:
            print(color.BOLD + color.RED + 'ШАГ 5: Обработка длинных программ с overlapping' + color.END)

        vimb_current = vimb_clean_after_4.copy()
        matches_current = all_matches.copy()

        df1, df2 = TVScheduleProcessor.split_programs_by_time(vimb_current)

        # 1. Обрабатываем df2
        if not df2.empty:
            # Сохраняем исходный VIMB для удаления
            original_vimb = vimb_current.copy()
            
            # Ищем совпадения
            matches_current, found_df2, vimb_cleaned_df2 = self.filter_long_overlap(
                df2, palomars, all_matches, 5, None, False
            )
            
            
            if found_df2 is not None and not found_df2.empty:
                # Удаляем найденные программы из исходного VIMB
                vimb_after_removal = self._remove_found_programs(
                    self._adjust_end_time(original_vimb, 'Время окончания'),
                    self._adjust_end_time(found_df2, 'Время окончания')
                ).reset_index(drop=True)
                
                vimb_current = self._extract_core_info(vimb_after_removal, False)

        # 2. Обрабатываем df1
        if not df1.empty:
            # Фильтруем df1 - только то, что осталось в vimb_clean
            # (комбинируем по ключевым колонкам)
            merged = pd.merge(
                vimb_current[['Дата', 'Название программы', 'Время выхода', 'Время окончания']],
                df1,
                on=['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
                how='inner'
            )
            
            if not merged.empty:
                original_vimb = vimb_current.copy()
                
                matches_current, found_df1, vimb_cleaned_df1 = self.find_mathes(
                    merged, palomars, all_matches, 10, 10, True
                )
                
                print('==============================')
                print(merged)
                print('.-----------------------------')
                print(palomars)
                print('.-----------------------------')
                print(found_df1)
                print('==============================')
                if found_df1 is not None and not found_df1.empty:
                    vimb_after_removal = self._remove_found_programs(
                        self._adjust_end_time(original_vimb, 'Время окончания'),
                        self._adjust_end_time(found_df1, 'Время окончания')
                    ).reset_index(drop=True)

                    
                    vimb_current = self._extract_core_info(vimb_after_removal, False)
        
        all_matches = matches_current
        vimb_clean = vimb_current.copy()

        
        if verbose:
            print(color.BOLD + 'НАЙДЕННЫЕ ПРОГРАММЫ:' + color.END, sep = '\n', end = '\n')
            print(pd.concat(all_matches).reset_index(drop = True))
            print('#' * 120)

            if len(vimb_clean) != 0:
                self.print_comments(pd.concat(all_matches).reset_index(drop = True), vimb_clean)

        ############################ ШАГ 6: Дополнительный шаг, если после остальных остались ненайденные ###########################
        if len(vimb_clean) != 0:

            if verbose:
                print(
                    color.BOLD + color.RED
                    + 'ШАГ 6: Обработка длинных программ с разбивкой по часам (округление 10 минут)'
                    + color.END, sep = '\n', end = '\n'
                )

            all_matches, found_programs_after_6, vimb_clean_after_6 = self.filter_hour_programs(vimb_clean, palomars, all_matches, 10)
            print(len(found_programs_after_5), len(found_programs_after_6))

            if verbose:
                self.print_comments(found_programs_after_6, vimb_clean_after_6)

            vimb_clean_after_5 = vimb_clean_after_6
            found_programs_after_5 = found_programs_after_6
        
        ############################ ШАГ 7: Дополнительный шаг, если после остальных остались ненайденные ###########################
        if len(vimb_clean_after_5) != 0:
            if verbose:
                print(
                    color.BOLD + color.RED
                    + 'ШАГ 7: Попытка обработки коротких программ (округление 10 минут)'
                    + color.END, sep = '\n', end = '\n'
                )
            
            all_matches, found_programs_after_7, vimb_clean_after_7 = self.find_mathes(vimb_clean_after_5, palomars, all_matches, 10, None)

            if verbose:
                self.print_comments(found_programs_after_7, vimb_clean_after_7)

        ############################ ШАГ 7: ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ ############################

        found_programs = pd.concat(all_matches).reset_index(drop = True)

        # Проверяем, что в таблице VIMB действительно все найдено
        merged = pd.merge(
            self._adjust_end_time(vimb),
            found_programs,
            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
            how = 'left',
        )

        # Проверяем, что в таблице действительно все найдено

        if len(merged) != len(vimb):
            print('Нужен дополнительный поиск!')

        if share_sum_init != np.sum(list(found_programs['Share'])):
            share_sum_curr = np.sum(list(found_programs['Share']))
            print(
                f'Обнаружено несовпадение суммарной доли по дню! Целевой показатель {np.round(share_sum_init, 4)}, а по итогу вышло {np.round(share_sum_curr, 4)}.'
            )

        return merged