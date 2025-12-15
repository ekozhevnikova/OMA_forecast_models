import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Callable
from datetime import timedelta, datetime, time
from dateutil.relativedelta import relativedelta
from difflib import SequenceMatcher


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

    def _add_original_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление колонок с исходным временем, чтобы можно было сопоставить с исходным датафреймом VIMB.
        Args:
            df: датафрейм, в котором хотим добавить всопомгательные столбцы
        Returns:
            pd.DataFrame: датафрейм df с двумя дополнительными колонками с исходными временами слотов
        """
        result = df.copy()

        if "Время выхода_исходное" not in result.columns:
            result["Время выхода_исходное"] = result["Время выхода"]

        if "Время окончания_исходное" not in result.columns:
            result["Время окончания_исходное"] = result["Время окончания"]

        return result

    def _prepare_dataframe(
        self, df: pd.DataFrame, minutes: int, include_share: bool = True
    ) -> pd.DataFrame:
        """
        Подготовка датафрейма. Времена слотов округляются до установленных минут.
        Таким образом, датафрейм подготавливается для дальнейшего анализа.
        """
        data = df.copy()

        # Сохраняем исходные значения времени
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
            "Время выхода_исходное",
            "Время окончания_исходное",
        ]

        if include_share and "Share" in data.columns:
            columns.insert(4, "Share")

        result = data[columns].copy()
        result["Дата"] = pd.to_datetime(result["Дата"]).dt.strftime("%Y-%m-%d")

        return result

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
            found_df[merge_keys], on=merge_keys, how="left", indicator=True
        )

        return merged.query('_merge == "left_only"').drop("_merge", axis=1)

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
                    drop=True
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
                columns=[
                    "Дата",
                    "Название программы",
                    "Время выхода",
                    "Время окончания",
                    "Share",
                ]
            )

    @staticmethod
    def join_broadcasts(data):
        """
        Упрощенная версия для объединения трансляций в рамках одного дня.
        Учитывает эфирные сутки (05:00-04:59).
        ИСПРАВЛЕНО: Теперь находит все сегменты программы в течение дня, даже если они прерываются другими программами.
        """
        df = data.copy()

        # Проверяем количество дней
        if df["Дата"].nunique() > 1:
            print("Предупреждение: Рекомендуется обрабатывать по одному дню за раз")

        # Сортируем по времени с учетом эфирных суток
        def broadcast_time_key(time_str):
            """Ключ для сортировки по эфирным суткам"""
            h, m, s = map(int, time_str.split(":"))
            # Время с 05:00 считаем текущего дня, с 00:00-04:59 - следующего
            return (0 if h >= 5 else 1, h, m, s)

        # Добавляем ключ сортировки
        df["sort_key"] = df["Время выхода"].apply(broadcast_time_key)
        df = df.sort_values(["Дата", "sort_key"]).reset_index(drop=True)
        df = df.drop("sort_key", axis=1)

        results = []

        # Обрабатываем каждую программу отдельно
        for program in df["Название программы"].unique():
            program_df = df[df["Название программы"] == program].copy()
            program_df["sort_key"] = program_df["Время выхода"].apply(
                broadcast_time_key
            )
            program_df = program_df.sort_values("sort_key").reset_index(drop=True)
            program_df = program_df.drop("sort_key", axis=1)

            # Находим все сегменты этой программы
            segments = []
            for _, row in program_df.iterrows():
                segments.append(
                    {
                        "date": row["Дата"],
                        "start": row["Время выхода"],
                        "end": row["Время окончания"],
                        "share": row["Share"],
                    }
                )

            if not segments:
                continue

            # Объединяем смежные сегменты
            merged_segments = []
            current_start = segments[0]["start"]
            current_end = segments[0]["end"]
            current_shares = [segments[0]["share"]]
            current_date = segments[0]["date"]

            for i in range(1, len(segments)):
                next_start = segments[i]["start"]
                next_end = segments[i]["end"]

                # Проверяем, идет ли следующая трансляция сразу после текущей
                # Учитываем переход через полночь
                if current_end == next_start or (
                    current_end == "04:59:59" and next_start == "05:00:00"
                ):
                    # Прямая последовательность или переход через границу эфирных суток
                    current_end = next_end
                    current_shares.append(segments[i]["share"])
                else:
                    # Сохраняем текущую группу и начинаем новую
                    merged_segments.append(
                        {
                            "date": current_date,
                            "start": current_start,
                            "end": current_end,
                            "shares": current_shares.copy(),
                        }
                    )
                    current_start = next_start
                    current_end = next_end
                    current_shares = [segments[i]["share"]]
                    current_date = segments[i]["date"]

            # Добавляем последнюю группу
            merged_segments.append(
                {
                    "date": current_date,
                    "start": current_start,
                    "end": current_end,
                    "shares": current_shares.copy(),
                }
            )

            # Создаем записи для каждой объединенной группы
            for segment in merged_segments:
                results.append(
                    {
                        "Дата": segment["date"],
                        "Название программы": program,
                        "Время выхода": segment["start"],
                        "Время окончания": segment["end"],
                        "Share": sum(segment["shares"]),
                        "Количество_сегментов": len(segment["shares"]),
                    }
                )

        result_df = pd.DataFrame(results)

        if len(result_df) > 0:
            # Сортируем итоговый результат
            result_df["sort_key"] = result_df["Время выхода"].apply(broadcast_time_key)
            result_df = (
                result_df.sort_values("sort_key")
                .drop("sort_key", axis=1)
                .reset_index(drop=True)
            )

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
                columns=[
                    "Дата",
                    "Название программы",
                    "Время выхода",
                    "Время окончания",
                    "Share",
                ]
            )

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

    def _process_schedule_step(
        self,
        palomars: pd.DataFrame,
        vimb: pd.DataFrame,
        minutes: int,
        step_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Обработка одного шага сопоставления с округлением времени
        """

        vimb = self._add_original_time_columns(vimb)

        # Подготовка данных
        pal_processed = self._prepare_dataframe(palomars, minutes, include_share=True)

        # Корректировка времени окончания
        pal_processed = self._adjust_end_time(pal_processed, "Время окончания")
        vimb_processed = self._adjust_end_time(vimb, "Время окончания")

        # Поиск совпадений
        merge_keys = ["Дата", "Название программы", "Время выхода", "Время окончания"]

        matches = pd.merge(vimb_processed, pal_processed, on=merge_keys, how="inner")[
            ["Дата", "Название программы", "Время выхода", "Время окончания", "Share"]
        ]

        # Сохранение статистики
        self.stats[f"{step_name}_matches"] = len(matches)

        # Удаление найденных
        vimb_remaining = self._remove_found_programs(
            vimb_processed, matches
        ).reset_index(drop=True)
        pal_remaining = self._remove_found_programs(pal_processed, matches).reset_index(
            drop=True
        )

        self.stats[f"{step_name}_vimb_remaining"] = len(vimb_remaining)
        self.stats[f"{step_name}_pal_remaining"] = len(pal_remaining)

        return matches, vimb_remaining, pal_remaining

    def _extract_core_info(
        self, df: pd.DataFrame, include_share: bool = True
    ) -> pd.DataFrame:
        """
        Извлечение основной информации с переименованием колонок
        """
        if "Время выхода_исходное" and "Время окончания_исходное" in df.columns:
            if include_share and "Share" in df.columns:
                columns = [
                    "Дата",
                    "Название программы",
                    "Время выхода_исходное",
                    "Время окончания_исходное",
                    "Share",
                ]
            else:
                columns = [
                    "Дата",
                    "Название программы",
                    "Время выхода_исходное",
                    "Время окончания_исходное",
                ]

            result = df[columns].copy()
            result.rename(
                columns={
                    "Время выхода_исходное": "Время выхода",
                    "Время окончания_исходное": "Время окончания",
                },
                inplace=True,
            )

            return result

        else:
            return df

    def broadcasts_overlaping_OLD(self, df, vimb):
        """
        Метод для сопоставления программ методом перекрытия двух длительностей.
        """
        vimb_ = self._adjust_end_time(vimb, "Время окончания")

        df["overlap"] = ""
        df["Время выхода VIMB"] = ""
        df["Время окончания VIMB"] = ""

        # Через расчет процента перекрытия понимаем, нужная это программа или нет
        for i in range(len(df)):
            name = df.iloc[i]["Название программы"]
            start_palomars = df.iloc[i]["Время выхода"]
            h_start_P, m, s = map(int, start_palomars.split(":"))

            end_palomars = df.iloc[i]["Время окончания"]
            h_end_P, m, s = map(int, end_palomars.split(":"))

            in_vimb = vimb_[vimb_["Название программы"] == name].reset_index(drop=True)

            # print(name, len(in_vimb))

            if len(in_vimb) != 0:

                for j in range(len(in_vimb)):
                    start_vimb = in_vimb.iloc[j]["Время выхода"]
                    h_start_V, m, s = map(int, start_vimb.split(":"))

                    end_vimb = in_vimb.iloc[j]["Время окончания"]
                    h_end_V, m, s = map(int, end_vimb.split(":"))

                    if (
                        h_start_P == h_start_V
                        and h_end_P == h_end_V
                        or (h_start_V - h_start_P) == 1
                        and (h_end_V - h_end_P) == 1
                        or h_start_P == h_start_V
                        and (h_end_V - h_end_P) == 1
                        or (h_start_V - h_start_P) == 1
                        and h_end_P == h_end_V
                    ):

                        # print(name, h_start_P, h_end_P, h_start_V, h_end_V)
                        overlap = TVScheduleProcessor.find_time_overlaps(
                            start_palomars, end_palomars, start_vimb, end_vimb
                        )
                        df.at[i, "overlap"] = overlap
                        df.at[i, "Время выхода VIMB"] = start_vimb
                        df.at[i, "Время окончания VIMB"] = end_vimb

                    else:
                        continue
            else:
                continue

        with_overlap = df[df["overlap"] == True].reset_index(drop=True)
        res_overlap = with_overlap[
            [
                "Дата",
                "Название программы",
                "Время выхода VIMB",
                "Время окончания VIMB",
                "Share",
            ]
        ]
        res_overlap.rename(
            columns={
                "Время выхода VIMB": "Время выхода",
                "Время окончания VIMB": "Время окончания",
            },
            inplace=True,
        )
        return res_overlap

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
        share_sum_init = np.sum(list(palomars["Share"]))
        # Инициализация
        self.vimb_init = vimb.copy()
        self.palomars_init = palomars.copy()
        self.stats = {}

        if verbose:
            print(
                f"Исходные данные: VIMB = {len(self.vimb_init)}, Palomars = {len(self.palomars_init)}"
            )

        # Список для хранения найденных программ
        all_matches = []

        ############################ ШАГ 1: Обработка мелких программ, которые легко сджойнить. Округляем Palomars до 5 мин ############################
        if verbose:
            print(
                color.BOLD
                + color.RED
                + "ШАГ 1: Обработка мелких программ, которые легко сджойнить"
                + color.END,
                sep="\n",
                end="\n",
            )

        vimb_10min = self._prepare_dataframe(vimb, 10, False)
        pal_5min = self._prepare_dataframe(palomars, 5, True)

        matches, vimb_remaining, pal_remaining = self._process_schedule_step(
            pal_5min, vimb_10min, 5, "5_minutes"
        )

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(matches, "Время окончания"))

        found_programs_after_1 = pd.concat(all_matches).reset_index(drop=True)

        # Удаление найденных длинных программ
        vimb_after_1 = self._remove_found_programs(
            self._adjust_end_time(vimb, "Время окончания"), found_programs_after_1
        ).reset_index(drop=True)
        vimb_clean_after_1 = self._extract_core_info(vimb_after_1, False)

        if verbose:
            print(color.BOLD + "НАЙДЕННЫЕ ПРОГРАММЫ:" + color.END, sep="\n", end="\n")
            print(found_programs_after_1)
            print("#" * 120)
            print(
                color.BOLD + color.BLUE + "ВИМБ, который осталось найти" + color.END,
                sep="\n",
                end="\n",
            )
            print(vimb_clean_after_1)
            print("#" * 120)

        ############################ ШАГ 2: Обработка длинных программ с разбивкой по часам (округление 10 минут) ############################
        if verbose:
            print(
                color.BOLD
                + color.RED
                + "ШАГ 2: Обработка длинных программ с разбивкой по часам (округление 10 минут)"
                + color.END,
                sep="\n",
                end="\n",
            )

        # Работаем с исходным датафреймом palomars
        pal_10min = self._prepare_dataframe(palomars, 10, True)

        # Поиск длинных программ, которые шли в течение 1 часа
        hour_programs = TVScheduleProcessor.group_broadcasts_by_hours(pal_10min)

        ### НОВЫЙ КУСОК

        # ДОБАВЛЯЕМ ФИЛЬТРАЦИЮ: оставляем только программы длительностью от 50 до 60 минут
        hour_programs_filtered = hour_programs.copy()

        # Функция для вычисления длительности в минутах
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

        # Добавляем столбец с длительностью в минутах
        hour_programs_filtered["Длительность_минуты"] = hour_programs_filtered.apply(
            calculate_duration_minutes, axis=1
        )

        # Фильтруем: только программы длительностью от 50 до 60 минут включительно
        hour_programs_filtered = hour_programs_filtered[
            (hour_programs_filtered["Длительность_минуты"] >= 50)
            & (hour_programs_filtered["Длительность_минуты"] <= 60)
        ].drop(columns=["Длительность_минуты"])

        ### КОНЕЦ НОВОГО КУСКА
        hour_matches = pd.merge(
            self._adjust_end_time(vimb_clean_after_1),
            self._adjust_end_time(hour_programs_filtered),
            on=["Дата", "Название программы", "Время выхода", "Время окончания"],
            how="inner",
        )

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(hour_matches, "Время окончания"))

        found_programs_after = pd.concat(all_matches).reset_index(drop=True)

        # Удаление найденных длинных программ
        vimb_after = self._remove_found_programs(
            self._adjust_end_time(vimb_clean_after_1, "Время окончания"),
            found_programs_after,
        ).reset_index(drop=True)

        vimb_clean_after = self._extract_core_info(vimb_after, False)

        hour_overlap = self.broadcasts_overlaping(
            self._adjust_end_time(hour_programs_filtered),
            self._adjust_end_time(vimb_clean_after),
            only_hour_programs=True,
        )

        hour_matches_ = pd.merge(
            self._adjust_end_time(vimb_clean_after_1),
            self._adjust_end_time(hour_overlap),
            on=["Дата", "Название программы", "Время выхода", "Время окончания"],
            how="inner",
        )
        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(hour_matches_, "Время окончания"))

        found_programs_after_2 = pd.concat(all_matches).reset_index(drop=True)

        # Удаление найденных длинных программ
        vimb_after_2 = self._remove_found_programs(
            self._adjust_end_time(vimb_clean_after_1, "Время окончания"),
            found_programs_after_2,
        ).reset_index(drop=True)

        vimb_clean_after_2 = self._extract_core_info(vimb_after_2, False)

        if verbose:
            print(color.BOLD + "НАЙДЕННЫЕ ПРОГРАММЫ:" + color.END, sep="\n", end="\n")
            print(found_programs_after_2)
            print("#" * 120)

        ############################ ШАГ 1*: Обработка мелких программ, которые легко сджойнить. Округляем Palomars до 10 мин ############################
        if verbose:
            print(
                color.BOLD
                + color.RED
                + "ШАГ 1*: Обработка мелких программ, которые легко сджойнить"
                + color.END,
                sep="\n",
                end="\n",
            )

        vimb_10min = self._prepare_dataframe(vimb_clean_after_2, 10, False)
        pal_10min = self._prepare_dataframe(palomars, 10, True)

        matches_10min, vimb_remaining, pal_remaining = self._process_schedule_step(
            pal_5min, vimb_10min, 10, "10_minutes"
        )

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(matches_10min, "Время окончания"))

        found_programs_after_3 = pd.concat(all_matches).reset_index(drop=True)

        # Удаление найденных длинных программ
        vimb_after_3 = self._remove_found_programs(
            self._adjust_end_time(vimb_clean_after_2, "Время окончания"),
            found_programs_after_3,
        ).reset_index(drop=True)
        vimb_clean_after_3 = self._extract_core_info(vimb_after_3, False)

        if verbose:
            print(color.BOLD + "НАЙДЕННЫЕ ПРОГРАММЫ:" + color.END, sep="\n", end="\n")
            print(found_programs_after_3)
            print("#" * 120)
            print(
                color.BOLD + color.BLUE + "ВИМБ, который осталось найти" + color.END,
                sep="\n",
                end="\n",
            )
            print(vimb_clean_after_3)
            print("#" * 120)

        ############################ ШАГ 3: Обработка длинных программ (округление 10 минут) ############################
        if verbose:
            print(
                color.BOLD
                + color.RED
                + "ШАГ 3: Обработка длинных программ (округление 10 минут)"
                + color.END,
                sep="\n",
                end="\n",
            )

        vimb_10min = self._prepare_dataframe(vimb_clean_after_3, 10, False)
        pal_10min = self._prepare_dataframe(palomars, 10, True)

        # Поиск длинных программ, которые шли более, чем 1 час
        long_programs = TVScheduleProcessor.join_broadcasts(pal_10min)

        long_matches = pd.merge(
            self._adjust_end_time(vimb_10min, "Время окончания"),
            self._adjust_end_time(long_programs, "Время окончания"),
            on=["Дата", "Название программы", "Время выхода", "Время окончания"],
            how="inner",
        )

        # Переименование колонок
        long_matches_clean = self._extract_core_info(long_matches, True)

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(long_matches_clean, "Время окончания"))

        found_programs_after_4 = pd.concat(all_matches).reset_index(drop=True)

        # Удаление найденных длинных программ
        vimb_after_4 = self._remove_found_programs(
            self._adjust_end_time(vimb_clean_after_3, "Время окончания"),
            found_programs_after_4,
        ).reset_index(drop=True)
        vimb_clean_after_4 = self._extract_core_info(vimb_after_4, False)

        if verbose:
            print(color.BOLD + "НАЙДЕННЫЕ ПРОГРАММЫ:" + color.END, sep="\n", end="\n")
            print(found_programs_after_4)
            print("#" * 120)
            print(
                color.BOLD + color.BLUE + "ВИМБ, который осталось найти" + color.END,
                sep="\n",
                end="\n",
            )
            print(vimb_clean_after_4)
            print("#" * 120)

        ############################ ШАГ 4: Обработка длинных программ с overlaping ###########################
        if verbose:
            print(
                color.BOLD
                + color.RED
                + "ШАГ 4: Обработка длинных программ с overlaping"
                + color.END,
                sep="\n",
                end="\n",
            )

        # Заменяем значения начиная со второго
        for i in range(1, len(palomars)):
            palomars.loc[i, "Время выхода"] = palomars.loc[i - 1, "Время окончания"]

        # Отбираем программы, которые шли больше 1 часа и соединяем
        res = TVScheduleProcessor.join_broadcasts(palomars)

        res_overlap = self.broadcasts_overlaping_OLD(res, vimb_clean_after_4)

        res_overlap["Дата"] = pd.to_datetime(res_overlap["Дата"])
        res_overlap["Дата"] = res_overlap["Дата"].dt.strftime("%Y-%m-%d")

        # Добавляем в список найденных программ
        all_matches.append(self._adjust_end_time(res_overlap, "Время окончания"))

        found_programs_after_5 = pd.concat(all_matches).reset_index(drop=True)

        # Удаление найденных длинных программ
        vimb_after_5 = self._remove_found_programs(
            self._adjust_end_time(vimb_clean_after_4, "Время окончания"),
            self._adjust_end_time(found_programs_after_5, "Время окончания"),
        ).reset_index(drop=True)
        vimb_clean_after_5 = self._extract_core_info(vimb_after_5, False)

        if verbose:
            print(color.BOLD + "НАЙДЕННЫЕ ПРОГРАММЫ:" + color.END, sep="\n", end="\n")
            print(found_programs_after_5)
            print("#" * 120)

            if len(vimb_clean_after_5) != 0:
                print(
                    color.BOLD
                    + color.BLUE
                    + "ВИМБ, который не удалось найти"
                    + color.END,
                    sep="\n",
                    end="\n",
                )
                print(vimb_clean_after_4)
                print("#" * 120)

        ############################ ШАГ 5: Дополнительный шаг, если после остальных остались ненайденные ###########################
        if len(vimb_clean_after_5) != 0:

            if verbose:
                print(
                    color.BOLD
                    + color.RED
                    + "ШАГ 5: Обработка длинных программ с разбивкой по часам (округление 10 минут)"
                    + color.END,
                    sep="\n",
                    end="\n",
                )

            # Работаем с исходным датафреймом palomars
            pal_10min = self._prepare_dataframe(palomars, 10, True)

            # Поиск длинных программ, которые шли в течение 1 часа
            hour_programs = TVScheduleProcessor.group_broadcasts_by_hours(pal_10min)

            hour_overlap = self.broadcasts_overlaping(
                self._adjust_end_time(hour_programs),
                vimb_clean_after_5,
                only_hour_programs=True,
            )

            hour_matches = pd.merge(
                self._adjust_end_time(vimb_clean_after_5),
                hour_overlap,
                on=["Дата", "Название программы", "Время выхода", "Время окончания"],
                how="inner",
            )
            # Добавляем в список найденных программ
            all_matches.append(self._adjust_end_time(hour_matches, "Время окончания"))

            found_programs_after_6 = pd.concat(all_matches).reset_index(drop=True)

            # Удаление найденных длинных программ
            vimb_after_6 = self._remove_found_programs(
                self._adjust_end_time(vimb_clean_after_5, "Время окончания"),
                found_programs_after_6,
            ).reset_index(drop=True)

            vimb_clean_after_6 = self._extract_core_info(vimb_after_6, False)

            if verbose:
                print(
                    color.BOLD + "НАЙДЕННЫЕ ПРОГРАММЫ:" + color.END, sep="\n", end="\n"
                )
                print(found_programs_after_6)
                print("#" * 120)

                if len(vimb_clean_after_6) != 0:
                    print(
                        color.BOLD
                        + color.BLUE
                        + "ВИМБ, который не удалось найти"
                        + color.END,
                        sep="\n",
                        end="\n",
                    )
                    print(vimb_clean_after_6)
                    print("#" * 120)

        ############################ ШАГ 6: ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ ############################

        found_programs = pd.concat(all_matches).reset_index(drop=True)

        # Проверяем, что в таблице VIMB действительно все найдено
        merged = pd.merge(
            self._adjust_end_time(vimb),
            found_programs,
            on=["Дата", "Название программы", "Время выхода", "Время окончания"],
            how="left",
        )

        # Проверяем, что в таблице действительно все найдено

        if len(merged) != len(vimb):
            print("Нужен дополнительный поиск!")

        if share_sum_init != np.sum(list(found_programs["Share"])):
            share_sum_curr = np.sum(list(found_programs["Share"]))
            print(
                f"Обнаружено несовпадение суммарной доли по дню! Целевой показатель {np.round(share_sum_init, 4)}, а по итогу вышло {np.round(share_sum_curr, 4)}."
            )

        return merged

    #        # Работаем с исходным датафреймом palomars
    #        pal_5min = self._prepare_dataframe(palomars, 5, True)
    #
    #        # Поиск длинных программ, которые шли в течение 1 часа
    #        hour_programs = TVScheduleProcessor.group_broadcasts_by_hours(pal_5min)
    #        hour_programs = self._adjust_end_time(hour_programs)
    #
    #        hour_matches = pd.merge(
    #            self._adjust_end_time(vimb_cleaned), hour_programs,
    #            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
    #            how = 'inner'
    #        )
    #        # Добавляем в список найденных программ
    #        all_matches.append(hour_matches)
    #
    #        # Удаление найденных длинных программ
    #        pal_remaining_new = self._remove_found_programs(pal_5min, hour_matches).reset_index(drop = True)
    #
    #        # Переименование колонок
    #        pal_clean_new = self._extract_core_info(pal_remaining_new, True)
    #
    #        vimb_remaining_new = self._remove_found_programs(vimb_cleaned, hour_matches).reset_index(drop = True)
    #
    #
    #
    #
    #
    #        ############################ Шаг 3: Поиск по недлительным программам. Поочереди округляем слоты Palomars ############################
    #        # Для анализа берем исходный датафрейм Palomars и последний преобразованный датафрейм VIMB
    #        # Последовательная обработка с разным округлением
    #        rounding_steps = [
    #            (1, "1_minute", palomars, vimb_remaining_new),
    #            (5, "5_minutes", None, None),
    #            (10, "10_minutes", None, None)
    #        ]
    #
    #        current_pal = palomars
    #        current_vimb = vimb_remaining_new
    #        for minutes, step_name, pal_input, vimb_input in rounding_steps:
    #            if verbose:
    #                print(f'\nШаг {step_name}: Округление {minutes} минут')
    #
    #            # Используем переданные данные или результаты предыдущего шага
    #            pal_to_process = pal_input if pal_input is not None else current_pal
    #            vimb_to_process = vimb_input if vimb_input is not None else current_vimb
    #
    #            # Ищем совпадения
    #            matches, vimb_remaining, pal_remaining = self._process_schedule_step(
    #                pal_to_process, vimb_to_process, minutes, step_name
    #            )
    #
    #            # Добавляем в список найденных программ
    #            all_matches.append(matches)
    #
    #            # Подготовка данных для следующего шага
    #            current_pal = self._extract_core_info(pal_remaining, True)
    #            current_vimb = self._extract_core_info(vimb_remaining, False)
    #
    #            if minutes < 10:  # Для следующих шагов добавляем исходные метки
    #                current_vimb = self._add_original_time_columns(current_vimb)
    #
    #        ############################ ДОПОЛНИТЕЛЬНЫЙ ШАГ: Повторное округление до 10 минут ############################
    #        if verbose:
    #            print('\nДополнительный шаг: Повторное округление до 10 минут')
    #
    #        # Снова округляем текущие данные до 10 минут
    #        vimb_round_to_10 = self._prepare_dataframe(current_vimb, 10, False)
    #        pal_round_to_10 = self._prepare_dataframe(current_pal, 10, True)
    #
    #        # Ищем совпадения
    #        additional_matches = pd.merge(
    #            vimb_round_to_10,
    #            pal_round_to_10,
    #            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
    #            how = 'inner'
    #        )[['Дата', 'Название программы', 'Время выхода', 'Время окончания', 'Share']]
    #
    #        self.stats['additional_10min_matches'] = len(additional_matches)
    #
    #        # Добавляем в список найденных программ
    #        all_matches.append(additional_matches)
    #
    #        # Удаление найденных программ
    #        vimb_deleted = self._remove_found_programs(vimb_round_to_10, additional_matches).reset_index(drop = True)
    #        pal_deleted = self._remove_found_programs(pal_round_to_10, additional_matches).reset_index(drop = True)
    #
    #        pal_deleted['Дата'] = pd.to_datetime(pal_deleted['Дата'], errors = 'coerce')
    #        pal_deleted['Дата'] = pal_deleted['Дата'].dt.strftime('%Y-%m-%d')
    #
    #        # Подготовка данных для следующего шага
    #        final_pal_remaining = self._extract_core_info(pal_deleted, True)
    #        final_vimb_remaining = self._extract_core_info(vimb_deleted, False)
    #
    #        ############################ ПОСЛЕДНИЙ ШАГ: Поиск через перекрытия ############################
    #        found_programs = pd.concat(all_matches).reset_index(drop = True)
    #
    #        found_programs_unique = found_programs.drop_duplicates(
    #            subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
    #        ).reset_index(drop = True)
    #
    #        print(final_vimb_remaining)
    #        print('#' * 120)
    #        print(found_programs_unique)
    #
    #        vimb_last = self._remove_found_programs(self._adjust_end_time(final_vimb_remaining, 'Время окончания'), found_programs_unique).reset_index(drop = True)
    #
    #        print('#' * 120)
    #        print(vimb_last)
    #        print('#' * 120)
    #
    #        if len(found_programs_unique) != len(self.vimb_init):
    #            #final_vimb_remaining = self._adjust_end_time(final_vimb_remaining, 'Время окончания')
    #            final_pal_remaining = self._adjust_end_time(final_pal_remaining, 'Время окончания')
    #            matches_df, unmatched_vimb_df, palomar_df = TVScheduleProcessor.match_programs_with_tolerance(vimb_last, final_pal_remaining, consolidate_palomars = True)
    #
    #            print(matches_df)
    #
    #            matched = matches_df[['Дата', 'Название_VIMB', 'Время_начала_VIMB', 'Время_окончания_VIMB', 'Share']]
    #            matched.rename(columns = {
    #                                'Название_VIMB': 'Название программы',
    #                                'Время_начала_VIMB': 'Время выхода',
    #                                'Время_окончания_VIMB': 'Время окончания'
    #                            },
    #                           inplace = True)
    #            # Добавляем в список найденных программ
    #            all_matches.append(matched)
    #
    #            found_programs = pd.concat(all_matches).reset_index(drop = True)
    #
    #            found_programs_unique = found_programs.drop_duplicates(
    #                subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
    #            ).reset_index(drop = True)
    #
    #        ############################ Объединение всех найденных программ ############################
    #        #found_programs = pd.concat(all_matches).reset_index(drop = True)
    #
    #        # Финальное сопоставление с исходными данными
    #        vimb_adjusted = self._adjust_end_time(self.vimb_init, 'Время окончания')
    #        found_adjusted = self._adjust_end_time(found_programs_unique, 'Время окончания')
    #
    #        result = pd.merge(
    #            vimb_adjusted,
    #            found_adjusted,
    #            on = ['Дата', 'Название программы', 'Время выхода', 'Время окончания'],
    #            how = 'left'
    #        )
    #
    #        # Расчёт статистики
    #        final_stats = self._calculate_final_stats(final_vimb_remaining, found_programs_unique)
    #
    #        # Вывод статистики
    #        if verbose:
    #            self.print_statistics(final_stats)
    #            print(f'\nОсталось ненайденных программ VIMB: {len(final_vimb_remaining)}')
    #            print(f'Осталось ненайденных программ Palomar: {len(final_pal_remaining)}')
    #
    #        #found_programs_unique = found_programs.drop_duplicates(
    #        #    subset = ['Дата', 'Название программы', 'Время выхода', 'Время окончания']
    #        #).reset_index(drop = True)
    #
    #        if len(found_programs_unique) != len(self.vimb_init):
    #            print(f'ВНИМАНИЕ: Размер результата ({len(found_programs_unique)}) не совпадает с исходным VIMB ({len(self.vimb_init)})')
    #
    #        if share_sum_init != np.sum(list(found_programs_unique['Share'])):
    #            share_sum_curr = np.sum(list(found_programs_unique['Share']))
    #            print(f'Обнаружено несовпадение суммарной доли по дню! Целевой показатель {np.round(share_sum_init, 2)}, а по итогу вышло {np.round(share_sum_curr, 2)}.')
    #
    #        return result, final_stats, found_programs_unique, final_vimb_remaining, final_pal_remaining

    def _calculate_final_stats(
        self, vimb_final: pd.DataFrame, found_programs: pd.DataFrame
    ) -> Dict:
        """
        Расчёт финальной статистики по найденным и ненайденным программам
        """
        # Общее количество программ в исходных данных
        total_vimb_programs = len(self.vimb_init)

        # Количество найденных программ (уникальных записей)
        found_count = len(
            found_programs.drop_duplicates(
                subset=["Дата", "Название программы", "Время выхода", "Время окончания"]
            )
        )

        # Количество ненайденных программ
        not_found_count = total_vimb_programs - found_count

        # Процент найденных
        found_percentage = (
            (found_count / total_vimb_programs * 100) if total_vimb_programs > 0 else 0
        )

        # Статистика по шагам поиска
        step_stats = {}
        for key, value in self.stats.items():
            if "_matches" in key:
                step_name = key.replace("_matches", "")
                step_stats[step_name] = value

        return {
            "total_vimb_programs": total_vimb_programs,
            "found_programs": found_count,
            "not_found_programs": not_found_count,
            "found_percentage": round(found_percentage, 2),
            "step_by_step_matches": step_stats,
            "remaining_vimb_after_all_steps": len(vimb_final),
        }

    def print_statistics(self, stats: Dict) -> None:
        """
        Вывод статистики в удобном формате
        """
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ОБРАБОТКИ ТЕЛЕПРОГРАММ")
        print("=" * 60)
        print(f"Всего программ в VIMB: {stats['total_vimb_programs']}")
        print(f"Найдено программ: {stats['found_programs']}")
        print(f"Не найдено программ: {stats['not_found_programs']}")
        print(f"Процент найденных: {stats['found_percentage']}%")
        print("-" * 60)
        print("Пошаговая статистика:")
        for step, count in stats["step_by_step_matches"].items():
            print(f"  {step}: {count} совпадений")
        print("-" * 60)
        print(
            f"Осталось ненайденных после всех шагов: {stats['remaining_vimb_after_all_steps']}"
        )
        print("=" * 60 + "\n")
