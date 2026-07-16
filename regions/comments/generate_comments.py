import pandas as pd
import pymorphy3 as pmrph
import docx
import numpy as np
from pathlib import Path
from OMA_tools.io_data.dates import Dates_Operations
from OMA_tools.io_data.operations import Dict_Operations, File
from OMA_tools.io_data.colors import *
import os
import re


class Generate_Comments:
    def __init__(self, year_num, month_num, day_num):
        self.year_num = year_num
        self.month_num = month_num
        self.day_num = day_num
        self.month_names = [
            'Январь', 'Февраль', 'Март',
            'Апрель', 'Май', 'Июнь',
            'Июль', 'Август', 'Сентябрь',
            'Октябрь', 'Ноябрь', 'Декабрь'
        ]
        self.target_to_sheet = {
            'ВСЕ 18+': 'All 18+',
            'ВСЕ 14-59': 'All 14-59',
            'ВСЕ 10-45': 'All 10-45',
            'ВСЕ 14-44': 'All 14-44',
            'ВСЕ 14-54': 'All 14-54',
            'ВСЕ 25-49': 'All 25-49',
            'ВСЕ 25-54': 'All 25-54',
            'ВСЕ 4-45': 'All 4-45',
            'ВСЕ 6-54': 'All 6-54',
            'Ж 14-44': 'W 14-44',
            'Ж 25-59': 'W 25-59',
        }

    @staticmethod
    def find_file_to_analysis(path: str, file_format: str = 'xlsx'):
        """
            Вспомогательный метод, который ищет в указанной папке самый свежий файл 'Для комментариев' формата .xlsx
        """
        # Укажите путь к вашей папке
        folder_path = Path(path)  # Замените на ваш путь

        # Ищем все .xlsx файлы и выбираем самый свежий по времени изменения
        try:
            latest_file = max(folder_path.glob(f"*.{file_format}"), key=os.path.getmtime)
            print(Color.BOLD + Color.VIOLET + f"Найден свежий файл: {latest_file.name}" + Color.END)
            
        except ValueError:
            print(f"В папке нет файлов .{file_format}")
        except FileNotFoundError:
            print(f"Папка не найдена: {folder_path}")
        
        return latest_file


    def _get_previous_month_info(self, base_month_idx, base_year, months_back):
        total_months = base_year * 12 + base_month_idx - months_back
        year = total_months // 12
        month_idx = total_months % 12
        return self.month_names[month_idx], year

    def _get_next_month_info(self):
        total_months = self.year_num * 12 + (self.month_num - 1) + 1
        year = total_months // 12
        month_idx = total_months % 12
        return self.month_names[month_idx], year

    def load_historical_data(self, filepath):
        historical_data = {}
        xl = pd.ExcelFile(filepath)
        for sheet_name in xl.sheet_names:
            if sheet_name in self.target_to_sheet.values():
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                melted_dfs = []
                for col in df.columns[1:]:
                    match = re.match(r'(.+?)\s*\((.+?)\)', col)
                    if match:
                        channel = match.group(1).strip()
                        city = match.group(2).strip()
                        temp_df = pd.DataFrame({
                            'Date': df['Date'],
                            'Телеканал': channel,
                            'Город': city,
                            'Значение': df[col].values
                        })
                        melted_dfs.append(temp_df)
                if melted_dfs:
                    sheet_df = pd.concat(melted_dfs, ignore_index=True)
                    sheet_df['Год'] = sheet_df['Date'].apply(lambda x: int(x.split()[-1]))
                    sheet_df['Месяц'] = sheet_df['Date'].apply(lambda x: x.split()[0])
                    for ta, sheet in self.target_to_sheet.items():
                        if sheet == sheet_name:
                            sheet_df['БЦА'] = ta
                            historical_data[ta] = sheet_df
                            break
        return historical_data

    def get_historical_value(self, historical_data, channel, city, target_audience, month_name, year):
        if target_audience not in historical_data:
            return None
        df = historical_data[target_audience]
        channel_norm = channel.strip().upper()
        city_norm = city.strip().upper()
        mask = (
            (df['Телеканал'].str.upper() == channel_norm) &
            (df['Город'].str.upper() == city_norm) &
            (df['Месяц'] == month_name) &
            (df['Год'] == year)
        )
        result = df[mask]['Значение'].values
        return result[0] if len(result) > 0 else None

    @staticmethod
    def get_right_city_name(city_name):
        if city_name in ('Ростове-на-Дону', 'ростове-на-дону', 'Ростове-на-дону', 'Ростов-На-Дону'):
            return 'Ростов-на-Дону', 'Ростове-на-Дону'
        elif city_name in ('Санкт-Петербурге', 'санкт-петербурге', 'Санкт-петербурге', 'Санкт-Петербург'):
            return 'Санкт-Петербург', 'Санкт-Петербурге'
        elif city_name in ('Нижний новгороде', 'нижний новгороде', 'Нижний новгороде', 'Нижний Новгород'):
            return 'Нижний Новгород', 'Нижнем Новгороде'
        elif city_name in ('Великом Новгороде', 'великом новгороде', 'Великом новгороде', 'Великом Новгород'):
            return 'Великий Новгород', 'Великом Новгороде'
        else:
            return city_name, city_name

    def get_reason_channels__and__res_data(self, data, data_api, historical_data,
                                          column_fact_month, column_last_14_days,
                                          column_prev_14_days, column_prev_to_fact_month,
                                          cond_grp, cond_share, cond_kus, cond_ttv):
        reason_channels = {
            'kus+': [], 'ttv+': [], 'share': [], 'share4': [], 'kus-': [], 'ttv-': [],
        }

        res_data_1 = pd.DataFrame(data={'Телеканал': [], 'Город': [], 'Было': [], 'Стало': [], 'Изменение': []})
        res_data_2 = pd.DataFrame(data={'Телеканал': [], 'Город': [], 'Было': [], 'Стало': [], 'Изменение': []})
        res_data_3 = pd.DataFrame(data={'Телеканал': [], 'Город': [], 'Было': [], 'Стало': [], 'Изменение': []})
        res_data_4 = pd.DataFrame(data={'Телеканал': [], 'Город': [], 'Было': [], 'Стало': [], 'Изменение': []})

        data_grp = data[data['Атрибут'] == 'GRP']
        data_volume = data[data['Атрибут'] == 'Объем']
        data_share = data[data['Атрибут'] == 'Share']
        data_kus = data[data['Атрибут'] == 'КУС']
        data_ttv = data[data['Атрибут'] == 'TTV']

        # Месяцы для отбора городов
        grp_months = []
        if self.day_num >= 8:
            cur_month_name = self.month_names[self.month_num - 1]
            cur_year = self.year_num
            grp_months.append((cur_month_name, cur_year))
            next_month_name, next_year = self._get_next_month_info()
            grp_months.append((next_month_name, next_year))
        else:
            cur_month_name = self.month_names[self.month_num - 1]
            cur_year = self.year_num
            grp_months.append((cur_month_name, cur_year))

        # Месяцы для анализа доли
        if self.day_num < 8:
            base_month_idx = self.month_num - 2
            base_year = self.year_num
            if base_month_idx < 0:
                base_month_idx += 12
                base_year -= 1
        else:
            base_month_idx = self.month_num - 1
            base_year = self.year_num

        base_month_name = self.month_names[base_month_idx]

        prev_month_name, prev_year = self._get_previous_month_info(base_month_idx, base_year, 1)
        prev_2_month_name, prev_2_year = self._get_previous_month_info(base_month_idx, base_year, 2)
        prev_3_month_name, prev_3_year = self._get_previous_month_info(base_month_idx, base_year, 3)
        prev_4_month_name, prev_4_year = self._get_previous_month_info(base_month_idx, base_year, 4)

        self._prev_month_name = prev_month_name
        self._prev_2_month_name = prev_2_month_name

        not_found_cities_channels = set()
        processed_cities = set()

        # Отбор городов по GRP
        eligible_cities = set()
        city_month_direction = {}

        for grp_month_name, grp_year in grp_months:
            possible_keys = [f'{grp_month_name}.2', f'{grp_month_name}']
            found_key = None
            for key in possible_keys:
                if key in data_grp.columns:
                    found_key = key
                    break
            if found_key is None:
                continue

            for irow, row in data_grp.iterrows():
                grp_change = row[found_key]
                if pd.isna(grp_change) or abs(grp_change) < cond_grp:
                    continue

                vol_row = data_volume[(data_volume['Телеканал'] == row['Телеканал']) & (data_volume['Город'] == row['Город'])]
                if len(vol_row) == 0:
                    continue
                vol_change = vol_row.iloc[0][found_key] if found_key in vol_row.iloc[0].index else None
                if vol_change is None or pd.isna(vol_change) or abs(grp_change - vol_change) < cond_grp:
                    continue

                city_key = self.get_right_city_name(city_name=row['Город'].lower().title())[0]
                city_channel_key = (row['Телеканал'], city_key)
                eligible_cities.add(city_channel_key)
                if city_channel_key not in city_month_direction:
                    city_month_direction[city_channel_key] = (found_key, grp_change)

        for irow, row in data_grp.iterrows():
            city_key = self.get_right_city_name(city_name=row['Город'].lower().title())[0]
            city_channel_key = (row['Телеканал'], city_key)

            if city_channel_key not in eligible_cities:
                continue

            row_vol = data_volume[(data_volume['Телеканал'] == row['Телеканал']) & (data_volume['Город'] == row['Город'])]
            if len(row_vol) == 0:
                continue
            row_vol = row_vol.iloc[0]

            api_mask = (data_api['Телеканал'] == row['Телеканал']) & (data_api['Город'] == row['Город'])
            if len(data_api[api_mask]) == 0:
                continue
            row_share_api = data_api[api_mask].iloc[0]
            target_audience = row_share_api['БЦА']

            month_key, grp_change_direction = city_month_direction[city_channel_key]

            if city_channel_key in processed_cities:
                continue

            row_share = data_share[(data_share['Телеканал'] == row['Телеканал']) & (data_share['Город'] == row['Город'])]
            if len(row_share) == 0:
                continue
            row_share = row_share.iloc[0]

            def get_share(month_name, year):
                if year < self.year_num:
                    val = self.get_historical_value(
                        historical_data, row['Телеканал'], row['Город'],
                        target_audience, month_name, year
                    )
                    return val
                else:
                    val = row_share.get(f'{month_name}.1')
                    if val is not None:
                        return val
                    val_hist = self.get_historical_value(
                        historical_data, row['Телеканал'], row['Город'],
                        target_audience, month_name, year
                    )
                    return val_hist

            share_prev = get_share(prev_month_name, prev_year)
            share_prev2 = get_share(prev_2_month_name, prev_2_year)
            share_prev3 = get_share(prev_3_month_name, prev_3_year)
            share_prev4 = get_share(prev_4_month_name, prev_4_year)

            row_kus = data_kus[(data_kus['Телеканал'] == row['Телеканал']) & (data_kus['Город'] == row['Город'])]
            row_ttv = data_ttv[(data_ttv['Телеканал'] == row['Телеканал']) & (data_ttv['Город'] == row['Город'])]

            if len(row_kus) > 0:
                row_kus = row_kus.iloc[0]
            if len(row_ttv) > 0:
                row_ttv = row_ttv.iloc[0]

            variant_applied = False

            # Вариант 1
            if not variant_applied and share_prev is not None:
                share_ratio_1 = row_share_api[column_fact_month] / share_prev
                if (grp_change_direction * (share_ratio_1 - 1)) > 0 and abs(share_ratio_1 - 1) > cond_share:
                    res_data_1.loc[len(res_data_1.index)] = [
                        row['Телеканал'],
                        city_key,
                        share_prev,
                        row_share_api[column_fact_month],
                        (share_ratio_1 - 1) * 100
                    ]
                    reason_channels['share'].append([row['Телеканал'], row['Город']])
                    processed_cities.add(city_channel_key)
                    variant_applied = True

            # Вариант 2
            if not variant_applied and share_prev2 is not None and share_prev is not None:
                avg_2 = (share_prev2 + share_prev) / 2.0
                share_ratio_2 = row_share_api[column_fact_month] / avg_2
                if (grp_change_direction * (share_ratio_2 - 1)) > 0 and abs(share_ratio_2 - 1) > cond_share:
                    res_data_2.loc[len(res_data_2.index)] = [
                        row['Телеканал'],
                        city_key,
                        avg_2,
                        row_share_api[column_fact_month],
                        (share_ratio_2 - 1) * 100
                    ]
                    reason_channels['share'].append([row['Телеканал'], row['Город']])
                    processed_cities.add(city_channel_key)
                    variant_applied = True

            # Вариант 3
            if not variant_applied:
                if column_prev_14_days in row_share_api.index and column_last_14_days in row_share_api.index:
                    share_ratio_14 = row_share_api[column_last_14_days] / row_share_api[column_prev_14_days]
                    if (grp_change_direction * (share_ratio_14 - 1)) > 0 and abs(share_ratio_14 - 1) > cond_share:
                        res_data_3.loc[len(res_data_3.index)] = [
                            row['Телеканал'],
                            city_key,
                            row_share_api[column_prev_14_days],
                            row_share_api[column_last_14_days],
                            (share_ratio_14 - 1) * 100
                        ]
                        reason_channels['share'].append([row['Телеканал'], row['Город']])
                        processed_cities.add(city_channel_key)
                        variant_applied = True

            # Вариант 4
            if not variant_applied and share_prev3 is not None and share_prev2 is not None:
                if column_prev_to_fact_month in row_share_api.index:
                    avg_34 = (share_prev3 + share_prev2) / 2.0
                    share_ratio_4 = row_share_api[column_prev_to_fact_month] / avg_34
                    if (grp_change_direction * (share_ratio_4 - 1)) > 0 and abs(share_ratio_4 - 1) > cond_share:
                        res_data_4.loc[len(res_data_4.index)] = [
                            row['Телеканал'],
                            city_key,
                            avg_34,
                            row_share_api[column_prev_to_fact_month],
                            (share_ratio_4 - 1) * 100
                        ]
                        reason_channels['share4'].append([row['Телеканал'], row['Город']])
                        processed_cities.add(city_channel_key)
                        variant_applied = True

            if not variant_applied:
                not_found_cities_channels.add((city_key, row['Телеканал']))

            # КУС и TTV
            if len(row_kus) > 0:
                kus_change = row_kus.get(month_key, 0)
                if abs(kus_change * 100) >= cond_kus:
                    if grp_change_direction > 0 and kus_change > 0:
                        reason_channels['kus+'].append([row['Телеканал'], row['Город']])
                    elif grp_change_direction < 0 and kus_change < 0:
                        reason_channels['kus-'].append([row['Телеканал'], row['Город']])

            if len(row_ttv) > 0:
                ttv_change = row_ttv.get(month_key, 0)
                if abs(ttv_change * 100) >= cond_ttv:
                    if grp_change_direction > 0 and ttv_change > 0:
                        reason_channels['ttv+'].append([row['Телеканал'], row['Город']])
                    elif grp_change_direction < 0 and ttv_change < 0:
                        reason_channels['ttv-'].append([row['Телеканал'], row['Город']])

        print('\033[1m' + 'Не удалось найти релевантный период для доли при объяснении следующих пар (город, канал):\n\n' + '\n'.join(list(map(lambda x: str(x[0]) + ', ' + str(x[1]), not_found_cities_channels))))
        return reason_channels, res_data_1, res_data_2, res_data_3, res_data_4

    @staticmethod
    def get_cities_formatted(dict_reasons):
        morph = pmrph.MorphAnalyzer()
        reason_channels_formatted = {k: [] for k in dict_reasons}
        for key, channels_cities in dict_reasons.items():
            for channel_city in channels_cities:
                channel_name = channel_city[0]
                city_name = morph.parse(channel_city[1].strip())[0]
                reason_channels_formatted[key].append([channel_name,
                                                       city_name.inflect({'loct'}).word.capitalize()])
        for key, values in reason_channels_formatted.items():
            for i in range(len(values)):
                values[i][1] = Generate_Comments.get_right_city_name(city_name=values[i][1])[1]
        return reason_channels_formatted

    @staticmethod
    def get_cities_and_reasons(reason_channels_formatted):
        cities = set()
        channels = set()
        for key, value in reason_channels_formatted.items():
            for channel_city in value:
                channels.add(channel_city[0])
                cities.add(channel_city[1])
        channel_reason_cities = {}
        for channel in channels:
            channel_reason_cities[channel] = {
                'ttv_kus+': set(), 'ttv+': set(), 'kus+': set(),
                'share': set(), 'share4': set(),
                'ttv_kus-': set(), 'ttv-': set(), 'kus-': set(),
            }
        for channel in channels:
            for city in cities:
                if [channel, city] in reason_channels_formatted['ttv+'] and [channel, city] in reason_channels_formatted['kus+']:
                    channel_reason_cities[channel]['ttv_kus+'].add(city)
                if [channel, city] in reason_channels_formatted['ttv-'] and [channel, city] in reason_channels_formatted['kus-']:
                    channel_reason_cities[channel]['ttv_kus-'].add(city)
                if [channel, city] in reason_channels_formatted['ttv+']:
                    channel_reason_cities[channel]['ttv+'].add(city)
                if [channel, city] in reason_channels_formatted['ttv-']:
                    channel_reason_cities[channel]['ttv-'].add(city)
                if [channel, city] in reason_channels_formatted['kus+']:
                    channel_reason_cities[channel]['kus+'].add(city)
                if [channel, city] in reason_channels_formatted['kus-']:
                    channel_reason_cities[channel]['kus-'].add(city)
                if [channel, city] in reason_channels_formatted['share']:
                    channel_reason_cities[channel]['share'].add(city)
                elif [channel, city] in reason_channels_formatted['share4']:
                    channel_reason_cities[channel]['share4'].add(city)
        return channel_reason_cities

    @staticmethod
    def remove_duplicate_elements_in_set(set_1, set_2):
        res_duplicates = set()
        res_unique = set()
        for i in set_1:
            if i in set_2:
                res_duplicates.add(i)
            else:
                res_unique.add(i)
        return res_duplicates, res_unique

    def get_explanations(self, channel_reason_cities: dict, sorted_keys: list,
                         channels_group_1, channels_group_2, channels_group_3):
        comment_header_1 = 'Изменение прогноза инвентаря в нижеперечисленных городах связано с динамикой доли канала в '
        comment_header_2 = 'Изменение прогноза инвентаря в нижеперечисленных городах связано со значительным изменением доли, начиная с '
        comment_start = ['Увеличение прогноза инвентаря в ', 'Уменьшение прогноза инвентаря в ']

        comment_ttv_kus_plus = [
            ' связано с ростом общего уровня телесмотрения, а также с ростом эффективности рекламной сетки.\n',
            ' обусловлено ростом общего уровня телесмотрения и ростом эффективности рекламной сетки.\n'
        ]
        comment_ttv_kus_minus = [
            ' связано с уменьшением общего уровня телесмотрения, а также с уменьшением эффективности рекламной сетки.\n',
            ' обусловлено уменьшением общего уровня телесмотрения и уменьшением эффективности рекламной сетки.\n'
        ]
        comment_kus_plus = [
            ' обусловлено ростом эффективности рекламной сетки.\n',
            ' дополнительно произошло за счет роста эффективности рекламной сетки.\n',
            ' произошло благодаря росту эффективности рекламной сетки.\n'
        ]
        comment_kus_minus = [
            ' связано с уменьшением эффективности рекламной сетки.\n',
            ' дополнительно произошло за счет снижения эффективности рекламной сетки.\n',
            ' произошло из-за снижения эффективности рекламной сетки.\n'
        ]
        comment_ttv_plus = [
            ' дополнительно связано с ростом общего уровня телесмотрения.\n',
            ' связано с высокими фактическими показателями телесмотрения. Тенденции проложены на будущий период.\n',
            ' обусловлено также ростом общего уровня телесмотрения.\n'
        ]
        comment_ttv_minus = [
            ' связано с низкими фактическими показателями телесмотрения. Тенденции проложены на будущий период.\n',
            ' обусловлено снижением общего уровня телесмотрения.\n',
            ' связано с уменьшением общего уровня телесмотрения.\n'
        ]

        channel_phrase = {}

        for channel, reason_cities in channel_reason_cities.items():
            if channel in channels_group_1:
                idx = 0
            elif channel in channels_group_2:
                idx = 1
            else:
                idx = 2

            phrase_kus_ttv_plus = {'cities': set(), 'phrase': ''}
            phrase_kus_ttv_minus = {'cities': set(), 'phrase': ''}
            phrase_kus_plus = {'cities': set(), 'phrase': ''}
            phrase_kus_minus = {'cities': set(), 'phrase': ''}
            phrase_ttv_plus = {'cities': set(), 'phrase': ''}
            phrase_ttv_minus = {'cities': set(), 'phrase': ''}
            phrase_share = {'cities': set(), 'phrase': ''}
            phrase_share4 = {'cities': set(), 'phrase': ''}
            phrase_kus_ttv_plus_4 = {'cities': set(), 'phrase': ''}
            phrase_kus_ttv_minus_4 = {'cities': set(), 'phrase': ''}
            phrase_kus_plus_4 = {'cities': set(), 'phrase': ''}
            phrase_kus_minus_4 = {'cities': set(), 'phrase': ''}
            phrase_ttv_plus_4 = {'cities': set(), 'phrase': ''}
            phrase_ttv_minus_4 = {'cities': set(), 'phrase': ''}

            res_reason_cities_unique = {}
            res_reason_cities_duplicates = {}
            for reason, city in reason_cities.items():
                res_reason_cities_duplicates[reason] = self.remove_duplicate_elements_in_set(
                    set_1=reason_cities[reason], set_2=reason_cities['share4'])[0]
                res_reason_cities_unique[reason] = self.remove_duplicate_elements_in_set(
                    set_1=reason_cities[reason], set_2=reason_cities['share4'])[1]

            if len(res_reason_cities_unique['ttv_kus+']) > 0:
                if channel in channels_group_1:
                    phrase_kus_ttv_plus = {
                        'cities': res_reason_cities_unique['ttv_kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['ttv_kus+']) + comment_ttv_kus_plus[0]
                    }
                else:
                    phrase_kus_ttv_plus = {
                        'cities': res_reason_cities_unique['ttv_kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['ttv_kus+']) + comment_ttv_kus_plus[1]
                    }

            if len(res_reason_cities_unique['ttv_kus-']) > 0:
                if channel in channels_group_1:
                    phrase_kus_ttv_minus = {
                        'cities': res_reason_cities_unique['ttv_kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['ttv_kus-']) + comment_ttv_kus_minus[0]
                    }
                else:
                    phrase_kus_ttv_minus = {
                        'cities': res_reason_cities_unique['ttv_kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['ttv_kus-']) + comment_ttv_kus_minus[1]
                    }

            if len(res_reason_cities_unique['kus+']) > 0 and len(res_reason_cities_unique['ttv_kus+']) == 0:
                if channel in channels_group_2:
                    phrase_kus_plus = {
                        'cities': res_reason_cities_unique['kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['kus+']) + comment_kus_plus[0]
                    }
                elif channel in channels_group_3:
                    phrase_kus_plus = {
                        'cities': res_reason_cities_unique['kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['kus+']) + comment_kus_plus[1]
                    }
                else:
                    phrase_kus_plus = {
                        'cities': res_reason_cities_unique['kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['kus+']) + comment_kus_plus[2]
                    }

            if len(res_reason_cities_unique['kus-']) > 0 and len(res_reason_cities_unique['ttv_kus-']) == 0:
                if channel in channels_group_2:
                    phrase_kus_minus = {
                        'cities': res_reason_cities_unique['kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['kus-']) + comment_kus_minus[0]
                    }
                elif channel in channels_group_3:
                    phrase_kus_minus = {
                        'cities': res_reason_cities_unique['kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['kus-']) + comment_kus_minus[1]
                    }
                else:
                    phrase_kus_minus = {
                        'cities': res_reason_cities_unique['kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['kus-']) + comment_kus_minus[2]
                    }

            if len(res_reason_cities_unique['ttv+']) > 0 and len(res_reason_cities_unique['ttv_kus+']) == 0:
                if channel in channels_group_2:
                    phrase_ttv_plus = {
                        'cities': res_reason_cities_unique['ttv+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['ttv+']) + comment_ttv_plus[0]
                    }
                elif channel in channels_group_3:
                    phrase_ttv_plus = {
                        'cities': res_reason_cities_unique['ttv+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['ttv+']) + comment_ttv_plus[1]
                    }
                else:
                    phrase_ttv_plus = {
                        'cities': res_reason_cities_unique['ttv+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_unique['ttv+']) + comment_ttv_plus[2]
                    }

            if len(res_reason_cities_unique['ttv-']) > 0 and len(res_reason_cities_unique['ttv_kus-']) == 0:
                if channel in channels_group_2:
                    phrase_ttv_minus = {
                        'cities': res_reason_cities_unique['ttv-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['ttv-']) + comment_ttv_minus[0]
                    }
                elif channel in channels_group_3:
                    phrase_ttv_minus = {
                        'cities': res_reason_cities_unique['ttv-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['ttv-']) + comment_ttv_minus[1]
                    }
                else:
                    phrase_ttv_minus = {
                        'cities': res_reason_cities_unique['ttv-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_unique['ttv-']) + comment_ttv_minus[2]
                    }

            if len(res_reason_cities_duplicates['ttv_kus+']) > 0:
                if channel in channels_group_1:
                    phrase_kus_ttv_plus_4 = {
                        'cities': res_reason_cities_duplicates['ttv_kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['ttv_kus+']) + comment_ttv_kus_plus[0]
                    }
                else:
                    phrase_kus_ttv_plus_4 = {
                        'cities': res_reason_cities_duplicates['ttv_kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['ttv_kus+']) + comment_ttv_kus_plus[1]
                    }

            if len(res_reason_cities_duplicates['ttv_kus-']) > 0:
                if channel in channels_group_1:
                    phrase_kus_ttv_minus_4 = {
                        'cities': res_reason_cities_duplicates['ttv_kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['ttv_kus-']) + comment_ttv_kus_minus[0]
                    }
                else:
                    phrase_kus_ttv_minus_4 = {
                        'cities': res_reason_cities_duplicates['ttv_kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['ttv_kus-']) + comment_ttv_kus_minus[1]
                    }

            if len(res_reason_cities_duplicates['kus+']) > 0 and len(res_reason_cities_duplicates['ttv_kus+']) == 0:
                if channel in channels_group_2:
                    phrase_kus_plus_4 = {
                        'cities': res_reason_cities_duplicates['kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['kus+']) + comment_kus_plus[0]
                    }
                elif channel in channels_group_3:
                    phrase_kus_plus_4 = {
                        'cities': res_reason_cities_duplicates['kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['kus+']) + comment_kus_plus[1]
                    }
                else:
                    phrase_kus_plus_4 = {
                        'cities': res_reason_cities_duplicates['kus+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['kus+']) + comment_kus_plus[2]
                    }

            if len(res_reason_cities_duplicates['kus-']) > 0 and len(res_reason_cities_duplicates['ttv_kus-']) == 0:
                if channel in channels_group_2:
                    phrase_kus_minus_4 = {
                        'cities': res_reason_cities_duplicates['kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['kus-']) + comment_kus_minus[0]
                    }
                elif channel in channels_group_3:
                    phrase_kus_minus_4 = {
                        'cities': res_reason_cities_duplicates['kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['kus-']) + comment_kus_minus[1]
                    }
                else:
                    phrase_kus_minus_4 = {
                        'cities': res_reason_cities_duplicates['kus-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['kus-']) + comment_kus_minus[2]
                    }

            if len(res_reason_cities_duplicates['ttv+']) > 0 and len(res_reason_cities_duplicates['ttv_kus+']) == 0:
                if channel in channels_group_2:
                    phrase_ttv_plus_4 = {
                        'cities': res_reason_cities_duplicates['ttv+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['ttv+']) + comment_ttv_plus[0]
                    }
                elif channel in channels_group_3:
                    phrase_ttv_plus_4 = {
                        'cities': res_reason_cities_duplicates['ttv+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['ttv+']) + comment_ttv_plus[1]
                    }
                else:
                    phrase_ttv_plus_4 = {
                        'cities': res_reason_cities_duplicates['ttv+'],
                        'phrase': comment_start[0] + ', '.join(res_reason_cities_duplicates['ttv+']) + comment_ttv_plus[2]
                    }

            if len(res_reason_cities_duplicates['ttv-']) > 0 and len(res_reason_cities_duplicates['ttv_kus-']) == 0:
                if channel in channels_group_2:
                    phrase_ttv_minus_4 = {
                        'cities': res_reason_cities_duplicates['ttv-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['ttv-']) + comment_ttv_minus[0]
                    }
                elif channel in channels_group_3:
                    phrase_ttv_minus_4 = {
                        'cities': res_reason_cities_duplicates['ttv-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['ttv-']) + comment_ttv_minus[1]
                    }
                else:
                    phrase_ttv_minus_4 = {
                        'cities': res_reason_cities_duplicates['ttv-'],
                        'phrase': comment_start[1] + ', '.join(res_reason_cities_duplicates['ttv-']) + comment_ttv_minus[2]
                    }

            if len(res_reason_cities_unique['share']) > 0:
                phrase_share = {
                    'cities': res_reason_cities_unique['share'],
                    'phrase': comment_header_1 + Dates_Operations.get_month(self.month_num, 1, 'Предложный')[0] + ':'
                }

            if len(reason_cities['share4']) > 0:
                if len(reason_cities['share4']) <= 3:
                    phrase_share4 = {
                        'cities': reason_cities['share4'],
                        'phrase': 'Изменение прогноза инвентаря в ' + ', '.join(reason_cities['share4']) + ' связано со значительным изменением доли, начиная с ' + Dates_Operations.get_month(self.month_num, 2, 'Родительный')[0] + ':'
                    }
                else:
                    phrase_share4 = {
                        'cities': reason_cities['share4'],
                        'phrase': comment_header_2 + Dates_Operations.get_month(self.month_num, 2, 'Родительный')[0] + ':'
                    }

            list_of_phrases = [
                phrase_kus_ttv_plus,
                phrase_kus_plus,
                phrase_ttv_plus,
                phrase_kus_ttv_minus,
                phrase_kus_minus,
                phrase_ttv_minus,
                phrase_kus_ttv_plus_4,
                phrase_kus_plus_4,
                phrase_ttv_plus_4,
                phrase_kus_ttv_minus_4,
                phrase_kus_minus_4,
                phrase_ttv_minus_4
            ]

            channel_phrase[channel + '\n'] = [phrase_share, phrase_share4] + list_of_phrases

        channel_phrase_new = Dict_Operations(channel_phrase).sort_keys_in_dict(sorted_keys)
        return channel_phrase_new

    def get__channel_df(self, channel_phrase, res_data, column_name_prev_period, column_name_period_now):
        channel_df = {}
        for channel in channel_phrase.keys():
            channel_normalized = channel.removesuffix('\n')
            df = res_data[res_data['Телеканал'] == channel_normalized].copy()
            if len(df) == 0:
                channel_df[channel_normalized] = pd.DataFrame()
                continue
            try:
                cities = df['Город'].copy()
                df = df.drop(['Телеканал', 'Изменение'], axis=1)
                df = df.rename(columns={'Было': column_name_prev_period, 'Стало': column_name_period_now})
                df[column_name_prev_period] = pd.to_numeric(df[column_name_prev_period], errors='coerce')
                df[column_name_period_now] = pd.to_numeric(df[column_name_period_now], errors='coerce')
                df['Динамика'] = (df[column_name_period_now] / df[column_name_prev_period] - 1) * 100
                formatted_df = pd.DataFrame()
                formatted_df['Город'] = cities
                formatted_df[column_name_prev_period] = df[column_name_prev_period].round(2).map('{:.2f}'.format)
                formatted_df[column_name_period_now] = df[column_name_period_now].round(2).map('{:.2f}'.format)
                formatted_df['Динамика'] = df['Динамика'].round(0).astype(int).map('{}%'.format)
                channel_df[channel_normalized] = formatted_df
            except Exception as e:
                print(f"Ошибка при обработке {channel_normalized}: {e}")
                channel_df[channel_normalized] = pd.DataFrame()
        return channel_df

    def to_file(self,
                output_filename,
                res_data_1,
                res_data_2,
                res_data_3,
                res_data_4,
                channel_phrase,
                column_fact_month,
                column_last_14_days,
                column_prev_14_days,
                column_prev_to_fact_month):
        doc = docx.Document()
        File.set_style_doc_file(doc, 1.0, 1.0, 1.5, 1.5)

        if self.day_num < 8:
            last_month = str('Доля ') + Dates_Operations.get_month(self.month_num, 3, 'Родительный')[1]
            avg_per_2_last_months = str('Доля ') + Dates_Operations.get_month(self.month_num, 4, 'Родительный')[1] + ' - ' + Dates_Operations.get_month(self.month_num, 3, 'Родительный')[1]
            avg_per_3_last_months = str('Доля ') + Dates_Operations.get_month(self.month_num, 5, 'Родительный')[1] + ' - ' + Dates_Operations.get_month(self.month_num, 4, 'Родительный')[1]
        else:
            last_month = str('Доля ') + Dates_Operations.get_month(self.month_num, 2, 'Родительный')[1]
            avg_per_2_last_months = str('Доля ') + Dates_Operations.get_month(self.month_num, 3, 'Родительный')[1] + ' - ' + Dates_Operations.get_month(self.month_num, 2, 'Родительный')[1]
            avg_per_3_last_months = str('Доля ') + Dates_Operations.get_month(self.month_num, 4, 'Родительный')[1] + ' - ' + Dates_Operations.get_month(self.month_num, 3, 'Родительный')[1]

        channel_df_var_1 = self.get__channel_df(channel_phrase, res_data_1, last_month, column_fact_month)
        channel_df_var_2 = self.get__channel_df(channel_phrase, res_data_2, avg_per_2_last_months, column_fact_month)
        channel_df_var_3 = self.get__channel_df(channel_phrase, res_data_3, column_prev_14_days, column_last_14_days)
        channel_df_var_4 = self.get__channel_df(channel_phrase, res_data_4, avg_per_3_last_months, column_prev_to_fact_month)

        channel_df = [
            {'Вариант 1': channel_df_var_1},
            {'Вариант 2': channel_df_var_2},
            {'Вариант 3': channel_df_var_3},
            {'Вариант 4': channel_df_var_4}
        ]

        for c, p in channel_phrase.items():
            empty_dfs_count = 0
            empty_dfs_count_not4 = 0
            for idf, name_df in enumerate(channel_df):
                name, df_dict = list(name_df.items())[0]
                df = df_dict.get(c.removesuffix('\n'), pd.DataFrame())
                if len(df) == 0:
                    empty_dfs_count += 1
                    if name != 'Вариант 4':
                        empty_dfs_count_not4 += 1

            if empty_dfs_count == len(channel_df):
                continue

            p1 = doc.add_paragraph()
            r1 = p1.add_run(c.removesuffix('\n'))
            r1.bold = True

            for idf, name_df in enumerate(channel_df):
                name, df_dict = list(name_df.items())[0]
                df = df_dict.get(c.removesuffix('\n'), pd.DataFrame())

                if len(df) == 0:
                    continue

                if name == 'Вариант 1' and empty_dfs_count_not4 != 3:
                    if p[0]['phrase'] != '':
                        doc.add_paragraph(p[0]['phrase'])
                if name == 'Вариант 4':
                    if p[1]['phrase'] != '':
                        doc.add_paragraph(p[1]['phrase'])

                t = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])
                t.style = 'Table Grid'

                first_row_cells = t.rows[0].cells
                columns = list(df.columns)
                for col_idx in range(len(columns)):
                    first_row_cells[col_idx].text = columns[col_idx]

                for cell in first_row_cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.bold = True

                for i in range(df.shape[0]):
                    for j in range(df.shape[1]):
                        t.cell(i + 1, j).text = str(df.values[i, j])

                is_need_to_write = True
                if name != 'Вариант 4' and idf != len(channel_df) - 1:
                    for i in range(idf + 1, len(channel_df)-1):
                        next_df = list(channel_df[i].values())[0].get(c.removesuffix('\n'), pd.DataFrame())
                        if not next_df.empty:
                            is_need_to_write = False
                            break

                p2 = doc.add_paragraph()
                if name != 'Вариант 4' and is_need_to_write:
                    for phrase in p[2:8]:
                        if phrase['phrase'] != '':
                            p2.add_run('\n')
                            p2.add_run(phrase['phrase'])
                elif name == 'Вариант 4':
                    for phrase in p[8:]:
                        if phrase['phrase'] != '':
                            p2.add_run('\n')
                            p2.add_run(phrase['phrase'])

        doc.save(output_filename)

    def get__comments(self, filename, output_filename, data, data_api, historical_filepath,
                     column_fact_month, column_last_14_days, column_prev_14_days,
                     column_prev_to_fact_month, sorted_keys, cond_grp, cond_share,
                     cond_kus, cond_ttv, channels_group_1, channels_group_2, channels_group_3):
        print("Загрузка исторических данных...")
        historical_data = self.load_historical_data(historical_filepath)
        print(f"Загружено {len(historical_data)} целевых аудиторий")

        reason_channels, res_data_1, res_data_2, res_data_3, res_data_4 = self.get_reason_channels__and__res_data(
            data, data_api, historical_data,
            column_fact_month, column_last_14_days, column_prev_14_days, column_prev_to_fact_month,
            cond_grp, cond_share, cond_kus, cond_ttv
        )

        reason_channels_formatted = self.get_cities_formatted(reason_channels)
        channel_reason_cities = self.get_cities_and_reasons(reason_channels_formatted)
        channel_phrase = self.get_explanations(channel_reason_cities, sorted_keys,
                                               channels_group_1, channels_group_2, channels_group_3)

        self.to_file(output_filename, res_data_1, res_data_2, res_data_3, res_data_4,
                     channel_phrase, column_fact_month, column_last_14_days,
                     column_prev_14_days, column_prev_to_fact_month)
