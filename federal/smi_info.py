import pandas as pd
import numpy as np
import pymorphy3 as pmrph
import re
import math
from datetime import datetime, timedelta

class color:
   BOLD = '\033[1m'
   PURPLE = '\033[95m'
   END = '\033[0m'


class SMI_info:
    """
        Класс для обработки файла от отдела СМИ по различным Переброскам - сокращениям.
    """
    def __init__(self, smi_filepath):
        self.smi_filepath = smi_filepath
    

    @staticmethod
    # Функция для округления
    def round_number(match):
        #return str(round(float(match.group())))
        return str(math.ceil(float(match.group())))


    @staticmethod
    # Функция для замены чисел на целые с округлением
    def replace_numbers(comment):
        # Используем регулярное выражение для замены
        return re.sub(r'\d+\.\d+', SMI_info.round_number, comment)


    @staticmethod
    # Функция для объединения комментариев
    def combine_comments(group):
        """
            Функция, объединяющая комментарии по одинаковым каналам, месяцам и датам.
        """
        combined_comment = ' '.join(group['Комментарий'])

        return pd.Series({
            'Канал': group['Канал'].iloc[0],
            'Месяц': group['Месяц'].iloc[0],
            'Дата': group['Дата'].iloc[0],
            #'Дата из СМИ': group['Дата из СМИ'].iloc[0],
            'Комментарий': combined_comment
        })
    

    @staticmethod
    def sum_identical_sentences(sentences):
        """
            Функция для объединения одинаковых предложений в одно. 
            При этом неодинаковые предложения остаются без изменения.
            Например: "Сокращение рекламных объемов 30 GRP. Сокращение рекламных объемов 20 GRP." =>
            "Сокращение рекламных объемов 50 GRP."
            Args:
                sentences: список из предложений.
            Returns:
                result: обновленный список из предложений.
        """
        result = []
        for sentence in sentences:
            # Разделяем строку на предложения
            parts = sentence.split('. ')
            templates_values = list(map(lambda part: [ re.split(r'\d+.+', part)[0], int(re.findall(r'\d+', part)[0]) ], parts))

            template_sum = {}
            for template, value in templates_values:
                if template in template_sum.keys():
                    template_sum[template] += value
                else:
                    template_sum[template] = value

            united_sentances = list(map(lambda template_value: f'{template_value[0]}{template_value[1]} GRP', template_sum.items()))
            result.append('. '.join(united_sentances) + '.')
        
        return result

    
    def read_smi_file(self, channel: str, month: str, channels_need_replace: dict, year: int):
        """
            Функция для чтения файла от отдела СМИ по переброскам.
            Args:
                smi_filepath: Имя файла/полный путь к файлу с Перебросками от отдела СМИ.
                channel: Канал, по которому будем смотреть изменения Объемов.
                month: Месяц, по которому будем смотерть изменения Объемов.
                channels_need_replace: Словарь каналов, которые нуждаются в замене имени.
                year: Год, по которому будем смотерть изменения Объемов.
            Returns:
                volume_transfer_: DataFrame с изменениями обьемов по конкретному каналу, месяцу и Году.
            
        """
        channels_need_replace = {
            '2Х2': '2X2',
            '5 КАНАЛ': 'ПЯТЫЙ КАНАЛ',
            'ПЕРВЫЙ': 'ПЕРВЫЙ КАНАЛ',
            'СТС ЛАВ': 'СТС LOVE',
            'ТВ3': 'ТВ-3',
            'ТНТ4': 'ТНТ 4'
        }
        channel_not_found = ''
        data = pd.read_excel(self.smi_filepath, skiprows = 1, sheet_name = 'Итоги')
        data = data.loc[(data['Статус'] == 'реализовано') & (data['Год'] == year)]
        volume_transfer = data[[ 
                     'Канал', 
                     'Месяц', 
                     'Год', 
                     'GRP сокращено', 
                     'GRP открыто', 
                     'Итог GRP в регионы -', 
                     'Итог GRP из регионов +', 
                     'Дата осуществления переброски', 
                     'Комментарий']]
        volume_transfer = volume_transfer.loc[(volume_transfer['Комментарий'] != 'стратегическая переброска') & (volume_transfer['Комментарий'] != 'кросс-промо')]
        #volume_transfer = volume_transfer.loc[(volume_transfer['Комментарий'] != 'стратегическая переброска')]
        
        #Изменение столбца с каналами
        channels_old = list(volume_transfer['Канал'])
        channels_new = []
        for i in range(len(channels_old)):
            channels_new.append(channels_old[i].upper())
        volume_transfer['Канал'] = volume_transfer['Канал'].replace(channels_old, channels_new)
        volume_transfer['Канал'].replace(channels_need_replace, inplace = True)
        
        #Изменение столбца с месяцем
        months_old = list(volume_transfer['Месяц'])
        months_new = []
        for i in range(len(months_old)):
            months_new.append(months_old[i].title())
        volume_transfer['Месяц'] = volume_transfer['Месяц'].replace(months_old, months_new)
        volume_transfer['Год'] = volume_transfer['Год'].astype(int)
        
        old_dates = list(volume_transfer['Дата осуществления переброски'])
        new_dates = []
        for i in range(len(old_dates)):
            new_date = old_dates[i].strftime('%Y-%m-%d')
            new_dates.append(new_date)
        volume_transfer['Дата осуществления переброски'] = volume_transfer['Дата осуществления переброски'].replace(old_dates, new_dates)
        volume_transfer_ = volume_transfer.loc[(volume_transfer['Месяц'] == month) & (volume_transfer['Канал'] == channel)].reset_index(drop = True)

        if len(volume_transfer_) < 1:
            channel_not_found = channel
            #print(f'Не нашлось релевантных данных от СМИ в {month} по каналу {channel}', sep = '\n\n', end = '\n')

        volume_transfer_['Комментарий'] = volume_transfer_['Комментарий'].apply(str)
        #volume_transfer_ = volume_transfer_[volume_transfer_['Комментарий'].notna()]
        return volume_transfer_, channel_not_found

    
    def get_volumes_comments(self, delta_df, channels_need_replace, year: int, df_limits, smi_criteria = 0.1):
        """
            Функция для генерации комментариев по изменению объемов из файла от отдела СМИ (Новое)Переброски-сокращения.xlsm.
            Args:
                delta_df: DataFrame с изменениями по дням или накопленными изменениями из Федерального Кубика.
                year: Год по которому смотрим изменения.
                smi_criteria: Критерий, согласно которому изменения будет относиться к критическим или нет.
            Returns:
                result_df: DataFrame с комментариями по изменению объемов на Федеральных каналах.
        """
        comments_full = {}
        channels_not_found = {}
        for i in range(len(delta_df)):
            #Выделяем канал, месяц, дату из таблицы с изменениями по дням Фед. Кубик
            channel_1 = delta_df.iloc[i]['Канал']
            #Определение порога для канала
            limit = int(df_limits[channel_1])

            date_1 = pd.to_datetime(delta_df.iloc[i]['Дата']).strftime('%Y-%m-%d')
            #Выделяем номер месяца из даты
            moth_cubik_date = pd.to_datetime(delta_df.iloc[i]['Дата']).month
            day_cubik = pd.to_datetime(delta_df.iloc[i]['Дата']).day
            month_1 = str(delta_df.iloc[i]['Месяц']).split('\'')[0].title()
            delta_grp = delta_df.iloc[i]['Изменение GRP']
            volume_transfer, channel_not_found = self.read_smi_file(channel_1, month_1, channels_need_replace, year)

            if (month_1 in channels_not_found.keys() and channel_not_found):
                channels_not_found[month_1].add(channel_not_found)
            elif channel_not_found:
                channels_not_found[month_1] = set([channel_not_found])
            
            pattern_telemag = 'телемагазины'
            pattern_setka = 'корректировка сетки'
            comments = {}
            used_channels_dates = []

            for j in range(len(volume_transfer)):
                #Выделяем канал, месяц, дату из таблицы с изменениями Объемов
                channel_2 = volume_transfer.iloc[j]['Канал']
                month_2 = volume_transfer.iloc[j]['Месяц']
                date_2 = pd.to_datetime(volume_transfer.iloc[j]['Дата осуществления переброски']).strftime('%Y-%m-%d')
                #Выделяем номер месяца из даты осуществления переброски
                month_smi_date = pd.to_datetime(volume_transfer.iloc[j]['Дата осуществления переброски']).month
                day_smi = pd.to_datetime(volume_transfer.iloc[j]['Дата осуществления переброски']).day

                if channel_1 == channel_2 and month_1 == month_2 and (moth_cubik_date - month_smi_date) == 0.0 and (day_cubik - day_smi) < 3 and (day_cubik - day_smi) >= 0:
                    if delta_grp < 0:
                        grp_minus = volume_transfer.iloc[j]['GRP сокращено']
                        #Обработка случая, если в столбце 'GRP открыто' и интересующей строчке j не NaN
                        if not np.isnan(grp_minus):
                            if np.abs(grp_minus) >= limit * smi_criteria:
                                #Генерация комментариев для снятия телемагазинов
                                if re.search(pattern_telemag, volume_transfer.iloc[j]['Комментарий']):
                                    comments = {
                                        'Канал': channel_2,
                                        'Дата': date_1,
                                        'Дата из СМИ': date_2,
                                        'Комментарий': f'Размещение телемагазинов {(-1) * grp_minus} GRP.'
                                    }
                                    
                                elif re.search(pattern_setka, volume_transfer.iloc[j]['Комментарий']):
                                    comments = {
                                        'Канал': channel_2,
                                        'Дата': date_1,
                                        'Дата из СМИ': date_2,
                                        'Комментарий': f'Корректировка сетки {(-1) * grp_minus} GRP.'
                                    }

                                #Генерация комментариев для остальных случаев
                                else:
                                    comments = {
                                        'Канал': channel_2,
                                        'Дата': date_1,
                                        'Дата из СМИ': date_2,
                                        'Комментарий': f'Сокращение рекламных объемов {(-1) * grp_minus} GRP.'
                                    }
                        #Обработка случая, если в столбце 'GRP открыто' и интересующей строчке j значение NaN
                        else:
                            grp_minus_ = volume_transfer.iloc[j]['Итог GRP в регионы -']
                            if not np.isnan(grp_minus_):
                                if np.abs(grp_minus_) >= limit * smi_criteria:
                                    comments = {
                                    'Канал': channel_2,
                                    'Дата': date_1,
                                    'Дата из СМИ': date_2,
                                    'Комментарий': f'Перераспределение в регионы {(-1) * grp_minus_} GRP.'
                                    }
        
                    elif delta_grp > 0:
                        grp_plus = volume_transfer.iloc[j]['GRP открыто']
                        #Обработка случая, если в столбце 'GRP открыто' и интересующей строчке j не NaN
                        if not np.isnan(grp_plus):
                            if grp_plus >= limit * smi_criteria:
                                #Генерация комментариев для размещения телемагазинов
                                if re.search(pattern_telemag, volume_transfer.iloc[j]['Комментарий']):
                                    comments = {
                                        'Канал': channel_2,
                                        'Дата': date_1,
                                        'Дата из СМИ': date_2,
                                        'Комментарий': f'Снятие телемагазинов {grp_plus} GRP.'
                                    }

                                elif re.search(pattern_setka, volume_transfer.iloc[j]['Комментарий']):
                                    comments = {
                                        'Канал': channel_2,
                                        'Дата': date_1,
                                        'Дата из СМИ': date_2,
                                        'Комментарий': f'Корректировка сетки {grp_plus} GRP.'
                                    }

                                #Генерация комментариев для остальных случаев
                                else:
                                    comments = {
                                        'Канал': channel_2,
                                        'Дата': date_1,
                                        'Дата из СМИ': date_2,
                                        'Комментарий': f'Дооткрытие рекламных объемов {grp_plus} GRP.'
                                    }
                        #Обработка случая, если в столбце 'GRP открыто' и интересующей строчке j значение NaN
                        else:
                            grp_plus_ = volume_transfer.iloc[j]['Итог GRP из регионов +']
                            if not np.isnan(grp_plus_):
                                if grp_plus_ >= limit * smi_criteria:
                                    comments = {
                                        'Канал': channel_2,
                                        'Дата': date_1,
                                        'Дата из СМИ': date_2,
                                        'Комментарий': f'Перераспределение из регионов {(-1) * grp_plus_} GRP.'
                                    }
                #Заполнение итогового словаря с изменениями             
                if comments:
                    channel_date = channel_2 + month_2
                    if month_2 in comments_full.keys():
                        if not channel_date in used_channels_dates:
                            comments_full[month_2].append(comments)
                    else:
                        comments_full[month_2] = [comments]
                        used_channels_dates.append(channel_date)
        #Создание DataFrame с комментариями
        result = []
        for month, channel_comments in comments_full.items():
            df = pd.DataFrame(comments_full[month])
            df['Месяц'] = month
            df = df[['Канал', 'Месяц', 'Дата', 'Дата из СМИ', 'Комментарий']]
            result.append(df)

        #Если не нашлось релевантных данных
        if len(result) == 0:
            return [], []
        
        result_df = pd.concat(result)
        result_df['Дата'] = pd.to_datetime(result_df['Дата'], format = '%Y-%m-%d').dt.strftime('%Y-%m-%d')
        result_df['Дата'] = pd.to_datetime(result_df['Дата'])
        result_df_cleaned = result_df.drop_duplicates(keep = 'first')
        df = result_df_cleaned.reset_index(drop = True)
        # Преобразуем столбцы 'Дата' и 'Дата из СМИ' в формат datetime
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['Дата из СМИ'] = pd.to_datetime(df['Дата из СМИ'])

        # Группируем по 'Канал', 'Месяц' и объединяем комментарии
        df_result = df.groupby(['Канал', 'Месяц', 'Дата', 'Дата из СМИ']).filter(
            lambda x: x['Дата'].iloc[0] - x['Дата из СМИ'].iloc[0] < timedelta(days = 3)
        ).groupby(['Канал', 'Месяц', 'Дата']).apply(SMI_info.combine_comments).reset_index(drop = True)

        #df_result.drop_duplicates(['Канал', 'Месяц', 'Дата'], inplace=True)

        #Приведение чисел в стобце Комментарий к целочисленному виду
        comments_old = list(df_result['Комментарий'])
        updated_comments = [SMI_info.replace_numbers(comment) for comment in comments_old]
        
        # Замена столбца дат на новый конвертированный столбец
        df_result['Комментарий'] = df_result['Комментарий'].replace(comments_old, updated_comments)
        df_result_cleaned = df_result.drop_duplicates(subset = ['Канал', 'Месяц', 'Комментарий'], keep = 'first')

        #Объединение комментариев, если тексты одинаковые
        text_init = list(df_result_cleaned['Комментарий'])
        # Применяем функцию к нашему списку
        text_updated = SMI_info.sum_identical_sentences(text_init)
        df_result_cleaned['Комментарий'] = df_result_cleaned['Комментарий'].replace(text_init, text_updated)
        
        return df_result_cleaned, channels_not_found