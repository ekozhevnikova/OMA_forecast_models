import pandas as pd
import numpy as np
import re
import pymorphy3 as pmrph


class Federal_Postprocessing:
    def __init__(self, df):
        self.df = df


    def replace_name_of_months(self, column_name_with_month: str, year: str):
        """
            Функция для замены названий месяцев на формат Май'25 вместо Май.
            Args:
                df: DataFrame, в котором нужно произвести замену столбца с Месяцем
                column_name_with_month: Название колокни с месяцем
                year: Год в формате строки
            Returns:
                df: Отформартированный DataFrame
        """
        year_formatted = '\'' + year[-2:]
        months = list(self.df[column_name_with_month])
        formatted_months = [month + year_formatted for month in months]
        self.df[column_name_with_month] = self.df[column_name_with_month].replace(list(self.df[column_name_with_month]), formatted_months)
        return self.df
    

    def clean_comments_from_new_ones(self, comments_filepath):
        """
            Функция для зачистки только что добавленных комментариев. Используется для генерации накопленных изменений за период.
            Args:
                comments_filepath: Путь к файлу с Комментариями
                Здесь подразумевается, что self.df: Свежие комментарии с изменениями по дням. (именно от них будем зачищать файл Комментарии.xlsx)
            Returns:
                result: зачищенный DataFrame от новых комментариев.
        """
        comments = pd.read_excel(comments_filepath)

        cols = ['Канал', 'Месяц', 'Дата', 'Изменение GRP']
        # Устанавливаем составной индекс
        df_indexed = comments.set_index(cols)
        b_indexed = self.df.set_index(cols)
        
        # Фильтруем строки, которых нет в B
        result = comments[~df_indexed.index.isin(b_indexed.index)]
        return result

    

    @staticmethod
    def filter_comments_channel_overtime(by_days, df_summ):
        """
            Функция, которая удаляет комментарии в накопленных изменениях, если данный канал по данному месяцу встретился выше в изменениях по дням.
            Удаляется канад, месяц, комментарий. А доп комментарий из столбца "Доп столбец" переносится выше в блок с изменениями по дням.
            Args:
                by_days: DataFrame по дням
                df_summ: DataFrame с накопленными изменениями за период
        """
        idx_to_delete = []
        #цикл по изменениям по дням
        for i in range(len(by_days)):
            channel_i = by_days.iloc[i]['Канал']
            month_i = by_days.iloc[i]['Месяц']
            #цикл по накопленным изменениям за период
            for j in range(len(df_summ)):
                channel_j = df_summ.iloc[j]['Канал']
                month_j = df_summ.iloc[j]['Месяц']
                additional_comment = df_summ.iloc[j]['Доп столбец']

                if channel_i == channel_j and month_i == month_j:
                    by_days.at[i, 'Доп столбец'] = additional_comment
                    idx_to_delete.append(j)
                else:
                    continue
        df_summ.drop(idx_to_delete, inplace = True)
        return by_days, df_summ
    

    def comments_dublicates_actualize(self):
        """
            Функция для удаления дубликатов Комментариев или дубликатов подстрок в Комментариях.
        """
        for i, irow in self.df.iterrows():
            if not irow['Комментарий'] or irow['Комментарий'] == np.nan:
                icomments = irow['Комментарий'].split('. ')
                for j, jrow in self.df.iloc[i + 1: ].iterrows():
                    if jrow['Канал'] != irow['Канал']:
                        continue
                    jcomments = jrow['Комментарий'].split('. ')
                    if len(icomments + jcomments) != len(set(icomments + jcomments)):
                        if irow['Дата'] < jrow['Дата']:
                            for comment in icomments:
                                jcomments = [c for c in jcomments if c != comment]
                                self.df.at[j, 'Комментарий'] = '. '.join(jcomments)
                        else:
                            for comment in jcomments:
                                icomments = [c for c in icomments if c != comment]

                                
    def clean_comments(self, result_df, date_of_forecast):
        """
            Функция для зачистки комментариев.
        """
        possible_comments = [
                     'Рост доли с', 
                     'Снижение доли с', 
                     'Рост телесмотрения', 
                     'Снижение телесмотрения', 
                     'Рост КУС', 
                     'Снижение КУС', 
                     'Рост КУС за счет роста прогноза внедомашнего телесмотрения', 
                     'Снижение КУС за счет роста прогноза внедомашнего телесмотрения', 
                     'Рост прогноза СП', 
                     'Снижение прогноза СП', 
                     'Рост ТП канала', 
                     'Снижение ТП канала'
                    ]
        possible_comments_smi = [
            'Размещение телемагазинов',
            'Снятие телемагазинов',
            'Корректировка сетки',
            'Сокращение рекламных объемов',
            'Дооткрытие рекламных объемов',
            'Перераспределение в регионы',
            'Перераспределение из регионов'
        ]
        #Конвертация названий столбцов в капс
        result_df['Канал'] = result_df['Канал'].str.upper()

        for i in range(len(self.df)):
            channel_i = self.df.iloc[i]['Канал']
            date_i = self.df.iloc[i]['Дата']
            for j in range(len(result_df)):
                channel_j = result_df.iloc[j]['Канал']
                date_j = result_df.iloc[j]['Дата']
                if 'Дата осуществления' in result_df.columns:
                    smi_date_flag = result_df.iloc[j]['Дата осуществления']
                    if channel_i == channel_j and date_i == date_j:
                        comment = result_df.iloc[j]['Комментарий']
                        if comment is not np.nan:
                            if date_j == date_of_forecast:
                                if smi_date_flag != True:
                                    comment_splitted = comment.split('. ')
                                    # Фильтруем комментарии
                                    filtered_comments = [comment for comment in comment_splitted if any(possible in comment for possible in possible_comments)]
                                    result_df.at[j, 'Комментарий'] = '. '.join(filtered_comments)
                                elif smi_date_flag == True:
                                    comment_splitted = comment.split('. ')
                                    # Фильтруем комментарии
                                    filtered_comments = [comment for comment in comment_splitted if any(possible_smi in comment for possible_smi in possible_comments_smi)]
                                    result_df.at[j, 'Комментарий'] = '. '.join(filtered_comments)
                            else:
                                result_df.at[j, 'Комментарий'] = ''
                else:
                    if channel_i == channel_j and date_i == date_j:
                        comment = result_df.iloc[j]['Комментарий']
                        if comment is not np.nan:
                            if date_j == date_of_forecast:
                                comment_splitted = comment.split('. ')
                                # Фильтруем комментарии
                                filtered_comments = [comment for comment in comment_splitted if any(possible in comment for possible in possible_comments)]
                                result_df.at[j, 'Комментарий'] = '. '.join(filtered_comments)
                            else:
                                result_df.at[j, 'Комментарий'] = ''

        result_df_ = result_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
        return result_df_


    @staticmethod
    def make_style_of_table(filepath, output_df, sheet_name):
        """
            Функция для генерации внешнего вида таблицы с Комментариями.
            Args:
                filepath: путь к файлу, в который будем сохранять итоговый результат
                output_df: DataFrame, который будем стилизировать
                sheet_name: имя листа, на который это будет записываться.
            Returns:
                Стилизированная таблица в файле xlsx
        """
        with pd.ExcelWriter(filepath, 
                        date_format = 'DD.MM.YYYY', 
                        datetime_format = 'DD.MM.YYYY',
                        engine = 'xlsxwriter') as writer:
            output_df.to_excel(writer, index = None)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
        
            #Стиль шапки таблицы
            header_format = workbook.add_format({'bold': True,
                                                'text_wrap': True, #перенос текста
                                                'align': 'center', #выравнение текста в ячейке
                                                'align': 'vcenter', #выравнение текста в ячейке
                                                'center_across': True
                                                })
            header_format_2 = workbook.add_format({'bold': True,
                                                'text_wrap': True, #перенос текста
                                                'align': 'center', #выравнение текста в ячейке
                                                'align': 'vcenter', #выравнение текста в ячейке
                                                'center_across': True,
                                                'bg_color': '#FFFFE1'
                                                })
            
            #Стиль тела таблицы для Канала, Месяца
            table_fmt_1 = workbook.add_format({'bold': True, 'align': 'left', 'border': 0})
            
            #Стиль тела таблицы для Канала, Месяца
            table_fmt_2 = workbook.add_format({'align': 'left'})
        
            #Стиль тела таблицы для Канала, Месяца
            table_fmt_3 = workbook.add_format({'align': 'left', 'italic': True})
        
            #Стиль тела таблицы для Даты, Изменения и Порога
            table_fmt_4 = workbook.add_format({'align': 'right', 'num_format': '0'})
        
            format_column = workbook.add_format({'align': 'right', 'bg_color': '#FFFFE1'})
        
            
            worksheet.write('A1', 'Канал', header_format)
            worksheet.write('B1', 'Месяц', header_format)
            worksheet.write('C1', 'Дата', header_format)
            worksheet.write('D1', 'Изменение GRP', header_format)
            worksheet.write('E1', 'Порог', header_format_2)
            worksheet.write('F1', 'Доп столбец', header_format)
            worksheet.write('G1', 'Комментарий', header_format)
            worksheet.set_column('A:A', 14.0, table_fmt_1)
            worksheet.set_column('B:B', 11.7, table_fmt_1)
            worksheet.set_column('C:C', 13.9, table_fmt_4)
            worksheet.set_column('D:D', 13.9, table_fmt_4)
            worksheet.set_column('E:E', 10.7, format_column)
            worksheet.set_column('F:F', 44.0, table_fmt_3)
            worksheet.set_column('G:G', 145.0, table_fmt_2)
    

    @staticmethod
    def update_comments_file(filepath: str, data_new):
        """
            Функция для обновления файла с Комментариями. В процессе работы считывается файл с исходными Комментариями и в конец добавляются новые.
            В конце файл сохраняется.
            Args:
                filepath: Файл со старыми комментариями
                data_new: Новые комментарии
            
        """
        if len(data_new) != 0:
            comments = pd.read_excel(filepath)
            comments_full = pd.concat([comments, data_new]).reset_index(drop = True)
            Federal_Postprocessing.make_style_of_table(filepath = filepath, 
                                            output_df = comments_full, 
                                            sheet_name = 'Sheet1')
        else:
            print('Ошибка! Вы пытаетесь сохранить пустой DataFrame!')