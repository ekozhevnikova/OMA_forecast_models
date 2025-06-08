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
                                self.df.at[i, 'Комментарий'] = '. '.join(icomments)

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
            worksheet.set_column('G:G', 85.0, table_fmt_2)
    

    @staticmethod
    def update_comments_file(filepath: str, data_new):
        """
            Функция для обновления файла с Комментариями. В процессе работы считывается файл с исходными Комментариями и в конец добавляются новые.
            В конце файл сохраняется.
            Args:
                filepath: Файл со старыми комментариями
                data_new: Новые комментарии
            
        """
        comments = pd.read_excel(filepath)
        comments_full = pd.concat([comments, data_new]).reset_index(drop = True)
        Federal_Postprocessing.make_style_of_table(filepath = filepath, 
                                           output_df = comments_full, 
                                           sheet_name = 'Sheet1')