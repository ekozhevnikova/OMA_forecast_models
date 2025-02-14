import numpy as np
import os
import img2pdf
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell
import uuid
mpl.rc('font',family = 'Arial')
from threading import Lock
lock = Lock()



class Postprocessing:
    """
        Класс для постобработки ДатаФреймов Временных Рядов, полученных в результате ML-моделей.
    """
    def __init__(self, df, forecast_df):
        self.df = df
        self.forecast_df = forecast_df


    @staticmethod
    def calculate_average_forecast(list_of_forecasts: list):
        """
        Объединяет несколько DataFrame с прогнозами, суммируя столбцы с одинаковыми названиями.
            Args: 
                forecasts: Список DataFrame с прогнозами.
            Returns: 
                avg_forecast: DataFrame с суммированными прогнозами.
        """
        combined_forecasts = pd.concat(list_of_forecasts, axis = 1)
        avg_forecast = combined_forecasts.groupby(combined_forecasts.columns, axis = 1).sum()
        return avg_forecast
    

    @staticmethod
    def ensemble_of_models(df, *avg_forecasts):
        if not avg_forecasts:  # Проверка, есть ли хотя бы один прогноз
            raise ValueError('Не передано ни одного прогноза для тестирования.')

        general_df = pd.concat(avg_forecasts, axis = 1) # Объединяем доступные прогнозы
        general_df = general_df[df.columns]  # Упорядочиваем колонки в соответствии с исходными данными
        return general_df
    

    @staticmethod
    def calculate_forecast_error(forecast_df, test_data):
        """
        Вычисляет ошибки прогноза и среднюю процентную ошибку.

        Args:
            forecast_df: DataFrame с прогнозными значениями.
            test_data: Тестовые данные для сравнения.

        Returns:
            percentage_error: DataFrame с ошибками прогноза (в %).
        """
        # Вычисляем процентные ошибки
        percentage_error = (((forecast_df / test_data) - 1) * 100).abs()
        mean_values = percentage_error.mean()
        # Добавляем новую строку с названием индекса "Mean"
        percentage_error.loc['Mean'] = mean_values

        return percentage_error
    

    def get_plot(self, 
             column_name_with_date: str,
             save_dir,
             test_data = None,
             last_n_list = None, 
             #colors = ['mediumorchid', 'mediumseagreen', 'orange'],
             nrows = 3, 
             ncols = 2,
             figsize = (10, 9.5)):
        """
            Функция для построения графиков
            Args:
                df: Dataframe с исходными фактическими данными
                last_n_list: list DataFrameов с последними N значениями фактических данных
                forecast_df: DataFrame с прогнозом
                save_dir: Путь к папке, куда будут сохраняться готовые графики
                nrows: Размер Картинки с графиками (количество строк) для plt.subplots
                ncols: Размер Картинки с графиками (количество столбцов) для plt.subplots
            Returns:
                График с историческими данными и прогнозными
        """
        #Проверка на то, что индексом в исходном DataFrame является столбец с датами
        if self.df.index.name != column_name_with_date:
            self.df.set_index(column_name_with_date, inplace = True)
            
        #Проверка на то, что индексом в DataFrame с прогнозом является столбец с датами
        if self.forecast_df.index.name != column_name_with_date:
            self.forecast_df = self.forecast_df.reset_index()
            self.forecast_df = self.forecast_df.rename(columns = {self.forecast_df.columns[0]: column_name_with_date})
            self.forecast_df.set_index(column_name_with_date, inplace = True)

        # Если test_data задан, проверяем его индекс
        if test_data is not None and test_data.index.name != column_name_with_date:
            test_data = test_data.set_index(column_name_with_date)
        
        # Создаем папку для сохранения графиков, если она не существует
        os.makedirs(save_dir, exist_ok = True)
        
        column_names = list(self.df.columns)
        
        #определение количества графиков
        nplots = len(column_names) // (nrows * ncols) if len(column_names) % (nrows * ncols) == 0 else len(column_names) // (nrows * ncols) + 1

        for npl in range(nplots):
            fig, axs = plt.subplots(nrows, ncols, figsize = figsize)

            c, r = 0, 0
            for col_idx in range(nrows * ncols * npl, min(nrows * ncols * (npl + 1), len(column_names))):
                axs[r, c].tick_params(axis = 'both', which = 'major', labelsize = 9)
                axs[r, c].set_title(column_names[col_idx], fontsize = 14)

                column = column_names[col_idx]
                
                list_of_y_label_min = [] #список из минимальных значений по оси y для каждого из рассматриваемых годов
                list_of_y_label_max = [] #список из максимальных значений по оси y для каждого из рассматриваемых годов
                #Построение графиков для случая, когда выбрано N последних значений из исходных данных
                if last_n_list is not None:
                    #построение графиков с фактическими данными за выбранный период
                    for i, last_n in enumerate(last_n_list):
                        axs[r, c].plot(last_n.index.strftime('%b'), 
                                    last_n[column], 
                                    linewidth = 4.0,
                                    label = f'{last_n.index.year[0]}')
                        list_of_y_label_min.append(min(list(last_n[column])))
                        list_of_y_label_max.append(max(list(last_n[column])))
                    #построение графика для прогнозируемых значений
                    axs[r, c].plot(self.forecast_df.index.strftime('%b'), 
                                self.forecast_df[column], 
                                label = f'Forecast {self.forecast_df.index.year[0]}',
                                linestyle = 'dashed',
                                linewidth = 4.0,
                                color = 'red')
                    list_of_y_label_min.append(min(list(self.forecast_df[column])))
                    list_of_y_label_max.append(max(list(self.forecast_df[column])))
                    y_label_min = min(list_of_y_label_min) #поиск минимального значения из всех минимумов по оси y
                    y_label_max = max(list_of_y_label_max) #поиск максимального значения из всех максимумов по оси y
                    axs[r, c].set_xlim(self.forecast_df.index.strftime('%b')[0], self.forecast_df.index.strftime('%b')[-1])
                    axs[r, c].set_ylim(y_label_min - 0.5, y_label_max + 0.5)
                else:
                    last_3_years = [12, 24, 36]  #последние 12, 24, 36 месяцев
                    for months_ago in last_3_years:
                        if months_ago > 12:
                            start_idx = -months_ago
                            end_idx = start_idx + 12
                            historical_data = self.df.iloc[start_idx:end_idx]
                            axs[r, c].plot(historical_data.index.strftime('%b'), 
                                        historical_data[column],
                                        label = f'{historical_data.index.year[0]}')
                            list_of_y_label_min.append(min(list(historical_data[column])))
                            list_of_y_label_max.append(max(list(historical_data[column])))
                        elif months_ago <= 12:
                            historical_data = self.df.iloc[-12:]
                            axs[r, c].plot(historical_data.index.strftime('%b'), 
                                        historical_data[column],
                                        label = f'{historical_data.index.year[0]}')
                            list_of_y_label_min.append(min(list(historical_data[column])))
                            list_of_y_label_max.append(max(list(historical_data[column])))

                # Если test_data задан, добавляем его на график
                if test_data is not None and column in test_data.columns:
                    axs[r, c].plot(test_data.index.strftime('%b'),
                                   test_data[column],
                                   label = 'Test Data',
                                   linestyle = 'dashdot',
                                   linewidth = 3.0,
                                   color = 'blue')
                    list_of_y_label_min.append(min(test_data[column]))
                    list_of_y_label_max.append(max(test_data[column]))
                    # График для прогнозируемых значений
                    axs[r, c].plot(self.forecast_df.index.strftime('%b'), 
                                self.forecast_df[column], 
                                label = f'Forecast {self.forecast_df.index.year[0]}', 
                                linestyle = 'dashed',
                                linewidth = 2.0,
                                color = 'red')
                    list_of_y_label_min.append(min(list(self.forecast_df[column])))
                    list_of_y_label_max.append(max(list(self.forecast_df[column])))
                    y_label_min = min(list_of_y_label_min) #поиск минимального значения из всех минимумов по оси y
                    y_label_max = max(list_of_y_label_max) #поиск максимального значения из всех максимумов по оси y
                    axs[r, c].set_xlim(self.forecast_df.index.strftime('%b')[0], self.forecast_df.index.strftime('%b')[-1])
                    axs[r, c].set_ylim(y_label_min - 0.5, y_label_max + 0.5)

                    
                #Выводим на графике только прогнозируемый период
                axs[nrows - 1, c].set_xlabel('Месяц', fontsize = 18, color = 'black')
                axs[r, 0].set_ylabel('Share', fontsize = 18, color = 'black')
                axs[r, c].set_xticklabels(labels = self.forecast_df.index.strftime('%b'), rotation = 30)
                axs[r, c].legend(title = 'Год', loc = 'best', ncol = 3, fontsize = 9, frameon = False)

                if col_idx % ncols == 0:
                    r = r + 1 if r + 1 < nrows else 0
                c = c + 1 if c + 1 < ncols else 0
                    
            plt.tight_layout()
            file_path = os.path.join(save_dir, f'{col_idx}.png')
            #file_path = os.path.join(save_dir, f'{col_idx}_{uuid.uuid4()}.png')
            plt.savefig(file_path, dpi = 300)
            #with lock:
            #    try:
            #        plt.savefig(file_path, dpi=300)
            #    except Exception as e:
            #        print(f"Ошибка при сохранении графика: {e}")
            #    finally:
            #        plt.close(fig)
            # plt.savefig(file_path, dpi = 300)
            # plt.close(fig)


    @staticmethod
    def save_img_to_pdf(directory_path, img_format: str, output_filename):
        """
            Функция для записи в ЕДИНЫЙ PDF - файл всех изображений определённого формата, находящихся в указанной директории.
                Args:
                    directory_path: полный путь к директории, где находятся изображения,
                    img_format: формат изображений ((JPEG, PNG, TIF и тд.).
                    output_filename: Имя выходного файла/полный путь к его сохранению В ФОРМАТЕ PDF.
                Returns:
                    Файл в формате PDF с пулом всех изображений из указанной директории.
        """
        img_format_full = '.' + img_format
        # Лист всех файлов в директории с конкретным форматом картинок (JPEG, PNG, TIF и тд.)
        image_files = [f'{directory_path}{i}' for i in os.listdir(directory_path) if i.endswith(img_format_full)]
        
        # Конвертация списка изображений в отдельный PDF
        pdf_data = img2pdf.convert(image_files)
        
        # Запись контента в PDF - файл
        with open(f'{directory_path}{output_filename}', 'wb') as file:
            file.write(pdf_data)
    
    @staticmethod
    def generate_report(filename, report_filename, name_for_histogram, N = 364):
        """
            Генерация отчета с гистограммами и средними ошибками по второму прогнозируемому месяцу.
                Args:
                    filename: Имя входного файла с сырыми данными по ошибкам.
                    report_filename: Имя выходного файла с отчётом в формате xlsx.
                    name_for_histogram: Имя для гистограммы. Например, "Прогноз на 12 месяцев"
                    N = 364: общее число каналов-городов, участвующих в прогнозе.
                Returns:
                    Сформированный отчет, а также среднюю ошибку по второму прогнозируемому месяцу.
        """
    
        report = pd.read_excel(filename)
        #поиск максимально возможной строки в исходном DataFrame
        max_idx = max(report.index) + 2
        report.set_index('Date', inplace = True)
        report.rename(index = {'Mean': 'Среднее по каналу, %'}, inplace = True)
        
        #Поиск среднего значения по месяцу N+2
        average_value = np.round(np.mean(list(report.iloc[1])), 1)
        #Поиск дисперсии по месяцу N+2
        disp_value = np.round(np.sqrt(np.var(list(report.iloc[1]))), 1)

        #Подсчет количества значений по кажджому диапазону процентовок
        next_month = list(report.iloc[1])
        list_between_0_10 = []
        list_between_10_20 = []
        list_between_20_30 = []
        list_between_30_75 = []
        list_more_75 = []
        for i in next_month:
            if i >= 0 and i < 10:
                list_between_0_10.append(i)
            elif i >= 10 and i < 20:
                list_between_10_20.append(i)
            elif i >= 20 and i < 30:
                list_between_20_30.append(i)
            elif i >= 30 and i < 75:
                list_between_30_75.append(i)
            else:
                list_more_75.append(i)
        count_0_10 = len(list_between_0_10)
        count_10_20 = len(list_between_10_20)
        count_20_30 = len(list_between_20_30)
        count_30_75 = len(list_between_30_75)
        count_75 =  len(list_more_75)
        number_of_values = [count_0_10, count_10_20, count_20_30, count_30_75, count_75]
        
        percentage_of_all_channels = []
        for i in range(len(number_of_values)):
            percentage_of_all_channels.append(number_of_values[i] / N)

        #Формирование отчета
        writer = pd.ExcelWriter(report_filename, engine = 'xlsxwriter')
        report.to_excel(writer, sheet_name = 'Sheet1', startrow = 1, header = False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        chart = workbook.add_chart({'type': 'column'})
        
        number_cols = len(report.columns)
        
        #Стиль шапки таблицы
        header_format = workbook.add_format({'bold': True,
                                            'text_wrap': True, #перенос текста
                                            'align': 'center', #выравнение текста в ячейке
                                            'align': 'vcenter', #выравнение текста в ячейке
                                            'center_across': True
                                            })
        #Стиль тела таблицы
        table_fmt = workbook.add_format({'num_format': '0.0', 
                                        'align': 'center'}) #выравнение текста в ячейке
        #Условное форматирование
        #[0; 10]
        format1 = workbook.add_format({'bg_color': '#daf2cf',
                                    'font_color': '#333300'})
        #[10; 20]
        format2 = workbook.add_format({'bg_color': '#05ff82',
                                    'font_color': '#333300'})
        #[20, 30]
        format3 = workbook.add_format({'bg_color': '#ffeb9c',
                                    'font_color': '#9c5700'})
        
        #>30
        format4 = workbook.add_format({'bg_color': '#FFC7CE',
                                    'font_color': '#9C0006'})
        #>75
        format5 = workbook.add_format({'bg_color': 'black',
                                    'font_color': 'white'})
        
        #[0; 10]
        worksheet.conditional_format(f'B3:{xl_col_to_name(number_cols)}3', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  0,
                                                                            'maximum':  10,
                                                                            'format':   format1})
        
        worksheet.conditional_format(f'B{max_idx}:{xl_col_to_name(number_cols)}{max_idx}', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  0,
                                                                            'maximum':  10,
                                                                            'format':   format1})
        #[10; 20]
        worksheet.conditional_format(f'B3:{xl_col_to_name(number_cols)}3', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  10,
                                                                            'maximum':  20,
                                                                            'format':   format2})
        
        worksheet.conditional_format(f'B{max_idx}:{xl_col_to_name(number_cols)}{max_idx}', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  10,
                                                                            'maximum':  20,
                                                                            'format':   format2})
        #[20, 30]
        worksheet.conditional_format(f'B3:{xl_col_to_name(number_cols)}3', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  20,
                                                                            'maximum':  30,
                                                                            'format':   format3})
        
        worksheet.conditional_format(f'B{max_idx}:{xl_col_to_name(number_cols)}{max_idx}', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  20,
                                                                            'maximum':  30,
                                                                            'format':   format3})
        
        
        #>30
        worksheet.conditional_format(f'B3:{xl_col_to_name(number_cols)}3', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  30,
                                                                            'maximum':  75,
                                                                            'format':   format4})
        worksheet.conditional_format(f'B{max_idx}:{xl_col_to_name(number_cols)}{max_idx}', {'type': 'cell',
                                                                            'criteria': 'between',
                                                                            'minimum':  30,
                                                                            'maximum':  75,
                                                                            'format':   format4})
        #>75
        worksheet.conditional_format(f'B3:{xl_col_to_name(number_cols)}3', {'type': 'cell',
                                                                            'criteria': '>=',
                                                                            'value':  75,
                                                                            'format':   format5})
        
        worksheet.conditional_format(f'B{max_idx}:{xl_col_to_name(number_cols)}{max_idx}', {'type': 'cell',
                                                                            'criteria': '>=',
                                                                            'value':  75,
                                                                            'format':   format5})
        text_format = workbook.add_format({'font_size': 16, 
                                        'font_color': '#8000FF', 
                                        'italic': True, 
                                        'align':'left'})
        text_format_for_disp_avg = workbook.add_format({'num_format': '0.0',
                                                        'font_size': 20, 
                                                        'bg_color': 'yellow',
                                                        'font_color': 'black', 
                                                        'italic': True, 
                                                        'align':'center'})
        worksheet.merge_range(f'A{max_idx + 2}:B{max_idx + 2}', 'Средняя ошибка по месяцу', text_format)
        worksheet.merge_range(f'E{max_idx + 2}:F{max_idx + 2}', 'Дисперсия по месяцу', text_format)
        #worksheet.write(f'A{max_idx + 2}', 'Средняя ошибка по месяцу', text_format)
        worksheet.write_formula(f'C{max_idx + 2}', f'=AVERAGE(B3:{xl_col_to_name(number_cols)}3)', text_format_for_disp_avg)
        
        #worksheet.write(f'E{max_idx + 2}', 'Дисперсия по месяцу', text_format)
        worksheet.write(f'G{max_idx + 2}', disp_value, text_format_for_disp_avg)
        
        #Добавление таблицы для построения гистрограммы
        data = (
            #['Диапазон', 'Кол-во значений', '% от общего числа'],
            ['[0, 10)', count_0_10, count_0_10 / N],
            ['[10, 20)', count_10_20, count_30_75 / N],
            ['[20, 30)', count_20_30, count_20_30 / N],
            ['[30, 75)', count_30_75, count_30_75 / N],
            ['>=75', count_75, count_75 / N]  
        )
        
        row_number = max_idx + 4
        col_number = 1
        
        percent_fmt = workbook.add_format({'num_format': '0%', 'align':'center'})
        quant_of_values_fmt = workbook.add_format({'num_format': '0', 'align':'center'})
        worksheet.write(f'B{max_idx + 4}', 'Диапазон', header_format)
        worksheet.write(f'C{max_idx + 4}', 'Кол-во значений', header_format)
        worksheet.write(f'D{max_idx + 4}', '% от общего числа', header_format)
        for diap, quant_of_values, percent_from_all in data:
            worksheet.write(row_number, col_number, diap)
            worksheet.write(row_number, col_number + 1, quant_of_values, quant_of_values_fmt)
            worksheet.write(row_number, col_number + 2, percent_from_all, percent_fmt)
            row_number += 1
        
        chart.add_series({
            'categories': f'=Sheet1!$B${max_idx + 5}:$B${max_idx + 9}',
            'values': f'=Sheet1!$C${max_idx + 5}:$C${max_idx + 9}',
            'line':       {'color': 'blue'},
            'fill':   {'color': 'blue'},
            'data_labels': {'value': True, 'font':  {'name': 'Arial', 'size': 16}}
        })
        worksheet.insert_chart(f'I{max_idx + 2}', chart, {'x_scale': 1.6, 'y_scale': 1.5})
        chart.set_title({
        'name': name_for_histogram,
        'name_font': {
            'name': 'Arial',
            'color': 'black',
            'size': 24
        }})
        chart.set_x_axis({'num_font':  {'name': 'Arial', 'size': 14}})
        chart.set_y_axis({'major_gridlines': {'visible': False}, 'num_font':  {'name': 'Arial', 'size': 14}})
        chart.set_legend({'position': 'none'})
        
        #Форматирование шапки таблицы
        for col_num, value in enumerate(report.columns):
            worksheet.write(0, col_num + 1, value, header_format)
            
        
        worksheet.set_column('A:A', 17.5)
        worksheet.set_column(f'B2:{xl_col_to_name(number_cols)}{max_idx}', 14, table_fmt)
        workbook.close()
        return average_value

    