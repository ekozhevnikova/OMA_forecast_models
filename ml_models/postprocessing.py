import matplotlib
import numpy as np
import os
import img2pdf
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rc('font',family = 'Arial')



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
            error_df: DataFrame с абсолютными ошибками прогноза.
            mean_error: Средняя процентная ошибка.
        """
        # Вычисляем ошибки прогноза
        error_df = test_data - forecast_df
        print("Абсолютные ошибки прогноза:\n", error_df.abs())

        # Вычисляем процентные ошибки
        percentage_error = (1 - (forecast_df / test_data))
        mean_error = percentage_error.mean() * 100
        #mean_error = mean_error.astype(float)
        mean_error = (np.round(mean_error), 2)

        print("Средняя ошибка прогноза MAPE, %:\n", mean_error)
        return error_df, mean_error
    

    def get_plot(self, 
             column_name_with_date: str,
             save_dir, 
             last_n_list = None, 
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
                                    linewidth = 3.0,
                                    label = f'{last_n.index.year[0]}')
                        list_of_y_label_min.append(min(list(last_n[column])))
                        list_of_y_label_max.append(max(list(last_n[column])))
                    #построение графика для прогнозируемых значений
                    axs[r, c].plot(self.forecast_df.index.strftime('%b'), 
                                self.forecast_df[column], 
                                label = 'Forecast', 
                                linestyle = 'dashed',
                                linewidth = 2.0,
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
                    # График для прогнозируемых значений
                    axs[r, c].plot(self.forecast_df.index.strftime('%b'), 
                                self.forecast_df[column], 
                                label = 'Forecast', 
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
                axs[nrows - 1, c].set_xlabel('Месяц', fontsize = 12, color = 'black')
                axs[r, 0].set_ylabel('Share', fontsize = 12, color = 'black')
                axs[r, c].set_xticklabels(labels = self.forecast_df.index.strftime('%b'), rotation = 30)
                axs[r, c].legend(title = 'Год', loc = 'best', ncol = 2, fontsize = 7)

                if col_idx % ncols == 0:
                    r = r + 1 if r + 1 < nrows else 0
                c = c + 1 if c + 1 < ncols else 0
                    
            plt.tight_layout()
            file_path = os.path.join(save_dir, f'{col_idx}.png')
            plt.savefig(file_path, dpi = 300)
            plt.show()


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
        with open(f'{directory_path}output_filename', 'wb') as file:
            file.write(pdf_data)

    