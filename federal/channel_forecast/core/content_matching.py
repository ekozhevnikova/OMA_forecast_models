import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from OMA_tools.io_data.time_series import TimeSeriesTransformer

import warnings
warnings.filterwarnings('ignore')

# Скачиваем стоп-слова если нужно
nltk.download('stopwords')


class Find_Similarity:
    """
        Класс для поиска схожих ТВ-программ с помощью векторизатора TF-IDF
    """
    def __init__(self, List, small_list, df_big, small_df):
        """
            List: список, в котором будем искать похожие элементы.
            small_list: лист, для которого будем искать похожие элементы в списке List.
            df_big: Датафрейм с историческими данными
            small_df: Новый датафрейм, для которого будем искать схожие элементы
        """
        self.List = List
        self.small_list = small_list
        self.df_big = df_big
        self.small_df = small_df

    
    @staticmethod
    def clean_text(df, column_name):
        """
            Функция зачистки текста, избавление от пунктуации и лишних элементов. Например, если изначально было
            название 'Комедийный сериал (Сериал "Универ")' -> 'Универ'	
            Результат зачистки добавляется в список result. А в DataFrame df добавляется дополнительный столбец 
            под названием 'program_name', в который записываются причесанные названия ТВ-программ
            Args:
                df: DataFrame с данными, где присутствует столбец с названиями ТВ-программ
                column_name: название колонки в DataFrame df, в котором необходимо произвести зачистку текста. 
                             Обычно это колонка, которая содержит названия ТВ-программ
            Returns:
                result: list 'причёсанных' программ
                df: обновленный DataFrame с новым столбцом
        """
        result = []
        set_of_programs = list(set(list(df[column_name])))

        df['program_name'] = ''

        for program in set_of_programs:
            if '"' in program:
                # Разделяем по кавычкам и берем второй элемент (между кавычками)
                parts = program.split('"')
                if len(parts) >= 3:
                    name = parts[1].strip()
                    name_cleaned = Find_Similarity.preprocess_text(name)
                    result.append(name_cleaned)
                    df.loc[df['Название программы'] == program, 'program_name'] = name_cleaned
            else:
                # Если кавычек нет, добавляем название как есть
                name = Find_Similarity.preprocess_text(program)
                result.append(name)
                df.loc[df['Название программы'] == program, 'program_name'] = name
        return result, df

    
    @staticmethod
    def preprocess_text(text: str):
        """
            Функция предобработки текста. Текст приводится к нижнему регистру, удаляются спец символы, пунктуация и пробелы.
            Args:
                text: str: Текст типа данных строка
            Returns:
                text_delete_tab: причёсанный текст
        """
        # Приводим к нижнему регистру
        text_lowered = text.lower()
        # Удаляем специальные символы, оставляем только буквы и пробелы
        text_cleaned = re.sub(r'[^а-яёa-z\s]', '', text_lowered)
        # Удаляем пунктуацию
        text_delete_punc = re.sub(r'[^\w\s]', '', text_cleaned) 
        # Удаляем лишние пробелы
        text_delete_tab = re.sub(r'\s+', ' ', text_delete_punc).strip()
        return text_delete_tab


    def compare_uneven_lists_tfidf(self, preprocess = True):
        """
            Сравнивает все элементы первого списка со всеми элементами второго списка, считается косинусное сходство для
            каждой пары. 
            Args:
                preprocess: Флаг: если True -> используем функцию preprocess_text, если False -> не используем функцию preprocess_text
            Returns:
                numpy.ndarray: Матрица схожести shape (len(list1), len(list2))
        """
        if preprocess:
            processed_list1 = [Find_Similarity.preprocess_text(text) for text in self.List]
            processed_list2 = [Find_Similarity.preprocess_text(text) for text in self.small_list]
        else:
            processed_list1 = self.List
            processed_list2 = self.small_list
        
        # Объединяем все тексты
        all_texts = processed_list1 + processed_list2
        
        # Создаем и обучаем TF-IDF векторзатор
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Разделяем матрицу
        tfidf_list1 = tfidf_matrix[: len(processed_list1)]
        tfidf_list2 = tfidf_matrix[len(processed_list1): ]
        
        # Сравниваем все со всеми
        similarity_matrix = cosine_similarity(tfidf_list1, tfidf_list2)
        return similarity_matrix


    def comparison(self, 
                   column_name_first: str = 'Palomars',
                   column_name_second: str = 'VIMB',
                   min_similarity: float = 0.2, 
                   top_n: int = 10, 
                   max_pairs = None, 
                   print_in_console = False):
        """
            Сравнение массивов с фильтрацией.
            Args:
                min_similarity: порог минимальной схожести.
                top_n: максимальное количество схожих пар, выводимых на экран. По дефолту 10.
                max_pairs: ограничение на максимальное количество схожих пар. Если None, то игнорируем этот параметр.
                print_in_console: флаг для вывода наиболее схожих пар в консоль. Если True: выводим в консоль. В противном случае нет.
            Returns
                data_unique: DataFrame, в котором приведены максимально схожие элементы в соответствии с порогом min_similarity. 
                             Если схожие элементы не найдены, то заполняем similarity нулями.
        """
        similarity_matrix = self.compare_uneven_lists_tfidf()
        
        # Собираем все пары выше порога
        pairs = []
        for i in range(len(self.List)):
            for j in range(len(self.small_list)):
                if similarity_matrix[i, j] >= min_similarity:
                    pairs.append((i, j, similarity_matrix[i, j]))
        
        # Сортируем по убыванию схожести
        pairs.sort(key = lambda x: x[2], reverse = True)

        if print_in_console:
            #Вывод ТОП N схожих пар в консоль
            print(f"Топ - {top_n} наиболее похожих пар:")
            print("-" * 80)
            
            for i, match in enumerate(pairs[: top_n]):
                print(f"{i + 1}. Схожесть: {match['similarity']:.3f}")
                print(f"   Список 1: '{match['text1']}'")
                print(f"   Список 2: '{match['text2']}'")
                print()
        
        # Ограничиваем количество пар если нужно
        if max_pairs:
            pairs = pairs[: max_pairs]
        
        # Форматируем результаты
        results = []
        for i, j, similarity in pairs:
            results.append({
                f'Программа {column_name_first}': self.List[i],
                f'Программа {column_name_second}': self.small_list[j],
                'similarity': similarity,
                f'index_{column_name_first}': i,
                f'index_{column_name_second}': j
            })
        data = pd.DataFrame(results)
    
        programs = list(set(data[f'Программа {column_name_second}']))
        #Удаляем дубликаты. Оставляем только те программы из дублец, для которых найдено максимальное сходство
        cleaned_results = []
        for i in range(len(programs)):
            df = data.loc[data[f'Программа {column_name_second}'] == programs[i]]
            #Если найдена только одна строка без дубликатов, то значит для программы скорее всего нашлось точное соответствие
            if len(df) == 1:
                cleaned_results.append(df)
            #Если найдено несколько похожих программ для выбранной, то оставляем только максимально похожую
            else:
                cleaned_results.append(df.loc[df['similarity'] == max(list(df['similarity']))])
        final = pd.concat(cleaned_results)
        
        #Встречаются ситуации, когда показатель similarity одинаковыйи выбрать максимальный не удается. 
        #Например, программа посчитала "Красавчик" == "Красавчик" и "Красавчик 2" == "Красавчик"
        #Воспользуемся удалением дубликатов
        data_unique = final.drop_duplicates(subset = [f'Программа {column_name_second}', 'similarity'], 
                                            keep = 'first').reset_index(drop = True)
        
        #Программы, для которых не нашлось похожих, в столбец схожести пишем 0
        programs_found = list(data_unique[f'Программа {column_name_second}'])
        programs_not_found = []
        for i in range(len(self.small_list)):
            if self.small_list[i] not in programs_found:
                programs_not_found.append(self.small_list[i])
                data_unique.loc[len(data_unique) + 1] = [0, self.small_list[i], 0.0, 0, 0]
        return data_unique
    

    def generate_similar_features(self, similarity_df, print_df = False) -> dict:
        """
            Функция для генерации совокупных Датасетов из исторической и новой ТВ-сеток на основе данных косинусного сходства
            между программами. В ходе работы функции вычисляется матрица схожести, составляется таблица
            Args:
                df_big: DataFrame, в котором будем искать схожие программы.
                df_small: DataFrame, для которого будем искать схожие программы.
                similarity_df: DataFrame из схожих программ
                print_df: флаг на вывод длин датафреймов. Если True, выводим. В противном случае нет.
            Return:
                dict_analysis: словарь, где ключ: название программы, значение: датафрейм, составленный из исторической и новой сеток.
        """
        #Отбираем колонки в исторической сетке Palomars
        plmrs_analysis = self.df_big[['Дата', 'program_name', 
                                    'Время выхода', 'Время окончания', 
                                    'Share']]
        plmrs_analysis = plmrs_analysis.rename(columns = {'program_name': 'Название программы'})

        #Отбираем колонки в новой сетке VIMB
        vimb_analysis = self.small_df[['Дата', 'program_name', 
                            'Время выхода', 'Время окончания'
                            ]]
        vimb_analysis['Share'] = ''
        vimb_analysis = vimb_analysis.rename(columns = {'program_name': 'Название программы'})

        #Приведение названий программ к нижнему регистру
        vimb_analysis['Название программы'] = vimb_analysis['Название программы'].str.lower()
        plmrs_analysis['Название программы'] = plmrs_analysis['Название программы'].str.lower()


        dict_analysis = {}

        for i in range(len(similarity_df)):
            #Название программы в Palomars
            program_plrms = similarity_df.iloc[i]['Программа Palomars']
            #Название программы в VIMB
            program_vimb = similarity_df.iloc[i]['Программа VIMB']
            vimb_df = vimb_analysis.loc[vimb_analysis['Название программы'] == program_vimb]
            #Если названия Palomars и VIMB не совпадабт, то заменяем название в VIMB на название Palomars
            if program_vimb != program_plrms:
                vimb_df['Название программы'].replace(program_vimb, program_plrms, inplace = True) 
            palomars = plmrs_analysis.loc[plmrs_analysis['Название программы'] == program_plrms]
            full_data = pd.concat([palomars, vimb_df]).reset_index(drop = True)
            full_data['Дата'] = pd.to_datetime(full_data['Дата'])
            #print(TimeSeriesTransformer.find_missing_dates(list(full_data['Дата'])))
            #full_data_new = TimeSeriesTransformer.create_reverse_dates_from_target(full_data, 'Дата', 'Share')
            dict_analysis[program_plrms] = full_data

        if print_df:
            for program, data in dict_analysis.items():
                print(f'{program}: длина DataFrame {len(data)}')
                print('#########################')
        return dict_analysis