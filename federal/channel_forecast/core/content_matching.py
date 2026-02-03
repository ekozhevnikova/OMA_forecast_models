import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
#import nltk
#from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')

# Скачиваем стоп-слова если нужно
#nltk.download('stopwords')


class TextPreprocessor:
    """
        Класс для предобработки названий телепрограмм.
        Отвечает только за чистку и нормализацию названий.
    """

    def __init__(self):
        # КОНСТАНТЫ

        # Удаляем обратный слеш и другие специальные символы
        self.SPECIAL_CHARS_TO_REMOVE = [
                '\\', '/', '|', ':', ';', '*', '?', '<', '>', '~', '`',
                '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', 
                '=', '[', ']', '{', '}', '-', '№'
            ]
        
        # Список из ограничений по возрасту
        self.YEAR_CONDITIONS = ['0+', '6+', '12+']

        # Список специальных названий
        self.SPECIAL_NAMES = {
            'сериал', 'комедийный сериал', 'художественный фильм', 'мультфильм', 'мультфильмы', 'мультфильм 0+'
            }

        # Список из стоп-слов
        self.STOP_WORDS = {
            'анимационный', 'мультфильм', 'сериал', 'художественный', 'фильм', 'х/ф', 'm/ф', 'р/б'
        }


    def _clean_special_chars(self, text: str):
        """
            Удаляет специальные символы из текста.
        """
        cleaned = text
        for char in self.SPECIAL_CHARS_TO_REMOVE:
            cleaned = cleaned.replace(char, '')
        
        # Убираем лишние пробелы, которые могли появиться после удаления символов
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Удаляем точки, запятые и другие знаки препинания в конце строки
        cleaned = re.sub(r'[.,;:!?\-_]+$', '', cleaned).strip()
        
        # Удаляем двойные и одинарные кавычки (если остались)
        cleaned = cleaned.replace('"', '').replace("'", '')
        
        return cleaned


    def _normalize_match(self, match):
        """
            Нормализует текст внутри скобок в строке, удаляя лишние пробелы вокруг содержимого скобок.
        """
        # Содержимое скобок
        content = match.group(1).strip()  # убираем пробелы с обеих сторон
        return f'({content})'


    def _extract_and_clean_title(self, text: str):
        """
            Извлекает и очищает название из текста, заключенного в кавычки, а если кавычек нет - удаляет скобки.
        """
        # Используем нежадный поиск для нахождения первых кавычек
        match = re.search(r'"([^"]*(?:"[^"]*"[^"]*)*)"', text)
        if match:
            title = match.group(1)
            # Удаляем все кавычки внутри названия
            title = title.replace('"', '')
            title = re.sub(r'\s+', ' ', title).strip()
            return title
        return re.sub(r'[()]', ' ', text)
    

    def _replace_whole_word(self, text: str, old_word: str, new_word: str):
        """
            Заменяет подстроки. Например "Комеди Клаб" -> "Камеди Клаб"
        """
        # Используем регулярное выражение с границами слов
        pattern = r'\b' + re.escape(old_word) + r'\b'
        return re.sub(pattern, new_word, text)


    def _remove_stop_words(self, text: str) -> str:
        """
            Удаляет стоп-слова из текста.
        """
        result = text
        for stop_word in self.STOP_WORDS:
            # Заменяем только целые слова
            pattern = r'\b' + re.escape(stop_word) + r'\b'
            result = re.sub(pattern, '', result, flags = re.IGNORECASE)
        return result


    def _remove_year_conditions(self, text: str) -> str:
        """
            Удаляет годовые ограничения из текста.
        """
        result = text
        for condition in self.YEAR_CONDITIONS:
            result = result.replace(condition, "")
        return result.strip()


    def _process_brackets_no_quotes(self, text: str) -> str:
        """
            Обработка текста со скобками, но без кавычек.
        """
        # 1. Находим все пары скобок и их содержимое
        pattern = r'\(([^)]+)\)'
        # Применяем нормализацию ко всем скобкам (удаляем пробелы после первой скобки и перед второй скобкой)
        normalized = re.sub(pattern, self._normalize_match, text)
        
        # 2. Заменяем скобки на пробелы
        text_simplified = re.sub(r'[()]', ' ', normalized)
        
        # 3. Убираем лишние пробелы
        text_simplified = re.sub(r'\s+', ' ', text_simplified)
        
        # 4. Убираем пробелы вначале и в конце строки
        text_simplified = text_simplified.strip()
        
        # 5. Удаляем стоп-слова
        new_str = self._remove_stop_words(text_simplified)
        
        # 6. Удаляем годовые условия
        new_str = self._remove_year_conditions(new_str)
        return self._clean_special_chars(new_str)


    def _process_brackets_with_quotes(self, text: str) -> str:
        """
            Обработка текста со скобками и кавычками.
        """
        # 1. Находим все пары скобок и их содержимое
        pattern = r'\(([^)]+)\)'
        # Применяем нормализацию ко всем скобкам (удаляем пробелы после первой скобки и перед второй скобкой)
        simplified = re.sub(pattern, self._normalize_match, text)
        
        # 2. Удаляем стоп-слова
        result = self._remove_stop_words(simplified)
        
        # 3. Удаляем лишние пробелы
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 4. Удаляем скобки
        result = re.sub(r'[()]', ' ', result)
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 5. Извлекаем текст из кавычек
        result = self._extract_and_clean_title(result)
        
        # 6. Удаляем годовые условия
        result = self._remove_year_conditions(result)
        
        return self._clean_special_chars(result)
    

    def _process_simple_text(self, text: str) -> str:
        """
            Обработка простого текста без скобок и кавычек.
        """
        result = text
    
        # 1. Удаляем стоп-слова
        result = self._remove_stop_words(result)
        
        # 2. Проверяем, не остался ли только стоп-слово
        result = result.strip()
        if result.lower() in (word.lower() for word in self.STOP_WORDS):
            result = ""
        
        # 3. Нормализуем пробелы
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 4. Удаляем годовые условия
        result = self._remove_year_conditions(result)
        
        return self._clean_special_chars(result)


    def preprocess_text(self, text: str):
        """
            Отдельный поток для стандартной обработки скобок/кавычек.
        """
        # --- НОВЫЙ БЛОК: Обработка мультфильмов со списком в скобках ---
        # Например: 'Мультфильмы (M/ф "Братья Лю"; M/ф "На задней парте"; M/ф "Кошкин дом"; M/ф "Муравьишка-хвастунишка"; M/ф "Тайна далекого острова")'
        # Проверяем, является ли строка списком мультфильмов в скобках
        # Паттерн: Название + (M/ф "фильм1"; M/ф "фильм2"; ...)
        # 1. Проверка на список мультфильмов/фильмов в скобках
        if re.search(r'(?:мультфильмы|фильмы|сериалы)\s*\([^)]*(?:M/ф|х/ф|фильм|сериал)[^)]*\)', 
                    text, re.IGNORECASE):
            # Определяем тип по первому слову
            first_word = text.split()[0].lower()
            if 'мультфильм' in first_word:
                return 'мультфильм'
            elif 'сериал' in first_word:
                return 'сериал'
            elif 'фильм' in first_word:
                return 'фильм'
            else:
                return first_word
        
        # 2. Проверка на специальные имена
        if text.lower() in (name.lower() for name in self.SPECIAL_NAMES):
            return self._clean_special_chars(
                self._remove_year_conditions(text)
        )

        # --- Основная обработка по типам ---
    
        has_brackets = '(' in text and ')' in text      # есть скобки
        has_quotes = '"' in text                        # есть кавычки
        
        if has_brackets and not has_quotes:

            # Случай 1: Скобки без кавычек
            return self._process_brackets_no_quotes(text)
        
        elif has_brackets and has_quotes:
            # Случай 2: Скобки с кавычками
            return self._process_brackets_with_quotes(text)
        
        else:
            # Случай 3: Нет ни скобок, ни кавычек
            return self._process_simple_text(text)
    


    def clean_text(self, df, column_name):
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

            original = program
            # Заменяем букву ё на е при необходимости
            text = original.lower().replace('ё', 'е')

            # Умышленно заменяем 'комеди' на 'камеди'
            if 'комеди' in text:
                text = self._replace_whole_word(text, 'комеди', 'камеди')
                
            new_str = self.preprocess_text(text)

            # Сначала нормализуем: удаляем точки и скобки
            cleaned = new_str.replace(".", "").replace("(", "").replace(")", "")
            parts = cleaned.split()
            unique_list = []
            [unique_list.append(x) for x in parts if x not in unique_list]
            res_str = " ".join(unique_list)

            
            result.append(res_str)
            df.loc[df['Название программы'] == program, 'program_name'] = res_str

        return result, df




class CosineSimilarity:
    """
        Класс для поиска схожих ТВ-программ с помощью TF-IDF.
        Отвечает только за сравнение и анализ схожести.
    """
    def __init__(
            self, 
            List: list, 
            small_list: list, 
            df_big: pd.DataFrame, 
            small_df: pd.DataFrame, 
            preprocessor = None
            ):
        """
            Args:
                preprocessor: экземпляр TextPreprocessor (или None для использования по умолчанию)
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        self.vectorizer = TfidfVectorizer()

        self.List = List
        self.small_list = small_list
        self.df_big = df_big
        self.small_df = small_df
    

    def compare_lists(self, preprocess: bool = True):
        """
            Сравнивает два списка текстов.
        """
        if preprocess:
            processed_list1 = [self.preprocessor.preprocess_text(text) for text in self.List]
            processed_list2 = [self.preprocessor.preprocess_text(text) for text in self.small_list]
        else:
            processed_list1 = self.List
            processed_list2 = self.small_list
        
        all_texts = processed_list1 + processed_list2
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        tfidf_list1 = tfidf_matrix[:len(processed_list1)]
        tfidf_list2 = tfidf_matrix[len(processed_list1):]
        
        return cosine_similarity(tfidf_list1, tfidf_list2)


    def comparison(self, 
                   vocabulary: pd.DataFrame,
                   column_name_first: str = 'Palomars',
                   column_name_second: str = 'VIMB',
                   min_similarity: float = 0.5, 
                   top_n: int = 10, 
                   max_pairs = None, 
                   print_in_console = False):
        """
            Сравнение массивов с фильтрацией.
            Args:
                vocabulary: pd.DataFrame: таблица-справочник для сопоставления программ.
                min_similarity: порог минимальной схожести.
                top_n: максимальное количество схожих пар, выводимых на экран. По дефолту 10.
                max_pairs: ограничение на максимальное количество схожих пар. Если None, то игнорируем этот параметр.
                print_in_console: флаг для вывода наиболее схожих пар в консоль. Если True: выводим в консоль. В противном случае нет.
            Returns
                data_unique: DataFrame, в котором приведены максимально схожие элементы в соответствии с порогом min_similarity. 
                             Если схожие элементы не найдены, то заполняем similarity нулями.
        """
        similarity_matrix = self.compare_lists()
    
        # Собираем все пары выше порога
        pairs = []
        for i in range(len(self.List)):
            for j in range(len(self.small_list)):
                if similarity_matrix[i, j] >= min_similarity:
                    pairs.append((i, j, similarity_matrix[i, j]))
        
        # Сортируем по убыванию схожести
        pairs.sort(key=lambda x: x[2], reverse=True)

        if print_in_console:
            #Вывод ТОП N схожих пар в консоль
            print(f"Топ - {top_n} наиболее похожих пар:")
            print("-" * 80)
            
            for i, match in enumerate(pairs[:top_n]):
                print(f"{i + 1}. Схожесть: {match['similarity']:.3f}")
                print(f"   Список 1: '{match['text1']}'")
                print(f"   Список 2: '{match['text2']}'")
                print()
        
        # Ограничиваем количество пар если нужно
        if max_pairs:
            pairs = pairs[:max_pairs]
        
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
        
        # Удаляем дубликаты. Оставляем только те программы из дубликатов, для которых найдено максимальное сходство
        cleaned_results = []
        for i in range(len(programs)):
            df = data.loc[data[f'Программа {column_name_second}'] == programs[i]]
            if len(df) == 1:
                cleaned_results.append(df)
            else:
                cleaned_results.append(df.loc[df['similarity'] == max(list(df['similarity']))])
        
        final = pd.concat(cleaned_results) if cleaned_results else pd.DataFrame()
        
        # Встречаются ситуации, когда показатель similarity одинаковый и выбрать максимальный не удается
        data_unique = final.drop_duplicates(
            subset=[f'Программа {column_name_second}', 'similarity'], 
            keep='first'
        ).reset_index(drop=True) if not final.empty else pd.DataFrame()
        
        # Программы, для которых не нашлось похожих, в столбец схожести пишем 0
        programs_found = list(data_unique[f'Программа {column_name_second}']) if not data_unique.empty else []
        programs_not_found = []
        
        for i in range(len(self.small_list)):
            if self.small_list[i] not in programs_found:
                programs_not_found.append(self.small_list[i])
                new_row = {
                    f'Программа {column_name_first}': 0,
                    f'Программа {column_name_second}': self.small_list[i],
                    'similarity': 0.0,
                    f'index_{column_name_first}': 0,
                    f'index_{column_name_second}': 0
                }
                
                if data_unique.empty:
                    data_unique = pd.DataFrame([new_row])
                else:
                    data_unique = pd.concat([data_unique, pd.DataFrame([new_row])], ignore_index = True)
        
        # Отбираем найденные программы
        found_programs = data_unique[data_unique['similarity'] != 0.00000].reset_index(drop = True)

        # Отбираем ненайденные программы
        not_found = data_unique[data_unique['similarity'] == 0.00000].reset_index(drop = True)

        # Пытаемся найти совпадающие программы, основываясь на данных справочника
        not_found_updated = pd.merge(not_found, vocabulary, on = 'Программа VIMB', how = 'left')
        not_found_updated = not_found_updated[['Программа', 'Программа VIMB', 'similarity_new', 'index_Palomars', 'index_VIMB']]
        not_found_updated.rename(columns = {'Программа': 'Программа Palomars', 'similarity_new': 'similarity'}, inplace = True)
        not_found_updated = not_found_updated.fillna(0)

        result_df = pd.concat([found_programs, not_found_updated]).reset_index(drop = True)

        # Отбираем финальные программы без соответствия
        result_not_found = result_df[result_df['similarity'] == 0.00000].reset_index(drop = True)
        programs_not_found = list(result_not_found['Программа VIMB'])

        # Отбираем финальные программы c соответствием
        result_found = result_df[result_df['similarity'] != 0.00000].reset_index(drop = True)
        programs_found = list(result_found['Программа VIMB'])

        
        # ДОБАВЛЕННАЯ ПРОВЕРКА: все ли программы нашли соответствия
        all_programs_matched = len(programs_not_found) == 0
        
        # Вывод информации о результатах сопоставления
        print(f"\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
        print(f"Всего программ для поиска: {len(self.small_list)}")
        print(f"Найдено соответствий (similarity >= {min_similarity}): {len(programs_found)}")
        print(f"Не найдено соответствий: {len(programs_not_found)}")
        
        if programs_not_found:
            print(f"Программы без соответствий: {programs_not_found}")
        
        if all_programs_matched:
            print("✓ УСПЕХ: Для всех программ найдены соответствия (хотя бы с минимальной схожестью)")
        else:
            print("⚠ ВНИМАНИЕ: Не для всех программ найдены соответствия")
        
        # Возвращаем и DataFrame, и флаг успешности
        return data_unique, all_programs_matched
    


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
            dict_analysis[program_plrms] = full_data

        if print_df:
            for program, data in dict_analysis.items():
                print(f'{program}: длина DataFrame {len(data)}')
                print('#########################')
        return dict_analysis