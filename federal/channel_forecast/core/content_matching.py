import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import fuzz, process
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
            'сериал', 'комедийный сериал', 'художественный фильм', 'документальный фильм',
            'мультфильм', 'мультфильмы', 'мультфильм 0+', 'мультсериал',  'худ. фильм',
            }

        # Список из стоп-слов
        self.STOP_WORDS = {
            'анимационный', 'мультфильм', 'сериал', 'художественный', 'фильм', 'мультсериал',  'огр', 'х/ф', 'm/ф', 'р/б'
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
        # Специальная обработка для "Мультфильм (M/ф "Фильм, фильм, фильм.")"
        if 'фильм, фильм, фильм' in text.lower():
            # Убираем точку в конце если есть
            if text.lower().endswith('фильм, фильм, фильм.")'):
                return 'фильм, фильм, фильм'
            elif text.lower().endswith('фильм, фильм, фильм)'):
                return 'фильм, фильм, фильм'
            else:
                return 'фильм, фильм, фильм'
            
        # --- НОВЫЙ БЛОК: Обработка мультфильмов со списком в скобках ---
        # Например: 'Мультфильмы (M/ф "Братья Лю"; M/ф "На задней парте"; M/ф "Кошкин дом"; M/ф "Муравьишка-хвастунишка"; M/ф "Тайна далекого острова")'
        # Проверяем, является ли строка списком мультфильмов в скобках
        # Паттерн: Название + (M/ф "фильм1"; M/ф "фильм2"; ...)
        # 1. Проверка на список мультфильмов/фильмов в скобках
        if re.search(r'(?:мультфильмы|мультфильм|фильмы|сериалы)\s*\([^)]*(?:M/ф|х/ф|фильм|сериал)[^)]*\)', 
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

            new_str = self._clean_special_chars(self._remove_year_conditions(text))
            if new_str == 'мультфильмы':
                return 'мультфильм'
            
            elif new_str == 'худ. фильм':
                return 'художественный фильм'

            else:
                return new_str

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
        # Создаем словарь для отслеживания всех обработок
        processing_dict = {}
        set_of_programs = list(set(list(df[column_name].dropna())))  # Убираем NaN значения
        
        #print(f"Всего уникальных названий для обработки: {len(set_of_programs)}")
        
        df['program_name'] = ''
        
        for program in set_of_programs:
            if not isinstance(program, str):
                # Преобразуем нестроковые значения в строку
                program = str(program)
                
            original = program
            original_key = original  # Сохраняем оригинал как ключ
            
            # Заменяем букву ё на е при необходимости
            text = original.lower().replace('ё', 'е')

            # Умышленно заменяем 'комеди' на 'камеди'
            if 'комеди' in text:
                text = self._replace_whole_word(text, 'комеди', 'камеди')
                
            new_str = self.preprocess_text(text)


            # Сначала нормализуем: удаляем точки и скобки
            if new_str != 'фильм, фильм, фильм':
                cleaned = new_str.replace(".", "").replace("(", "").replace(")", "")
                parts = cleaned.split()
                unique_list = []
                [unique_list.append(x) for x in parts if x not in unique_list]
                new_str = " ".join(unique_list)
            

            if program == '1 + 1':
                print(new_str)
            

            if new_str == '':
                print(program)
            
    
            # Сохраняем результат обработки в словарь
            processing_dict[original_key] = {
                'original': original,
                'cleaned': new_str,
                'is_empty': not new_str.strip()
            }
            
            # Присваиваем обработанное значение в DataFrame
            df.loc[df[column_name] == original_key, 'program_name'] = new_str
        
        # Теперь формируем result, гарантируя сохранение порядка и всех элементов
        result = []
        failed_parses = []
        
        for program in set_of_programs:
            if not isinstance(program, str):
                program = str(program)
                
            if program in processing_dict:
                cleaned_value = processing_dict[program]['cleaned']
                result.append(cleaned_value)
                
                if processing_dict[program]['is_empty']:
                    failed_parses.append(program)
            else:
                # На всякий случай, если что-то потерялось
                result.append('')
                failed_parses.append(program)
                #print(f"ВНИМАНИЕ: Название '{program}' отсутствует в словаре обработки!")

        return result, df, processing_dict




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


    def _fuzzy_search(self, query: str, choices: list, threshold: int = 80):
        """
        Поиск лучшего совпадения с использованием fuzzywuzzy.
        
        Args:
            query: строка для поиска.
            choices: список строк для сравнения.
            threshold: минимальный порог совпадения (0-100).
            
        Returns:
            tuple: (best_match, score, index) или (None, 0, -1) если совпадений нет.
        """
        if not choices:
            return None, 0, -1
            
        # Используем fuzzywuzzy для поиска лучшего совпадения
        result = process.extractOne(
            query = query,
            choices = choices,
            score_cutoff = threshold,
            scorer = fuzz.token_sort_ratio  # Используем token_sort_ratio для лучшего сравнения
        )
        
        if result:
            best_match, score = result
            # Находим индекс лучшего совпадения
            index = choices.index(best_match) if best_match in choices else -1
            return best_match, score, index
        
        return None, 0, -1


    def comparison(self, vocabulary: pd.DataFrame = None,
               column_name_first: str = 'Palomars',
               column_name_second: str = 'VIMB',
               min_similarity: float = 0.5,
               max_pairs = None,
               print_in_console = False,
               use_fuzzy_backup: bool = True,
               fuzzy_threshold: int = 70,
               use_vocabulary: bool = False):
        """
        Сравнение массивов с фильтрацией и резервным fuzzy matching.
        
        Args:
            vocabulary: таблица-справочник для сопоставления программ. Может быть None.
            column_name_first: название первой колонки.
            column_name_second: название второй колонки.
            min_similarity: порог минимальной схожести для TF-IDF.
            max_pairs: ограничение на максимальное количество пар.
            print_in_console: флаг для вывода в консоль.
            use_fuzzy_backup: использовать ли fuzzy matching для ненайденных программ.
            fuzzy_threshold: минимальный порог схожести для fuzzy matching (0-100).
            use_vocabulary: использовать ли справочник для поиска совпадений.
                
        Returns:
            DataFrame с результатами сравнения и флаг успешности.
        """
        # Основная логика TF-IDF сравнения
        similarity_matrix = self.compare_lists()
        
        # Собираем все пары выше порога
        pairs = []
        for i in range(len(self.List)):
            for j in range(len(self.small_list)):
                if similarity_matrix[i, j] >= min_similarity:
                    pairs.append((i, j, similarity_matrix[i, j]))
        
        # Сортируем по убыванию схожести
        pairs.sort(key=lambda x: x[2], reverse=True)
        
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
                f'index_{column_name_second}': j,
                'method': 'tfidf'
            })
        
        data = pd.DataFrame(results)
        
        # Удаляем дубликаты. Оставляем только те программы из дубликатов, для которых найдено максимальное сходство
        programs = list(set(data[f'Программа {column_name_second}'])) if not data.empty else []
        
        cleaned_results = []
        for i in range(len(programs)):
            df = data.loc[data[f'Программа {column_name_second}'] == programs[i]]
            if len(df) == 1:
                cleaned_results.append(df)
            else:
                cleaned_results.append(df.loc[df['similarity'] == max(list(df['similarity']))])
        
        final = pd.concat(cleaned_results) if cleaned_results else pd.DataFrame()
        
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
        
        # ДОПОЛНЕНИЕ: Fuzzy matching для ненайденных программ
        if use_fuzzy_backup and programs_not_found:
            fuzzy_results = []
            programs_to_remove = []
            
            for program in programs_not_found:
                # Ищем лучшее совпадение с помощью fuzzy matching
                best_match, score, index = self._fuzzy_search(
                    query=program,
                    choices=self.List,
                    threshold=fuzzy_threshold
                )
                
                if best_match:
                    # Находим индекс программы в small_list
                    small_index = self.small_list.index(program)
                    
                    fuzzy_results.append({
                        f'Программа {column_name_first}': best_match,
                        f'Программа {column_name_second}': program,
                        'similarity': score / 100.0,  # Приводим к шкале 0-1
                        f'index_{column_name_first}': index,
                        f'index_{column_name_second}': small_index,
                        'method': 'fuzzy'
                    })
                    programs_to_remove.append(program)
            
            # Удаляем найденные через fuzzy программы из списка ненайденных
            for program in programs_to_remove:
                if program in programs_not_found:
                    programs_not_found.remove(program)
            
            # Добавляем fuzzy результаты к основным
            if fuzzy_results:
                fuzzy_df = pd.DataFrame(fuzzy_results)
                if data_unique.empty:
                    data_unique = fuzzy_df
                else:
                    data_unique = pd.concat([data_unique, fuzzy_df], ignore_index=True)
        
        # Добавляем программы, которые так и не нашли (ни TF-IDF, ни fuzzy)
        for program in programs_not_found:
            small_index = self.small_list.index(program)
            new_row = {
                f'Программа {column_name_first}': 0,
                f'Программа {column_name_second}': program,
                'similarity': 0.0,
                f'index_{column_name_first}': 0,
                f'index_{column_name_second}': small_index,
                'method': 'none'
            }
            
            if data_unique.empty:
                data_unique = pd.DataFrame([new_row])
            else:
                data_unique = pd.concat([data_unique, pd.DataFrame([new_row])], ignore_index=True)
        
        # Разделяем найденные и ненайденные программы
        found_programs = data_unique[data_unique['similarity'] != 0.0].reset_index(drop=True)
        not_found = data_unique[data_unique['similarity'] == 0.0].reset_index(drop=True)
        
        # Используем справочник только если он предоставлен и use_vocabulary=True
        if use_vocabulary and vocabulary is not None:
            # Пытаемся найти совпадающие программы, основываясь на данных справочника
            not_found_updated = pd.merge(
                not_found, 
                vocabulary, 
                on=f'Программа {column_name_second}', 
                how='left'
            )
            
            # Проверяем наличие необходимых колонок в vocabulary
            if 'Программа' in not_found_updated.columns and 'similarity_new' in not_found_updated.columns:
                not_found_updated = not_found_updated[[
                    'Программа', 
                    f'Программа {column_name_second}', 
                    'similarity_new', 
                    f'index_{column_name_first}', 
                    f'index_{column_name_second}'
                ]]
                not_found_updated.rename(
                    columns={'Программа': f'Программа {column_name_first}', 'similarity_new': 'similarity'}, 
                    inplace=True
                )
                not_found_updated = not_found_updated.fillna(0)
            else:
                # Если колонок нет, оставляем как есть
                not_found_updated['similarity'] = 0.0
            
            # Собираем финальный результат с использованием справочника
            result_df = pd.concat([found_programs, not_found_updated]).reset_index(drop=True)
        else:
            # Если справочник не используется, просто объединяем найденные и ненайденные
            result_df = pd.concat([found_programs, not_found]).reset_index(drop=True)
            
            if not_found.empty:
                print("Справочник не используется. Ненайденные программы остаются без соответствий.")
        
        # Отбираем финальные программы без соответствия
        result_not_found = result_df[result_df['similarity'] == 0.0].reset_index(drop=True)
        programs_not_found = list(result_not_found[f'Программа {column_name_second}'])
        
        # Отбираем финальные программы с соответствием
        result_found = result_df[result_df['similarity'] != 0.0].reset_index(drop=True)
        programs_found = list(result_found[f'Программа {column_name_second}'])
        
        # Анализируем методы поиска
        tfidf_found = result_found[result_found['method'] == 'tfidf']
        fuzzy_found = result_found[result_found['method'] == 'fuzzy']
        vocabulary_found = result_found[result_found['method'].isna()] if not result_found.empty else pd.DataFrame()
        
        # ДОБАВЛЕННАЯ ПРОВЕРКА: все ли программы нашли соответствия
        all_programs_matched = len(programs_not_found) == 0
        
        if print_in_console:
            # Вывод информации о результатах сопоставления
            print(f"\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
            print(f"Всего программ для поиска: {len(self.small_list)}")
            print(f"Найдено соответствий: {len(programs_found)}")
            print(f"  - TF-IDF совпадений (similarity >= {min_similarity}): {len(tfidf_found)}")
            print(f"  - Fuzzy совпадений (threshold >= {fuzzy_threshold}%): {len(fuzzy_found)}")
            
            if use_vocabulary and vocabulary is not None and not vocabulary_found.empty:
                print(f"  - Совпадений из справочника: {len(vocabulary_found)}")
            
            print(f"Не найдено соответствий: {len(programs_not_found)}")
            
            if not use_vocabulary or vocabulary is None:
                print("Справочник: НЕ ИСПОЛЬЗУЕТСЯ")
            else:
                print("Справочник: используется")
            
            if len(fuzzy_found) > 0:
                print(f"\nНайденные через fuzzy matching:")
                for idx, row in fuzzy_found.iterrows():
                    print(f"  • '{row[f'Программа {column_name_second}']}' -> "
                        f"'{row[f'Программа {column_name_first}']}' "
                        f"({row['similarity']:.2%})")
            
            if programs_not_found:
                print(f"\nПрограммы без соответствий:")
                for program in programs_not_found:
                    print(f"  • {program}")
            
            if all_programs_matched:
                print(f"\n✓ УСПЕХ: Для всех программ найдены соответствия")
            else:
                print(f"\n⚠ ВНИМАНИЕ: Не для всех программ найдены соответствия")
        
        return result_df, programs_not_found
    


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
    






"""
text_prepr = TextPreprocessor()

# 1. Программы в Palomars
plmrs_prgms, data_plmrs = text_prepr.clean_text(palomars_init, 'Название программы')
# Сет из уникальных программ Паломарса
plmrs_uniq_progs = list(set(plmrs_prgms))

# 2. Программы в VIMB
vimb_prgms, vimb = text_prepr.clean_text(vimb_init, 'Название программы')
# Сет из уникальных программ ВИМБа
vimb_uniq_progs = list(set(vimb_prgms))


similar = CosineSimilarity(plmrs_uniq_progs, vimb_uniq_progs, data_plmrs, vimb)

# Читаем данные из справочника
vocabulary = pd.read_excel(f'{prefix}Справочники/Справочник.xlsx', sheet_name = 'СТС LOVE')


result, matched_programs = similar.comparison(vocabulary, min_similarity = 0.5)

result_df = result[result['similarity'] != 0.00000].reset_index(drop = True)
"""