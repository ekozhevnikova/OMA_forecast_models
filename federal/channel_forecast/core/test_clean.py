import pandas as pd
import numpy as np
import re

import warnings
warnings.filterwarnings('ignore')


# Общие константы для очистки
SPECIAL_CHARS_TO_REMOVE_GLOBAL = [
    '\\', '/', '|', ':', ';', '*', '?', '<', '>', '~', '`',
    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', 
    '=', '[', ']', '{', '}', '-', '№', '"', "'", '«', '»'
]


STOP_PATTERNS_GLOBAL = [
    # Скобочные формы
    r'\((повтор|премьера|посвящение|сери[яи]|част[ьи]|эпизод[ы]?)\)',
    
    # Слова без скобок
    r'\b(повтор|премьера|посвящение|сезон|сери[яи]|част[ьи]|эпизод[ы]?)\b',
    
    # Специальные паттерны
    r'№\s*\d+',
    r'\bгг\.\s*',
    r'\bматч за\b',
    r'\bg[- ]?drive\b',
    r'\bальфа[- ]?банк\b',
    r'\b\d{8,}\b' # последовательность из 8ми и более цифр
]


SPECIAL_NAMES = {
    'все на матч', 'мультфильм', 'все о главном', 'что за спорт', #'матч парад',
    'непридуманные истории', 'век нашего спорта', 'география спорта', 
    'что по спорту', 'лица страны', 'культовые', 'команда мечты', 'третий тайм'
}

SPECIAL_PATTERNS_FOR_SAVE = [
                r'\b\d+\s+лет\s+в\s+спорте\b', # n лет в спорте
                r'\b\d+\s+лет\s+ufc\b'         # лет ufc
]


class SportChannelParsing:
    def __init__(self, channel):
        self.channel = channel

    @staticmethod
    def final_cleaning(text):
        """
            Финальная очистка текста - удаляет паттерны только в начале или в конце строки
        """
        if not isinstance(text, str):
            return text
        
        # Сохраняем оригинал для отладки
        original = text
        
        # Удаляем упоминания сезонов/частей в конце
        text = re.sub(
            r'\.?\s*(?:\d+\s*(?:сез(?:он|я)?|часть|сери[яи]?)|(?:сез(?:он|я)?|часть|сери[яи]?)\s*\d+)\s*$', 
            '', 
            text, flags=re.IGNORECASE
        )
        
        # СПЕЦИАЛЬНЫЙ ПАТТЕРН: удаляем номера серий в начале (2 серия, 2 часть и т.д.)
        # Это то, что нужно для случая "2 . тяжеловес"
        text = re.sub(r'^\s*\d+\s*[.\s]*', '', text, flags=re.IGNORECASE)
        
        # Паттерны для удаления В НАЧАЛЕ строки
        start_patterns = [
            r'^\s*\d+[а-яё](?=\s|\.|,|$)',        # 6я, 2я в начале
            r'^\s*\d+\s+[а-яё]\b',                 # 6 я, 2 я в начале
            r'^\s*\d+[а-яё]\b',                    # 6я, 2я как слово в начале
            r'^\s*\d+\.\b',                        # 6. в начале
            r'^\s*\d+\s*\.\s*',                    # 6 . в начале
            r'^\s*\d+,\b',                         # 6, в начале
            r'^\s*\d+\s*,\s*',                     # 6 , в начале
            r'^\s*\d+\s*',                         # просто число в начале (добавлено)
        ]
        
        # Паттерны для удаления В КОНЦЕ строки
        end_patterns = [
            r'\d+[а-яё]\s*$',                      # 6я в конце
            r'\b\d+\s+[а-яё]\s*$',                 # 6 я в конце
            r'\b\d+[а-яё]\s*$',                    # 6я как слово в конце
            r'\b\d+\.\s*$',                        # 6. в конце
            r'\b\d+\s*\.\s*$',                     # 6 . в конце
            r',\s*\d+\s*$',                        # , 6 в конце
            r'\b\d+,\s*$',                         # 6, в конце
            r'\b\d+\s*,\s*$',                      # 6 , в конце
            r'\.\s*\d+\s*$',                       # . 6 в конце
            r'\s*\d+\s*$',                         # просто число в конце (добавлено)
        ]
        
        # Применяем паттерны для начала строки
        for pattern in start_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            text = re.sub(r'^\s+', '', text)  # удаляем пробелы в начале после удаления
        
        # Применяем паттерны для конца строки
        for pattern in end_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+$', '', text)  # удаляем пробелы в конце после удаления
        
        # Удаляем точки, которые могли остаться между словами (но не внутри слов)
        # Например, "2 . тяжеловес" -> "2 тяжеловес" (точка уже убрана выше)
        
        # Знаки пунктуации - удаляем только в начале и конце
        punctuaction_marks = [
            '.', ',', '\\', '/', '|', ':', ';', '*', '?', '<', '>', '~', '`',
            '!', '@', '#', '$', '%', '^', '&', '(', ')', '+', 
            '=', '[', ']', '{', '}', '-', '№', '"', "'", '«', '»'
        ]
        escaped_marks = ''.join(re.escape(mark) for mark in punctuaction_marks)
        punctuation_pattern = f'[{escaped_marks}]'
        
        # Удаляем знаки пунктуации в начале и конце
        text = re.sub(f'^{punctuation_pattern}+\\s*', '', text)
        text = re.sub(f'\\s*{punctuation_pattern}+$', '', text)
        
        # ДОПОЛНИТЕЛЬНО: удаляем одиночные точки с пробелами (как в "2 . тяжеловес")
        text = re.sub(r'\s+\.\s+', ' ', text)  # " . " -> " "
        text = re.sub(r'^\s*\.\s*', '', text)  # точка в начале
        text = re.sub(r'\s*\.\s*$', '', text)  # точка в конце
        
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Финальная проверка: если осталось что-то типа "2 тяжеловес", удаляем число в начале
        text = re.sub(r'^\d+\s+', '', text)
        
        return text


    @staticmethod
    def extract_before_brackets(text):
        """
            Извлекает часть строки до первых скобок
        """
        if not isinstance(text, str):
            return text
        
        match = re.match(r'^([^(]+)\s*\(', text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return text


    def clean_program_name(
        self,
        text: str,
        program_type: str = 'other',  # 'films', 'sport', 'not_sport', 'other'
        stop_words_specific: set = None,
        stop_patterns: list = None,
        special_names: set = SPECIAL_NAMES,
        special_chars_to_remove: list = SPECIAL_CHARS_TO_REMOVE_GLOBAL,
        stop_patterns_general: list = STOP_PATTERNS_GLOBAL,
        special_patterns: set = SPECIAL_PATTERNS_FOR_SAVE,
        debug: bool = False) -> str:
        """
        Универсальная функция для очистки названий программ
        
        Параметры:
        - text: исходное название программы
        - program_type: тип программы ('films', 'sport', 'not_sport', 'other')
        - stop_words_specific: специфичные стоп-слова для типа программы
        - stop_patterns: паттерны для удаления

        - special_names: специальные имена для сохранения
        - special_chars_to_remove: спецсимволы для удаления
        - stop_words_general: общие стоп-слова для удаления (глобальные)
        - special_patterns: специальные паттерны для сохранения
        - debug: печатать отладочную информацию
        """
        
        if debug:
            print(f"\n{'='*50}")
            print(f"Тип: {program_type}")
            print(f"Канал: {self.channel}")
            print(f"Исходный текст: '{text}'")
        
        # === ЭТАП 1: ПОДГОТОВКА ===
        text_lower = text.lower().strip().replace('ё', 'е')
        result = text_lower

        # Удаляем упоминания сезонов/частей в конце
        result = re.sub(
            r'\.?\s*(?:\d+[-]?[яи]?\s*(?:сез(?:он|я)?|часть|сери[яи]?)|(?:сез(?:он|я)?|часть|сери[яи]?)\s*\d+[-]?[яи]?)\s*', 
            ' ', 
            result, flags = re.IGNORECASE
        )

        result = re.sub(
            r'\b\d+\s*[-–—/\s]?\s*\d+\s+(?:финала?|полуфинала?|четвертьфинала?)\b',
            ' ', 
            result, flags = re.IGNORECASE
        )
        
        if debug:
            print(f"ЭТАП 1 - После подготовки: '{result}'")
        
        # === ЭТАП 2: ОЧИСТКА ОТ ГЛОБАЛЬНЫХ ПАТТЕРНОВ ===
        if stop_patterns_general:
            for pattern in stop_patterns_general:
                result = re.sub(r'\s*' + pattern + r'\s*', ' ', result, flags=re.IGNORECASE | re.UNICODE)
                result = result.strip()

                if debug:
                    print(f"  -> После удаления стоп-слов: '{result}'")


        
        # === ЭТАП 3: ПРОВЕРКА НА СПЕЦИАЛЬНЫЕ ИМЕНА ===
        if special_names:

            # Удаляем от специальных символов
            if special_chars_to_remove:
                for char in special_chars_to_remove:
                    result = result.replace(char, '')

            if debug:
                print(f"ЭТАП 3 - Проверка special_names: '{result.strip()}' содержит special_names?")
            
            # Проверяем, содержит ли строка любое специальное имя
            result_stripped = result.strip()
            for special_name in special_names:
                if special_name in result_stripped:
                    if debug:
                        print(f"  -> НАЙДЕНО специальное имя: '{special_name}' в строке")
                    return special_name.strip()  # Возвращаем только имя
        
        # === ЭТАП 4: ПРОВЕРКА НА СПЕЦИАЛЬНЫЕ ПАТТЕРНЫ ===
        if special_patterns:
            # Удаляем от специальных символов
            if special_chars_to_remove:
                for char in special_chars_to_remove:
                    result = result.replace(char, '')

                for pattern in special_patterns:
                    match = re.search(pattern, result, flags=re.IGNORECASE | re.UNICODE)
                    if match:
                        found_text = match.group(0)  # получаем всю найденную подстроку
                        if debug:
                            print(f"ЭТАП 4 - Специальный паттерн '{pattern}' найден")
                            print(f"  -> Найденная подстрока: '{found_text}'")
                            print(f"  -> Возвращаем: '{found_text.strip()}'")
                        return found_text.strip()
        
        # === ЭТАП 5: УДАЛЕНИЕ ВОЗРАСТНЫХ РЕЙТИНГОВ ===
        old_result = result
        result = re.sub(r'\s*\d+\+', ' ', result)
        if debug and old_result != result:
            print(f"ЭТАП 5 - После удаления возрастных рейтингов: '{result}'")
        
        # === ЭТАП 5: УДАЛЕНИЕ ОБЩИХ СТОП-СЛОВ ===
        #if stop_words_general:
        #    if debug:
        #        print(f"ЭТАП 5 - Применяем stop_words_general ({len(stop_words_general)} паттернов)")
        #    
        #    for i, pattern in enumerate(stop_words_general):
        #        old = result
        #        result = re.sub(r'\s*' + pattern + r'\s*', ' ', result, flags=re.IGNORECASE | re.UNICODE)
        #        result = result.strip()
        #        if debug and old != result:
        #            print(f"  Паттерн {i}: '{pattern}' -> '{result}'")
        
        # === ЭТАП 6: ОБРАБОТКА В ЗАВИСИМОСТИ ОТ ТИПА ===
        words_to_remove = stop_words_specific if stop_words_specific else set()
        
        if debug:
            print(f"ЭТАП 6 - Тип программы: {program_type}")
            print(f"  Специфичные стоп-слова: {words_to_remove}")
            print(f"  Текущий результат: '{result}'")
        
        if program_type == 'films':

            # Сначала удаляем все стоп-слова вместе с прилегающими символами
            for stop_word in words_to_remove:
                # Удаляем слово, даже если оно слиплось с другими буквами
                pattern = rf'{re.escape(stop_word)}'
                result = re.sub(pattern, ' ', result, flags=re.IGNORECASE | re.UNICODE)
                result = re.sub(r'\s+', ' ', result).strip()
                
                if debug:
                    print(f"    Удалили '{stop_word}' (игнорируя границы слов): '{result}'")
            
            # Для фильмов - удаляем паттерны и стоп-слова
            if stop_patterns:
                if debug:
                    print(f"  Применяем stop_patterns для фильмов")
                for pattern in stop_patterns:
                    old = result
                    result = re.sub(r'\s*' + pattern + r'\s*', ' ', result, flags=re.IGNORECASE | re.UNICODE)
                    result = re.sub(pattern, ' ', result, flags=re.IGNORECASE | re.UNICODE)
                    result = re.sub(r'\s+', ' ', result).strip()
                    if debug and old != result:
                        print(f"    Паттерн '{pattern}' -> '{result}'")
            
            for stop_word in words_to_remove:
                pattern = r'\b' + re.escape(stop_word) + r'\b'
                old = result
                result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.UNICODE)
                result = re.sub(r'\s+', ' ', result).strip()
                if debug and old != result:
                    print(f"    Стоп-слово '{stop_word}' -> '{result}'")
                
        elif program_type == 'sport':
            # Для спортивных программ - только стоп-слова
            for stop_word in words_to_remove:
                pattern = r'\b' + re.escape(stop_word) + r'\b'
                old = result
                result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.UNICODE)
                result = re.sub(r'\s+', ' ', result).strip()
                if debug and old != result:
                    print(f"    Стоп-слово '{stop_word}' -> '{result}'")
                
        elif program_type == 'not_sport':
            # Для не-спортивных программ - специальная логика со скобками
            before_removal = result
            
            if debug:
                print(f"  not_sport: результат до удаления специфичных слов: '{before_removal}'")
            
            if words_to_remove:
                for stop_word in words_to_remove:
                    pattern = r'\b' + re.escape(stop_word) + r'\b'
                    old = result
                    result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.UNICODE)
                    result = re.sub(r'\s+', ' ', result).strip()
                    if debug and old != result:
                        print(f"    Удалили '{stop_word}': '{old}' -> '{result}'")
            
            # Проверяем, нужно ли извлечь из скобок
            if debug:
                print(f"  Проверка на извлечение из скобок:")
                print(f"    result пустой? {not result.strip()}")
                print(f"    len(result) < 3? {len(result.strip()) < 3}")
                print(f"    '(' in text_lower? {'(' in text_lower}")
            
            if (not result.strip() or len(result.strip()) < 3) and '(' in text_lower:
                if debug:
                    print(f"  -> Условие выполнено, ищем скобки в '{text_lower}'")
                
                match = re.search(r'\(([^)]+)\)', text_lower)
                if match:
                    result = match.group(1).strip()
                    if debug:
                        print(f"  -> Извлечено из скобок: '{result}'")
            
            # Проверяем специальные имена после извлечения
            if special_names and result.strip() in special_names:
                if debug:
                    print(f"  -> После извлечения получено специальное имя: '{result}'")
                return result.strip()
        
        # === ЭТАП 7-10: ОСТАЛЬНАЯ ОЧИСТКА ===
        if debug:
            print(f"ЭТАП 7 - Удаление скобок и кавычек")
        result = result.replace('"', '').replace("'", '')
        result = result.replace('«', '').replace('»', '')
        result = result.replace('(', '').replace(')', '')
        result = result.replace('[', '').replace(']', '')
        result = result.replace('{', '').replace('}', '')
        
        if debug:
            print(f"  После ЭТАПА 7: '{result}'")
            print(f"ЭТАП 8 - Базовая очистка")
        
        result = re.sub(r'\s+', ' ', result).strip()
        
        if debug:
            print(f"  После ЭТАПА 8: '{result}'")
            print(f"ЭТАП 9 - Удаление спецсимволов")
        
        if special_chars_to_remove:
            for char in special_chars_to_remove:
                result = result.replace(char, '')
        
        result = re.sub(r'^\s*\.\s*', '', result)  # точка в начале
        result = re.sub(r'\s*\.\s*$', '', result)  # точка в конце
        result = re.sub(r'^\s*\,\s*', '', result)  # точка в начале
        result = re.sub(r'\s*\,\s*$', '', result)  # точка в конце
        result = re.sub(r'\s*\,\s*$', '', result)  # точка в конце

        if debug:
            print(f"  После ЭТАПА 9: '{result}'")
            print(f"ЭТАП 10 - Финальная очистка")
        
        result = SportChannelParsing.final_cleaning(result)
        
        if debug:
            print(f"  После ЭТАПА 10: '{result}'")
        
        # Финальная проверка
        if not result or len(result) < 3:
            if debug:
                print(f"Результат пустой или короткий, проверяем special_names")
            if special_names and text_lower.strip() in special_names:
                if debug:
                    print(f"  -> Возвращаем специальное имя: '{text_lower.strip()}'")
                return text_lower.strip()
        
        if debug:
            print(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: '{result}'")
            print('='*50)
        
        result = re.sub(r'\.', '', result)  # точка в любом месте
        result = re.sub(r'\,', '', result)  # запятая в любом месте
        result = re.sub(r'\b\d{4,}\b', '', result) #удаление последовательности из 8ми и более цифр
        result = re.sub(r'\s+', ' ', result).strip() # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
        return result