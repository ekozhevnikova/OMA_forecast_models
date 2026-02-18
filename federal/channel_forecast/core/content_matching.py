import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Set, List, Dict, Optional, Tuple, Any
from OMA_tools.io_data.colors import *
from fuzzywuzzy import fuzz, process

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ГЛОБАЛЬНЫЕ КОНСТАНТЫ С ГРУППИРОВКОЙ КАНАЛОВ
# ============================================================================
SPECIAL_CHARS_TO_REMOVE_GLOBAL = [
            '\\', '/', '|', ':', ';', '*', '?', '<', '>', '~', '`',
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', 
            '=', '[', ']', '{', '}', '-', '№', '"', "'", '«', '»'
        ]


class GeneralTextCleaner:
    """
        Класс для очистки названий телепередач от служебных пометок,
        стоп-слов и лишних символов в зависимости от канала.

        !!! ВАЖНО !!!
        Данный класс предназначен для работы с каналами:
        - ТНТ4
        - 2Х2
        - СТСЛав
        - Солнце
        - Карусель
        - Суббота
        - Че
        - МузТВ
        - Мир
        - Спас
        - Ю
        - ТВЦ
        - Звезда
    """
    

    def __init__(self, channel):
        """
            Инициализация класса и сборка финальных словарей
            Args:
                channel: название канала
        """
        self.channel = channel


        # ============================================================================
        # ГРУППЫ КАНАЛОВ ПО ТИПАМ
        # ============================================================================

        # Группа 1: Каналы с документальными фильмами/сериалами
        self.DOC_CHANNELS = {'СПАС', 'ЗВЕЗДА', 'ТВЦ'}

        # Группа 2: Детские/анимационные каналы
        self.KIDS_CHANNELS = {'СОЛНЦЕ', 'КАРУСЕЛЬ', 'СУББОТА', 'СТСЛав', '2X2'}

        # Группа 3: Развлекательные каналы
        self.ENTERTAINMENT_CHANNELS = {'ТНТ4', 'ЧЕ', 'МИР', 'МузТВ', 'Ю'}

        # ============================================================================
        # БАЗОВЫЕ НАБОРЫ ДЛЯ КАЖДОЙ ГРУППЫ
        # ============================================================================

        # Базовые специальные названия
        self.BASE_SPECIAL_NAMES = {
            'мультфильм', 'сериал', 'художественный фильм', 'документальный фильм',
            'мультсериал', 'худ. фильм', 'фильм, фильм, фильм'
        }

        # Базовые стоп-слова
        self.BASE_STOP_WORDS = {
            'сериал', 'серия', 'часть', 'сезон', 'фильм', 'художественный'
        }

        # Базовые паттерны
        self.BASE_PATTERNS = [
            r'х\W*ф', r'№\s*\d+', r'\bсериал\b', r'\bсерия\b', r'\bсезон\b', 
            r'\bчасть\b', r'\d+\s*серия\b', r'\d+\s*сезон\b', r'\d+\s*часть\b'
        ]

        # ============================================================================
        # СПЕЦИАЛИЗИРОВАННЫЕ НАБОРЫ ДЛЯ ГРУПП
        # ============================================================================

        # Для документальных каналов
        self.DOC_SPECIFIC = {
            'special_names': {
                'док. фильм/сериал', 'сериал/х.ф. (п)', 'сериал/фильм (п)', 'худ.фильм/сериал',
                'сериал/фильм', 'док. сериал/фильм (п)', 'док. сериал/фильм', 'док.фильм/сериал',
                'сериал/х.ф.'
            },
            'stop_words': {'документальный', 'док.'},
            'patterns': [
                r'\bм\W*ф\b', r'\bm\W*ф\b', r'\bа\W*ф\b', r'\bд\W*ф\b', r'\bх\W*ф\b',  r'\bв\W*[сc]\b',
                r'\(\s*[а-яё]+\s*\)', r'\bдок\.\s*', r'\bхуд\.\s*']
        }

        # Для детских/анимационных каналов
        self.KIDS_SPECIFIC = {
            'special_names': {'мультфильм', 'мультсериал'},
            'stop_words': {
                'большая анимация', 'анимационный', 'мультфильм', 'мультсериал',
                'огр', 'муз', 'р/б', 'т/с', 'а/с'
            },
            'patterns': [
                # Удаление сочетаний 'мф', 'mф', 'аф', 'мс', 'тс', 'ас', если это они отдельно стоящие 
                r'\bм\W*ф\b', r'\bm\W*ф\b', r'\bа\W*ф\b', r'\bд\W*ф\b', 
                r'\bм\W*[сc]\b', r'\bт\W*[сc]\b', r'\bа\W*[сc]\b',  # [сc] - русская или латинская с
                r'\d{4}(?:\s*г(?:од)?\.?)?', r'\bмультфильм\b', r'\bсоюзмультфильм\b',
                r'\(\s*(?:сериал|серия|сезон|часть)\s*\d*\s*\)',
                r'\d+\s*сез\b', r'\bсез\s*\d+\b'
            ]
        }

        # Для развлекательных каналов
        self.ENTERTAINMENT_SPECIFIC = {
            'special_names': {},
            'stop_words': {'комедийный', 'концерт', 'спец', 'реалити'},
            'patterns': [
                r'\bкомедийный\b', r'\bконцерт\b', r'\bреалити[- ]?шоу\b', r'\bдок\.\s*',
                r'\bреалити\b', r'\bспец\b', r'\bм\W*ф\b', r'\bm\W*ф\b', r'\bд\W*ф\b', 
                r'\bм\W*[сc]\b', r'\bт\W*[сc]\b', r'\bа\W*[сc]\b'
            ]
        }

        # ============================================================================
        # ИНДИВИДУАЛЬНЫЕ НАСТРОЙКИ ДЛЯ КАНАЛОВ (ПЕРЕОПРЕДЕЛЕНИЯ)
        # ============================================================================

        self.CHANNEL_SPECIFIC = {
            'ТНТ4': {
                'special_names': {'комедийный сериал'},
                'patterns': [r'\bкомедийный\b']
            },
            'ЧЕ': {
                'patterns': [r'\bмультфильм\b']
            },
            'МИР': {
                'patterns': [r'\bдокументальный\b', r'\bхуд\.\s*', r'\bмультфильм\b']
            },
            'СПАС': {
                'patterns': [r'\bдокументальный\b', r'\bхуд\.\s*', r'\bмультфильм\b']
            },
            'МузТВ': {
                'stop_words': {'спец', 'концерт'},
                'patterns': [r'\bконцерт\b', r'\bспец\b', r'\b[тt]\W*9\b']
            },
            'Ю': {
                'stop_words': {'реалити'},
                'patterns': [r'\bреалити[- ]?шоу\b', r'\bреалити\b', r'\bмультфильм\b', r'\bмультсериал\b']
            },
            'СОЛНЦЕ': {
                'patterns': [
                    r'\bх\W*ф\b',  # дополнительный паттерн для Солнца
                ]
            }
        }


        self.special_names_global = self._build_special_names()
        self.stop_words_global = self._build_stop_words()
        self.stop_patterns_global = self._build_stop_patterns()


    def _build_special_names(self) -> Dict[str, Set[str]]:
        """
            Собирает SPECIAL_NAMES_GLOBAL из групп и индивидуальных настроек
        """
        special_names = {}
        
        for channel in self.DOC_CHANNELS:
            special_names[channel] = self.BASE_SPECIAL_NAMES | self.DOC_SPECIFIC['special_names']
            if channel in self.CHANNEL_SPECIFIC and 'special_names' in self.CHANNEL_SPECIFIC[channel]:
                special_names[channel] |= self.CHANNEL_SPECIFIC[channel]['special_names']
        
        for channel in self.KIDS_CHANNELS:
            special_names[channel] = self.BASE_SPECIAL_NAMES | self.KIDS_SPECIFIC['special_names']
            if channel in self.CHANNEL_SPECIFIC and 'special_names' in self.CHANNEL_SPECIFIC[channel]:
                special_names[channel] |= self.CHANNEL_SPECIFIC[channel]['special_names']
        
        for channel in self.ENTERTAINMENT_CHANNELS:
            special_names[channel] = self.BASE_SPECIAL_NAMES.copy()
            if channel in self.CHANNEL_SPECIFIC and 'special_names' in self.CHANNEL_SPECIFIC[channel]:
                special_names[channel] |= self.CHANNEL_SPECIFIC[channel]['special_names']
        
        return special_names
    

    def _build_stop_words(self) -> Dict[str, Set[str]]:
        """
            Собирает STOP_WORDS_GLOBAL из групп и индивидуальных настроек
        """
        stop_words = {}
        
        for channel in self.DOC_CHANNELS:
            stop_words[channel] = self.BASE_STOP_WORDS | self.DOC_SPECIFIC['stop_words']
            if channel in self.CHANNEL_SPECIFIC and 'stop_words' in self.CHANNEL_SPECIFIC[channel]:
                stop_words[channel] |= self.CHANNEL_SPECIFIC[channel]['stop_words']
        
        for channel in self.KIDS_CHANNELS:
            stop_words[channel] = self.BASE_STOP_WORDS | self.KIDS_SPECIFIC['stop_words']
            if channel in self.CHANNEL_SPECIFIC and 'stop_words' in self.CHANNEL_SPECIFIC[channel]:
                stop_words[channel] |= self.CHANNEL_SPECIFIC[channel]['stop_words']
        
        for channel in self.ENTERTAINMENT_CHANNELS:
            stop_words[channel] = self.BASE_STOP_WORDS | self.ENTERTAINMENT_SPECIFIC['stop_words']
            if channel in self.CHANNEL_SPECIFIC and 'stop_words' in self.CHANNEL_SPECIFIC[channel]:
                stop_words[channel] |= self.CHANNEL_SPECIFIC[channel]['stop_words']
        
        return stop_words
    

    def _build_stop_patterns(self) -> Dict[str, List[str]]:
        """
            Собирает STOP_PATTERNS_GLOBAL из групп и индивидуальных настроек
        """
        stop_patterns = {}
        
        for channel in self.DOC_CHANNELS:
            stop_patterns[channel] = self.BASE_PATTERNS + self.DOC_SPECIFIC['patterns']
            if channel in self.CHANNEL_SPECIFIC and 'patterns' in self.CHANNEL_SPECIFIC[channel]:
                stop_patterns[channel].extend(self.CHANNEL_SPECIFIC[channel]['patterns'])
        
        for channel in self.KIDS_CHANNELS:
            stop_patterns[channel] = self.BASE_PATTERNS + self.KIDS_SPECIFIC['patterns']
            if channel in self.CHANNEL_SPECIFIC and 'patterns' in self.CHANNEL_SPECIFIC[channel]:
                stop_patterns[channel].extend(self.CHANNEL_SPECIFIC[channel]['patterns'])
        
        for channel in self.ENTERTAINMENT_CHANNELS:
            stop_patterns[channel] = self.BASE_PATTERNS + self.ENTERTAINMENT_SPECIFIC['patterns']
            if channel in self.CHANNEL_SPECIFIC and 'patterns' in self.CHANNEL_SPECIFIC[channel]:
                stop_patterns[channel].extend(self.CHANNEL_SPECIFIC[channel]['patterns'])
        
        return stop_patterns
    

    def _get_default_stop_words(self) -> Set[str]:
        result = self.stop_words_global.get(self.channel, set())
        # ДОБАВЬТЕ ДЕБАГ:
        #print(f"DEBUG _get_default_stop_words: channel={self.channel}, result={result}")
        return result
    
    def _get_default_stop_patterns(self) -> List[str]:
        """Возвращает паттерны для текущего канала"""
        result = self.stop_patterns_global.get(self.channel, [])
        #print(f"DEBUG _get_default_stop_patterns: channel={self.channel}, найдено {len(result)} паттернов")
        return result
    
    def _get_default_special_names(self) -> Set[str]:
        """Возвращает специальные имена для текущего канала"""
        result = self.special_names_global.get(self.channel, set())
        #print(f"DEBUG _get_default_special_names: channel={self.channel}, result={result}")
        return result
    

    def _replace_whole_word(self, text: str, old_word: str, new_word: str) -> str:
        """
            Заменяет целые слова в тексте
        """
        pattern = r'\b' + re.escape(old_word) + r'\b'
        return re.sub(pattern, new_word, text)
    

    def clean_title(self, title: str) -> str:
        """
            Очищает название от служебных пометок
        """
        # Удаляем внутренние пометки
        title = re.sub(r'\s*\([^)]*\)\s*', ' ', title)
        
        # Удаляем паттерны типа т/с, а/с, м/с, a/ф
        title = re.sub(r'\s*[а-яa-z]+/[а-яa-z]+\s*', ' ', title, flags = re.IGNORECASE)
        
        # Удаляем упоминания сезонов и серий
        title = re.sub(r'\s*\d+\s*сезон\s*', ' ', title)
        title = re.sub(r'\.?\s*сезон\s*\d+\s*$', '', title, flags = re.IGNORECASE)
        title = re.sub(r'\s*\d+\s*сезон\s*$', '', title, flags = re.IGNORECASE)
        
        title = re.sub(r'\.?\s*серия\s*\d+\s*$', '', title, flags = re.IGNORECASE)
        title = re.sub(r'\s*\d+\s*серия\s*$', '', title, flags = re.IGNORECASE)
        
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Убираем точку в конце
        if title.endswith('.'):
            title = title[:-1].strip()
        
        return title


    def _handle_special_cases(self, text: str, text_lower: str) -> tuple[bool, str]:
        """
            Обрабатывает специальные случаи для конкретных каналов
        """
        
        # Специальная обработка для ЗВЕЗДА
        if self.channel == 'ЗВЕЗДА':
            # Только специфичные для ЗВЕЗДА случаи, которые НЕ могут быть обработаны общим алгоритмом
            if re.search(r'док\.?\s*сериал/?фильм\s*\(\s*п\s*\)', text):
                return True, 'документальный сериал'
            
            # А всё остальное (например, извлечение названия) возвращаем False,
            # чтобы основной алгоритм обработал
            return False, text
        
        # Специальная обработка для 2X2
        if self.channel == '2X2' and 'фильм, фильм, фильм' in text:
            match = re.search(r'"([^"]+)"', text)
            if match:
                return True, match.group(1).strip()

            return False, text
        
        # Специальная обработка для ТНТ4
        if self.channel == 'ТНТ4' and 'комеди' in text:
            if 'комеди' in text and 'комедийный' not in text:
                return True, self._replace_whole_word(text, 'комеди', 'камеди')
            
            return False, text
            
        # Специальная обработка для СОЛНЦЕ
        if self.channel == 'СОЛНЦЕ':
            cleaned = text
            
            # Удаляем м/с, мс, м/с, мс в разных вариациях
            # Паттерны: м/с, мс, м/с, мс (с точкой или без, с пробелами или без)
            ms_patterns = [
                r'\bм/?с\b',           # м/с, мс
                r'\bм\.?\s*с\.?\b',    # м.с, м с, м. с.
                r'\(\s*м/?с\s*\)',      # (м/с), (мс)
                r'\s+м/?с\s+',          # м/с с пробелами
            ]
            
            for pattern in ms_patterns:
                cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
            
            # Удаляем (с) в любом виде: (с), (С), (c) латиницей
            if re.search(r'\(\s*[сcСC]\s*\)', cleaned):
                cleaned = re.sub(r'\s*\(\s*[сcСC]\s*\)\s*', ' ', cleaned)
            
            # Схлопываем пробелы после удалений
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Проверяем, не осталось ли других служебных пометок
            if 'киносолнце' in cleaned.lower() or 'кино солнце' in cleaned.lower():
                return True, 'киносолнце'
            
            # Извлекаем из кавычек если есть
            match = re.search(r'"([^"]+)"', cleaned)
            if match:
                return True, self.clean_title(match.group(1).strip())
            
            # Если были какие-то удаления, возвращаем очищенный текст
            if cleaned != text:
                return True, self.clean_title(cleaned)
        
        # Проверка на сборники мультфильмов
        if ('m/ф' in text or 'м/ф' in text_lower) and text_lower.count('м/ф') + text_lower.count('m/ф') >= 2:
            return True, 'мультфильм'
        
        return False, text


    def final_clean(self, text: str):
        """
            Метод для финальной очистки текста от всех ненужных символов.
        """
        result = text

        # Финальное схлопывание пробелов        
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\.', '', result)  # точка в любом месте
        result = re.sub(r'\,', '', result)  # запятая в любом месте
        result = re.sub(r'\b\d{8,}\b', '', result) #удаление последовательности из 8ми и более цифр
        result = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', result) # удаление всех символов, кроме букв и цифр
        result = re.sub(r'\s+', ' ', result).strip() # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
        return result


    def clean(
            self,
            text: str, 
            stop_words: Optional[Set[str]] = None, 
            stop_patterns: Optional[List[str]] = None,
            special_names: Optional[Set[str]] = None,
            special_chars: list = SPECIAL_CHARS_TO_REMOVE_GLOBAL,
            debug: bool = False
        ) -> str | None:
        """
        Args:
            text: исходный текст
            channel: название канала
            stop_words: стоп-слова для канала
            stop_patterns: паттерны для канала
            special_names: специальные имена
            special_chars: спецсимволы для удаления
            debug: если True, выводит отладочную информацию
        """
        
        # Сохраняем исходный текст для дебага
        original_text = text
        
        # ===== ШАГ 1: ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ =====
        if debug:
            print(f"\n{'='*60}")
            print(f"ДЕБАГГЕР: remove_stop_words для канала '{self.channel}'")
            print(f"{'='*60}")
            print(f"Исходный текст: '{original_text}'")
            

        ##############################################################################
        # Если параметры не переданы, берем настройки для канала
        if special_names is None:
            special_names = self._get_default_special_names()
            if debug:
                print(f"special_names загружены: {special_names}")
        
        if stop_words is None:
            stop_words = self._get_default_stop_words()
            if debug:
                print(f"stop_words загружены: {stop_words}")  # ЭТО НЕ ВИДНО ВО ВТОРОМ СЛУЧАЕ!
        
        if stop_patterns is None:
            stop_patterns = self._get_default_stop_patterns()
            if debug:
                print(f"stop_patterns загружены: {len(stop_patterns)} паттернов")
                for i, p in enumerate(stop_patterns[:3]):
                    print(f"  паттерн {i+1}: {p}")
                if len(stop_patterns) > 3:
                    print(f"  ... и еще {len(stop_patterns)-3}")
        
        if special_chars is None:
            special_chars = SPECIAL_CHARS_TO_REMOVE_GLOBAL
        
        # ДОБАВИТЬ ПРОВЕРКУ!
        if debug:
            print(f"ИТОГОВЫЕ НАСТРОЙКИ:")
            print(f"  special_names: {special_names}")
            print(f"  stop_words: {stop_words}")
            print(f"  stop_patterns: {len(stop_patterns)} паттернов")
        ##############################################################################
        

        # ===== ШАГ 2: ПОДГОТОВКА ТЕКСТА =====
        text_lower = text.lower().strip().replace('ё', 'е')
        result = text_lower

        # Удаляем возрастные рейтинги (12+, 16+ и т.д.)
        result = re.sub(r'\s*(0|[68]|1[0268]|2[04]|[3-9][02468])\+', ' ', result, flags=re.IGNORECASE)

        old_result = result
        result = re.sub(r'\(\s*[а-яa-z]\s*\)', ' ', result, flags=re.IGNORECASE)
        if debug and old_result != result:
            print(f"Удалены одиночные буквы в скобках: '{result}'")

        # Удаляем упоминания серий с номерами
        old_result = result
        result = re.sub(r'\s*\d+(?:\s*,\s*\d+\s*)*\s*сери[яи]?\s*', ' ', result, flags=re.IGNORECASE)
        if debug and old_result != result:
            print(f"Удалены номера серий: '{result}'")
        
        # Удаляем номера с символом № или n
        old_result = result
        result = re.sub(r'\s*[#№]\s*\d+\s*', ' ', result, flags=re.IGNORECASE)
        if debug and old_result != result:
            print(f"Удалены номера с №/n: '{result}'")
        
        if debug:
            print(f"\n--- ШАГ 2: Подготовка текста ---")
            print(f"После lower() и strip(): '{result}'")

        # ===== ШАГ 3: ПРОВЕРКА СПЕЦИАЛЬНЫХ СЛУЧАЕВ =====
        if debug:
            print(f"\n--- ШАГ 3: Проверка специальных случаев ---")
        # Удаление кавычек
        result = result.replace('"', '').replace("'", '').replace('«', '').replace('»', '')

        handled, special_result = self._handle_special_cases(result, text_lower)
        if handled:
            if debug:
                print(f"Сработал специальный случай!")
                print(f"Результат: '{special_result}'")
                print(f"{'='*60}\n")

            # Удаляем специальные символы из списка
            for char in special_chars:
                special_result = special_result.replace(char, ' ')

            special_result = self.final_clean(special_result)
            return special_result
        
        elif debug:
            print(f"Специальных случаев не найдено")

        # ===== ШАГ 4: ПРОВЕРКА НА СПЕЦИАЛЬНЫЕ ИМЕНА =====
        if debug:
            print(f"\n--- ШАГ 4: Проверка на специальные имена ---")
            print(f"Текст: '{result.strip()}'")
            print(f"Ищем в special_names: {result.strip() in special_names}")
        
        if result.strip() in special_names:
            if debug:
                print(f"Найдено специальное имя!")
                print(f"Результат: '{result.strip()}'")
                print(f"{'='*60}\n")

            result = self.final_clean(result)
            return result
        elif debug:
            print(f"Не является специальным именем")

        # ===== ШАГ 5: ИЗВЛЕЧЕНИЕ ИЗ КАВЫЧЕК =====
        if debug:
            print(f"\n--- ШАГ 6: Извлечение из кавычек ---")
        
        result = result.replace('"', '').replace("'", '').replace('«', '') \
                .replace('»', '').replace('(', '').replace(')', '')

        if debug:
            print(f"После: '{result}'")


        # ===== ШАГ 6: СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ КАНАЛА МУЗТВ =====
        if self.channel == 'МузТВ':
            if debug:
                print(f"\n--- ШАГ 7: Спецобработка для МузТВ ---")
            
            MUZTV_REPLACEMENTS = {
                r'(хит сториз(?:\s*спец)?)\s*\([^()]*(?:\([^()]*\)[^()]*)*\)': r'\1',
                r'(10 самых)\s*\([^()]*(?:\([^()]*\)[^()]*)*\)': r'\1',
                r'(натальная\s*карта)\s*\([^)]*\)': r'\1',
                r'вк(?:\s+спец)?\s*\(\s*контакты[^)]*\)': 'вк контакты',
                r'вк(?:\s+спец)?\s*\(\s*натальная\s*карта[^)]*\)': 'вк натальная карта',
                r'вк(?:\s+спец)?\s*\(\s*громкий\s*вопрос[^)]*\)': 'вк громкий вопрос',
                r'вк\s*\(\s*неигры[^)]*\)': 'вк неигры',
                r'вк\s*\(\s*[^)]+\)': 'вк',
                r'(московский выпускной)\s*\([^()]*(?:\([^()]*\)[^()]*)*\)': r'\1',
            }

            for i, (pattern, replacement) in enumerate(MUZTV_REPLACEMENTS.items(), 1):
                old_result = result
                result = re.sub(pattern, replacement, result, flags = re.IGNORECASE)
                if debug and old_result != result:
                    print(f"  Применен паттерн {i}: '{pattern[:30]}...' -> '{replacement}'")
                    print(f"  Результат: '{result}'")

        # ===== ШАГ 8: УДАЛЕНИЕ СТОП-ПАТТЕРНОВ =====
        if debug:
            print(f"\n--- ШАГ 8: Удаление стоп-паттернов ---")
        
        sorted_patterns = sorted(stop_patterns, key = len, reverse = True)
        
        for i, pattern in enumerate(sorted_patterns, 1):
            old_result = result
            result = re.sub(r'\s*' + pattern + r'\s*', ' ', result, flags = re.IGNORECASE)
            if debug and old_result != result:
                print(f"  Паттерн {i}: '{pattern[:30]}...' -> '{result}'")

        # ===== ШАГ 9: УДАЛЕНИЕ КАВЫЧЕК И СКОБОК =====
        if debug:
            print(f"\n--- ШАГ 9: Удаление кавычек и скобок ---")
            print(f"До: '{result}'")
        
        result = result.replace('"', '').replace("'", '').replace('«', '') \
                    .replace('»', '').replace('(', '').replace(')', '')
        
        if debug:
            print(f"После: '{result}'")

        # ===== ШАГ 10: УДАЛЕНИЕ СТОП-СЛОВ =====
        if debug:
            print(f"\n--- ШАГ 10: Удаление стоп-слов ---")
        
        for stop_word in stop_words:
            pattern = r'\b' + re.escape(stop_word) + r'\b'
            old_result = result
            result = re.sub(pattern, '', result, flags = re.IGNORECASE)
            if debug and old_result != result:
                print(f"  Удалено стоп-слово '{stop_word}': '{result}'")

        # ===== ШАГ 11: ФИНАЛЬНАЯ ОЧИСТКА =====
        if debug:
            print(f"\n--- ШАГ 11: Финальная очистка ---")
            print(f"До: '{result}'")
        
        # Убираем множественные пробелы
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Удаляем специальные символы из списка
        for char in special_chars:
            old_result = result
            result = result.replace(char, ' ')
            if debug and old_result != result and char.strip():
                print(f"  Удален символ '{char}': '{result}'")
        
        # Удаляем упоминания сезонов и серий в конце строки
        old_result = result
        result = re.sub(r'\.?\s*(?:\d+\s*(?:сез(?:он|я)?|часть)|(?:сез(?:он|я)?|часть)\s*\d+)\s*$', 
                        '', result, flags=re.IGNORECASE)
        if debug and old_result != result:
            print(f"  Удалены сезоны/серии в конце: '{result}'")
        
        # Удаляем точку в конце
        old_result = result
        result = re.sub(r'\.$', '', result).strip()
        if debug and old_result != result:
            print(f"  Удалена точка в конце: '{result}'")
        
        result = self.final_clean(result)
        
        if debug:
            print(f"После финальной очистки: '{result}'")
            print(f"\n{'='*60}")
            print(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: '{result}'")
            print(f"{'='*60}\n")

        if result == '':
            print(Color.BOLD + Color.RED + f'‼️ Для {self.channel} строка {text} оказалась пустой. Проверьте обработку текста!' + Color.END)

        return result
    





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