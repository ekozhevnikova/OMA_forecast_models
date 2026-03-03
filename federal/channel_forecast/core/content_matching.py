import numpy as np
import pandas as pd
import geonamescache
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
        - Мир
        - Ю
    """
    

    def __init__(self, channel, df):
        """
            Инициализация класса и сборка финальных словарей
            Args:
                channel: название канала
                df: датафрейм, в котором будем добавлять дополнительный столбец 'program_name' с очищенным названием программы
        """
        self.channel = channel
        self.df = df


        # ============================================================================
        # ГРУППЫ КАНАЛОВ ПО ТИПАМ
        # ============================================================================

        # Группа 1: Каналы с документальными фильмами/сериалами
        #self.DOC_CHANNELS = {'СПАС', 'ЗВЕЗДА', 'ТВЦ'}

        # Группа 2: Детские/анимационные каналы
        self.KIDS_CHANNELS = {'СОЛНЦЕ', 'КАРУСЕЛЬ', 'СУББОТА', 'СТСЛав', '2X2'}

        # Группа 3: Развлекательные каналы
        self.ENTERTAINMENT_CHANNELS = {'ТНТ4', 'ЧЕ', 'МИР', 'Ю'}

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
            r'\sх\W*ф\s', r'№\s*\d+', r'\bсериал\b', r'\bсерия\b', r'\bсезон\b', 
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
            'stop_words': {'документальный', 'док.', 'сериал', 'мультфильм'},
            'patterns': [
                r'\bм\W*ф\b', r'\bm\W*ф\b', r'\bа\W*ф\b', r'\bд\W*ф\b', r'(?<!\w)х\W*ф(?!\w)',  r'\bв\W*[сc]\b',
                r'\(\s*[а-яё]+\s*\)', r'\bдок\.\s*', r'\bхуд\.\s*', r'\bмультфильм\b',
                r'\bдокументальный\b', r'\bфильм\b', r'\(\s*(?:сериал|серия|сезон|часть)\s*\d*\s*\)',
                r'\d+\s*сез\b', r'\bсез\s*\d+\b'
                ]
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
                r'\d{4}(?:\s*г(?:од)?\.?)?', r'\bмультфильм\b', r'\bсоюзмультфильм\b', r'\bповтор\b',
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
            '2X2': {
                'patterns': [r'\bхуд\.\s*', r'\bдокументальный\b']
            },
            'МИР': {
                'patterns': [r'\bдокументальный\b', r'\bхуд\.\s*', r'\bмультфильм\b']
            },
            'ЗВЕЗДА': {
                'special_names': {'1812'} 
            },
            'ТВЦ': {
                'patterns': [r'\bтелесериал\b', r'\bсериал\b']
            },
            'СПАС': {
                'patterns': [r'\bдокументальный\b', r'\bхуд\.\s*', r'\bмультфильм\b'],
                'special_names': {'старцы'} 
            },
            'Ю': {
                'stop_words': {'реалити'},
                'patterns': [r'\bреалити[- ]?шоу\b', r'\bреалити\b', r'\bмультфильм\b', r'\bмультсериал\b']
            },
            'СОЛНЦЕ': {
                'patterns': [
                    r'\bх\W*ф\b',  # дополнительный паттерн для Солнца
                ],
                'special_names': {'фильм.фильм.фильм', 'фильм фильм фильм'} 
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
        
        # Слова для удаления с номерами
        words = ['сезон', 'серия', 'фильм']
        
        for word in words:
            # Варианты: "1 сезон", "сезон 1", "1-й сезон", "сезон 1-й"
            patterns = [
                rf'\s*\d+\s*{word}\s*',
                rf'\s*{word}\s*\d+\s*',
                rf'\s*\d+[-яй]?\s*{word}\s*',
                rf'\s*{word}\s*\d+[-яй]?\s*',
                rf'\.?\s*{word}\s*\d+\s*$',
                rf'\s*\d+\s*{word}\s*$',
            ]
            
            for pattern in patterns:
                title = re.sub(pattern, ' ', title, flags=re.IGNORECASE)
        
        # Схлопываем пробелы
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Убираем точку в конце
        if title.endswith('.'):
            title = title[:-1].strip()
        
        return title


    def _handle_special_cases(self, text: str, text_lower: str) -> tuple[bool, str]:
        """
            Обрабатывает специальные случаи для конкретных каналов
        """

        # ===== УНИВЕРСАЛЬНАЯ ПРОВЕРКА НА СБОРНИКИ МУЛЬТФИЛЬМОВ =====
        # Проверяем для ЛЮБОГО канала
        mf_count = text_lower.count('м/ф') + text_lower.count('m/ф')
        if mf_count >= 2:
            # Можно добавить дополнительные проверки
            # Например, проверять наличие слова "мультфильм" в тексте
            if 'мультфильм' in text_lower or 'мультсериал' in text_lower:
                return True, 'мультфильм'
            # Или просто возвращать при 2+ мультфильмах
            return True, 'мультфильм'

        # Специальная обработка для МИР. Обрабатывает названия по типу 
        # 'Худ.фильм/Сериал (Сериал "Меч". ("ЕДИНСТВЕННЫЙ ВЫХОД") 14, 15, 16, 17 серии)' -> 'меч'
        if self.channel == 'МИР':
            # Ищем название в кавычках после слова "сериал"
            patterns = [
                r'сериал\s+"([^"]+)"',           # сериал "Название"
                r'сериал\s+«([^»]+)»',           # сериал «Название»
                r'сериал\s+([а-яё\s]+?)(?:\s*\(|\.|$)',  # сериал Название (до точки или скобки)
                r'"([^"]+)"',                      # просто "Название"
                r'«([^»]+)»',                      # просто «Название»
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Очищаем от лишних символов
                    title = re.sub(r'[^\w\s-]', '', title)
                    title = re.sub(r'\s+', ' ', title).strip()
                    if title and len(title) > 1:
                        return True, title
            
            # Если не нашли в кавычках, пробуем извлечь из скобок
            match = re.search(r'\([^)]*"([^"]+)"[^)]*\)', text)
            if match:
                title = match.group(1).strip()
                return True, title


            special_phrases = [
                'специальный репортаж', 'славянский базар в витебске'
                ]
            
            for phrase in special_phrases:
                # Создаем паттерн с границами слов для каждого слова во фразе
                if ' ' in phrase:
                    # Для многословных фраз проверяем точное вхождение
                    if phrase in text_lower:
                        cleaned_phrase = self.clean_title(phrase)
                        return True, cleaned_phrase
                else:
                    # Для однословных используем границы слов
                    pattern = r'\b' + re.escape(phrase) + r'\b'
                    if re.search(pattern, text_lower):
                        cleaned_phrase = self.clean_title(phrase)
                        return True, cleaned_phrase
            
            return False, text
        
        
        # Специальная обработка для 2X2
        if self.channel == '2X2' and 'фильм, фильм, фильм' in text_lower:
            # Проверяем, не является ли это специальным именем
            match = re.search(r'фильм,\s*фильм,\s*фильм\.?\s*"([^"]+)"', text_lower)
            if match:
                # Извлекаем название в кавычках
                return True, self.clean_title(match.group(1).strip())
            
            # Если это просто "фильм, фильм, фильм" без названия
            if text_lower.strip() == 'фильм, фильм, фильм' or 'фильм, фильм, фильм' in text_lower:
                return True, 'фильм, фильм, фильм'
        
        # Специальная обработка для ТНТ4
        if self.channel == 'ТНТ4' and 'комеди' in text:
            if 'комеди' in text and 'комедийный' not in text:
                return True, self._replace_whole_word(text, 'комеди', 'камеди')
            
            return False, text
        

        # Специальная обработка для Ю
        if self.channel == 'Ю':
            special_phrases = [
                'маша и медведь', 'супермама', 'ждули'
            ]

            for phrase in special_phrases:
                # Создаем паттерн с границами слов для каждого слова во фразе
                if ' ' in phrase:
                    # Для многословных фраз проверяем точное вхождение
                    if phrase in text_lower:
                        cleaned_phrase = self.clean_title(phrase)
                        return True, cleaned_phrase
                else:
                    # Для однословных используем границы слов
                    pattern = r'\b' + re.escape(phrase) + r'\b'
                    if re.search(pattern, text_lower):
                        cleaned_phrase = self.clean_title(phrase)
                        return True, cleaned_phrase
        

        # Специальная обработка для ЗВЕЗДА
        if self.channel == 'ЗВЕЗДА':
            # Сначала проверяем специальные фразы
            special_phrases = [
                'голоса победы', 'дневники памяти', 'люди донбасса', 
                'военный врач', 'восход победы', 'операция', 'проект "альфа"',
                'армия "трясогузки"'
                ]
            
            for phrase in special_phrases:
                if phrase in text_lower:
                    cleaned_phrase = self.clean_title(phrase)
                    return True, cleaned_phrase
            
            # 5. Если ничего не нашли, идем в общий алгоритм
            return False, text
        
        # Специальная обработка для ТВЦ
        if self.channel == 'ТВЦ':

            if ('док.фильм' in text or 'д/ф' in text or 'худ.фильм' in text or 'х/ф' in text):
                
                # Убираем маркеры
                text = re.sub(r'(д/ф|х/ф|сериал|документальный|док|фильм|худ|художественный)\s*', '', text, flags = re.IGNORECASE)

                # Удаляем тире, чтобы '80-х' -> '80х'
                text = re.sub(r'[-—–]', '', text)
                
                # Удаляем всю пунктуацию, оставляем цифры и буквы
                text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', text)
            
            elif 'специальный репортаж' in text:
                pattern = r'\b' + re.escape('специальный репортаж') + r'\b'
                text = re.sub(pattern, '', text, flags = re.IGNORECASE)
                return True, text
            
            # Сначала проверяем специальные фразы
            special_phrases = [
                '10 самых', '90е', 'девяностые'
                ]
            
            for phrase in special_phrases:
                if phrase in text:
                    cleaned_phrase = self.clean_title(phrase)
                    return True, cleaned_phrase
            
            return False, text
        

        # Специальная обработка для СПАС
        if self.channel == 'СПАС':
            special_phrases = [
                'старцы', 'лики богородицы', 'день ангела', 'искатели', 'утреня', 'дом у большой реки',
                'люди донбасса', 'дети донбасса', 'детство. возвращение', 'апостолы',
                'святые воины', 'лето господне', 'неизвестная европа', 'пилигрим',
                'византия. жизнь после смерти', 'паисий святогорец', 'бесогон', 'ной',
                'патриаршая литературная премия', 'военкоры', 'русские праведники',
                'тропами алании', 'митрополит антоний сурожский', 'притчи', 'знаменный распев',
                'глобус православия', 'добровидение', 'восход победы', 'голос церкви',
                'проповедники'
                ]
            
            for phrase in special_phrases:
                # Создаем паттерн с границами слов для каждого слова во фразе
                if ' ' in phrase:
                    # Для многословных фраз проверяем точное вхождение
                    if phrase in text_lower:
                        cleaned_phrase = self.clean_title(phrase)
                        return True, cleaned_phrase
                else:
                    # Для однословных используем границы слов
                    pattern = r'\b' + re.escape(phrase) + r'\b'
                    if re.search(pattern, text_lower):
                        cleaned_phrase = self.clean_title(phrase)
                        return True, cleaned_phrase

            
        
        # Специальная обработка для СОЛНЦЕ
        if self.channel == 'СОЛНЦЕ':
            # ===== СПЕЦИАЛЬНАЯ ПРОВЕРКА ДЛЯ "ФИЛЬМ.ФИЛЬМ.ФИЛЬМ" =====
            film_patterns = [
                r'фильм\.фильм\.фильм',
                r'фильм\s+фильм\s+фильм',
                r'фильм[.\s]+фильм[.\s]+фильм',
            ]
            
            for pattern in film_patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    if 'фильм.фильм.фильм' in text.lower():
                        return True, 'фильм.фильм.фильм'
                    else:
                        return True, 'фильм фильм фильм'
    
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
        
        return False, text


    def final_clean(self, text: str):
        """
            Метод для финальной очистки текста от всех ненужных символов.
        """
        result = text

        # Финальное схлопывание пробелов        
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\.', ' ', result)  # точка в любом месте
        result = re.sub(r'\,', ' ', result)  # запятая в любом месте
        result = re.sub(r'\b\d{8,}\b', ' ', result) #удаление последовательности из 8ми и более цифр
        result = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', result) # удаление всех символов, кроме букв и цифр

        # Паттерн 1: слово + пробел + число + пробел + й/я/е/ё (фильм 5 й)
        result = re.sub(r'\b([а-яё]+)\s+(\d+)\s+([йяеё])\b', r'\1', result, flags=re.IGNORECASE)
        
        # Паттерн 2: слово + пробел + число + дефис + й/я/е/ё (фильм 5-й)
        result = re.sub(r'\b([а-яё]+)\s+(\d+)[-–—]?([йяеё])\b', r'\1', result, flags=re.IGNORECASE)

        # Удаляем паттерны типа "1 я ч", "2 я ч"
        result = re.sub(r'\b\d+\s+я\s+ч\b', ' ', result, flags = re.IGNORECASE)
        result = re.sub(r'\b\d+\s+я\b', '', result, flags = re.IGNORECASE)
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
        self.df['program_name'] = ''

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

        # Паттерн для возрастных рейтингов, включая варианты со скобками
        pattern = r'\b\s*\(?\s*\+?(0|6|12|14|16|18)\+?\s*\)?\s*\b'
        result = re.sub(pattern, ' ', result, flags = re.IGNORECASE)
        if debug:
            print(f"Удалены возрастные рейтинги: '{result}'")

        old_result = result
        result = re.sub(r'\(\s*[а-яa-z]{1,3}\s*\)', ' ', result, flags=re.IGNORECASE)
        if debug and old_result != result:
            print(f"Удалены одиночные буквы в скобках: '{result}'")

       
        # Универсальный паттерн для удаления любых комбинаций цифр и слова "серии"
        #################################### Удаление серий ####################################
        if re.search(r'сери[яи]|часть|части|сезон', result, flags=re.IGNORECASE):
            old_result = result
            result = re.sub(
                r'\s*(?:\d+(?:\s*,\s*\d+\s*)*\s*)?сери[яиюе]{1,2}(?:\s*\d+(?:\s*,\s*\d+\s*)*)?\s*|\s*\d+(?:\s*,\s*\d+\s*)*\s*',
                ' ', result, flags=re.IGNORECASE
            )
            if debug and old_result != result:
                print(f"Удалены номера с серий: '{result}'")
        ########################################################################################
        
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
    

    def clean_dataframe(self, program_name_column: str = 'Название программы'):
        """
            Финальный метод для предобработки текста. В процессе работы метода создаётся дополнительный столбец в датафрейме, в 
            который записывается очищенное название программы.
        """
        data = self.df.copy()

        # Создаем сет уникальных программ
        unique_programs = data[program_name_column].unique()

        # Создаем словарь маппинга уникальных программ
        mapping = {prog: self.clean(prog) for prog in unique_programs}
        
        # Применяем маппинг к DataFrame. Записываем очищенные названия программ в новый столбец под названием 'program_name'.
        data['program_name'] = data[program_name_column].map(mapping)
        
        # Возвращаем список уникальных очищенных программ
        return list(set(mapping.values())), data
    


class DocumentaryChannelCleaner:
    """
        Класс для очистки названий телепередач с документальным посылом.

        !!! ВАЖНО !!!
        Данный класс предназначен для работы с каналами ЗВЕЗДА, СПАС и ТВЦ
    """
    def __init__(self, channel):
        self.channel = channel

        self.SPECIAL_NAMES = [
            'специальный репортаж', 'мультфильм', 'док сериал фильм', 'док фильм сериал', 'худ фильм сериал',
            'документальный фильм', 'юмористический концерт', 'сериал х ф', 'анимационный фильм', 'х фильмы',
            'короткометражные х фильмы'
        ]


        self.STOP_PATTERNS_GENERAL = [
            r'\bдокументальный\b', r'\bюмористический\b', r'\bспециальный\b', r'\bанимационный\b', 
            r'\bкороткометражные\b', r'\bфильм\b', r'\bсериал\b', r'\bрепортаж\b', r'\bтелесериал\b', 
            r'\bх[\s\-–—/]*фильм[ы]?\b', r'\bмультфильм\b', r'\bспец\b', r'\b[дхмmx][^а-яА-Яa-zA-Z0-9]+ф\b',
            r'\bдок\.?\s*', r'\bхуд\.?\s*', r'\bк[\s\-–—/]*хф\b'
        ]


        self.PROGRAMS_TO_REMAIN = {
            'ТВЦ': ['10 самых', 'военный парад'],
            'ЗВЕЗДА': [
                'голоса победы', 'дневники памяти', 'битва за небо', 
                'секретные материалы', 'хроника победы', 'загадки века',
                'подпольщики', 'шедевры военных музеев', 'битва за днепр'],
            'СПАС': [
                'бесогон', 'голос церкви', 'лествица', 
                'добровидение', 'тропами алании', #'святой', 
                'утреня', 'притчи', 'восход победы', 'военкоры',
                'люди донбасса', 'детство возвращение', 'неизвестная европа',
                'folk без границ'
                ]
        }


        self.STOP_PATTERNS_CHANNELS = {
            'ТВЦ': [r'\bконцерт\b', r'\bсезон\b'],
            'ЗВЕЗДА': [r'\bцикл\b', r'\bконцерт\b'],
            
            'СПАС': [r'\bцикл\b', r'\bкорреспондент\b']
        }

    
    def general_cleaner(self, text: str, debug = False):
        """
            Предварительный очиститель
        """
        
        if debug:
            print(f"\n{'='*80}")
            print(f'ЗАПУСКАЮ ДЕББАГЕР ДЛЯ СТРОКИ:      {text}')
            print(f"{'='*80}")
            print('\n')

        # 1. Приведение текста к нижнему регистру, а также замена буквы ё на е
        text_lower = text.lower().strip().replace('ё', 'е')
        if debug:
            print(f'После приведения к нижнему регистру: {text_lower}')

        
        # ===== УНИВЕРСАЛЬНАЯ ПРОВЕРКА НА СБОРНИКИ МУЛЬТФИЛЬМОВ =====
        # Проверяем для ЛЮБОГО канала
        mf_count = text_lower.count('м/ф') + text_lower.count('m/ф')
        if mf_count >= 2:
            # Можно добавить дополнительные проверки
            # Например, проверять наличие слова "мультфильм" в тексте
            if 'мультфильм' in text_lower or 'мультсериал' in text_lower:
                return 'мультфильм'
            # Или просто возвращать при 2+ мультфильмах
            return 'мультфильм'


        # 2. Удаление подстрок типа '№5', '№09'
        text_lower = re.sub(r'[#№]\d+', '', text_lower)
        if debug:
            print(f'После удаления символов с №: {text_lower}')

        # 3. Удаление возрастных ограничений
        pattern = r'\b\s*\(?\s*\+?(0|6|12|14|16|18)\+?\s*\)?\s*\b'
        text_lower = re.sub(pattern, ' ', text_lower, flags = re.IGNORECASE)
        if debug:
            print(f"Удалены возрастные рейтинги: '{text_lower}'")

        # Зачистка от любых 1-2 буквенных обозначений в скобках
        pattern_remove = r'\s*\([а-яa-z]{1,2}\)\s*'
        text_lower = re.sub(pattern_remove, ' ', text_lower, flags=re.IGNORECASE)

        # 3. Удаление пунктуации
        # Удаляем тире, чтобы '80-х' -> '80х'
        #text_lower = re.sub(r'[-—–]', '', text_lower)
        text_lower = re.sub(r'(\d+)[-—–]+|[-—–]+(\d+)', lambda m: m.group(1) or m.group(2), text_lower)
        # Удаляем всю пунктуацию, оставляем цифры и буквы
        text_lower = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', text_lower)
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        if debug:
            print(f'После удаления пунктуации: {text_lower}')

        # 4. Проверка на специальные имена
        for special_name in self.SPECIAL_NAMES:
            if text_lower == special_name.lower().strip():
                if debug:
                    print(Color.GREEN + f'Прошёл проверку на специальное имя: {text_lower}')
                return special_name


        if self.channel in self.PROGRAMS_TO_REMAIN and self.PROGRAMS_TO_REMAIN[self.channel]:
            for program_to_remain in self.PROGRAMS_TO_REMAIN[self.channel]:
                if program_to_remain in text_lower:
                    if debug:
                        print(Color.GREEN + f'Нашёл программу с ключевым именем: {text_lower}')
                    return program_to_remain
                

        # 5. Удаляем номера частей 
        pattern = r'\b(?:\d+\s+(?:часть|части|ч)\b|\b(?:часть|части|ч)\s+\d+(?:\s+\d+)*)\b'

        text_lower = re.sub(pattern, '', text_lower, flags=re.IGNORECASE)

        # 6. Удаляем номера серий (включая перечисления через пробел)
        pattern = r'\b(?:\d+(?:\s+\d+)*\s+(?:серия|серии|сер|с)\b|\b(?:серия|серии|сер|с)\s+\d+(?:\s+\d+)*)\b'
        
        text_lower = re.sub(pattern, '', text_lower, flags=re.IGNORECASE)

        # 7. Удаляем номеров фильмов (включая перечисления через пробел)
        pattern = r'\b(?:\d+\s+(?:фильм|фильма|фильмов|ф)\b|\b(?:фильм|фильма|фильмов)\s+\d+(?:[-\s]*[йой])?(?:\s+\d+(?:[-\s]*[йой])?)*)\b'

        text_lower = re.sub(pattern, '', text_lower, flags=re.IGNORECASE)

        if debug:
            print(f'После удаления номеров серий и частей: {text_lower}')


        sorted_patterns = sorted(self.STOP_PATTERNS_GENERAL, key = len, reverse = True)
        for pattern in sorted_patterns:
            text_lower = re.sub(r'\s*' + pattern + r'\s*', ' ', text_lower, flags = re.IGNORECASE)

        text_lower = re.sub(r'\s+', ' ', text_lower).strip()

        # Удаление специальных паттернов для каналов
        if self.channel in self.STOP_PATTERNS_CHANNELS and self.STOP_PATTERNS_CHANNELS[self.channel]:
            sorted_special_patterns = sorted(self.STOP_PATTERNS_CHANNELS[self.channel], key = len, reverse = True)
            for pattern in sorted_special_patterns:
                text_lower = re.sub(r'\s*' + pattern + r'\s*', ' ', text_lower, flags = re.IGNORECASE)

            text_lower = re.sub(r'\s+', ' ', text_lower).strip()
            
        if debug:
            print(f'После удаления стоп-паттернов: {text_lower}')


        # Удаление последовательности из 4х и более цифр (МОЖЕТ БЫТЬ СТОИТ ПОМЕНЯТЬ ДЛЯ КАНАЛА "ЗВЕЗДА")
        if self.channel != 'ЗВЕЗДА':
            text_lower = re.sub(r'\b\d{4,}\b', ' ', text_lower)

        else:
            text_lower = re.sub(r'\b\d{6,}\b', ' ', text_lower)


        if debug:
            print(f'После удаления последовательности цифр: "{text_lower}"')

        if text_lower == '':
            print(Color.BOLD + Color.RED + f'‼️ Строка {text} оказалась пустой. Проверьте обработку текста!' + Color.END)
            return ''
            
        else:
            if debug:
                print(f"\n{'='*80}")
                print(f'ПОСЛЕ ФИНАЛЬНОЙ ОЧИСТКИ:           {text_lower}')
                print(f"{'='*80}")
            return text_lower
    

    def clean_dataframe(self, df, program_name_column: str = 'Название программы'):
        """
            Финальный метод для предобработки текста. В процессе работы метода создаётся дополнительный столбец в датафрейме, в 
            который записывается очищенное название программы.
        """
        data = df.copy()

        # Создаем сет уникальных программ
        unique_programs = data[program_name_column].unique()

        # Создаем словарь маппинга уникальных программ
        mapping = {prog: self.general_cleaner(prog) for prog in unique_programs}
        
        # Применяем маппинг к DataFrame. Записываем очищенные названия программ в новый столбец под названием 'program_name'.
        data['program_name'] = data[program_name_column].map(mapping)
        
        # Возвращаем список уникальных очищенных программ
        return list(set(mapping.values())), data

    




class SportChannelCleaner:
    """
        Класс для очистки названий телепередач от служебных пометок,
        стоп-слов и лишних символов в зависимости от канала.

        !!! ВАЖНО !!!
        Данный класс предназначен для работы с каналом МАТЧ ТВ
    """
    def __init__(self, channel):
        self.channel = channel

        # Инициализируем geonamescache
        self.gc = geonamescache.GeonamesCache()
        
        # Загружаем города для нужных стран
        self.country_codes = {
            'russia': 'RU',
            'germany': 'DE',
            'spain': 'ES',
            'italy': 'IT'
        }
        
        # Загружаем все города
        self.cities_by_country = self._load_cities_by_country()
        self.all_european_cities = self._get_all_european_cities()
        
        # Для обратной совместимости
        self.RUSSIAN_CITIES = self.cities_by_country['russia']

        self.STOP_PATTERNS_GLOBAL = [
        # Скобочные формы
        r'\((повтор|премьера|посвящение|сери[яи]|част[ьи]|эпизод[ы]?)\)',
        
        # Слова без скобок
        r'\b(повтор|премьера|посвящение|сезон|сери[яи]|част[ьи]|эпизод[ы]?)\b',

        r'\bд\W*ф\b',

        #r'\bпремьер\s*[-–—]?\s*лиг[аи]?\b',
        
        # Специальные паттерны
        r'№\s*\d+',
        r'\bгг\.\s*',
        r'\bматч за\b', r'\bуефа\b',
        r'\bg[- ]?drive\b',
        r'\bальфа[- ]?банк\b',
        r'\b\d{8,}\b' # последовательность из 8ми и более цифр
        ]


        self.SPECIAL_NAMES = {
            'все на матч', 'мультфильм', 'мульт', 'все о главном', 'что за спорт', 'матч парад',
            'непридуманные истории', 'век нашего спорта', 'география спорта', 'спортивный детектив',
            'что по спорту', 'лица страны', 'культовые', 'команда мечты', 'третий тайм',
            'смешанные единоборства', 'смешанные единоборства ufc', 'смешанные единоборства one ufc', 
            'смешанные единоборства one fc', 'смешанные единоборства аса', 'смешанные единоборства ifc', 
            'смешанные единоборства uralfc', 'бокс bare knuckle fc', 'профессиональный бокс',
            'бокс чемпионат мира', 'бокс кубок победы', 'пляжный волейбол чемпионат европы',
            'пляжный волейбол кубок россии', 'пляжный волейбол чемпионат россии', 'боулинг кубок клб',
            'дзюдо чемпионат мира', 'вечер профессионального бокса', 'плавание кубок россии',
            'художественная гимнастика кубок россии', 'прыжки на лыжах', 'прыжки с трамплина на лыжах',
            'художественная гимнастика международный турнир небесная грация', 'бокс bare knuckle',
            'художественная гимнастика небесная грация', 'футбол матч легенд', 'бильярд', 'дартс',
            'мотоспорт', 'плавание чемпионат мира', 'пляжный волейбол', 'волейбол', 'баскетбол', 'бадминтон',
            'велоспорт', 'прыжки воду', 'гандбол', 'хайдайвинг', 'karate combat', 'теннис', 'аквабайк', 'дзюдо',
            'пляжное регби', 'сноубординг', 'кудо', 'лыжный спорт', 'лыжные гонки', 'легкая атлетика', 'неделя легкой атлетики'
        }

        self.SPECIAL_PATTERNS_FOR_SAVE = [
                        r'\b\d+\s+лет\s+в\s+спорте\b', # n лет в спорте
                        r'\b\d+\s+лет\s+ufc\b'         # лет ufc
        ]


        self.STOP_PATTERNS = {
            'films': [
                        # Полные слова
                        r'\b(анимационный|документальный|худ\.?|фильм|цикл|сериал|мультфильм|мульт)\b',
                        r'\bд\W*ф\b',

                        # Аббревиатуры с любым разделителем (х/ф, х.ф, х ф, и т.д.)
                        r'[хмд]\W*[ф]',  # х/ф, х.ф, м/ф, д/ф и т.д.
                        r'm\W*ф',         # английская 'm'
                        
                        # Специальные случаи
                        r'\bг\.\s*'
            ],
            
            'not_sport': [r'\b(спецрепортаж)\b'],

            'sport': [
                # Финал/полуфинал
                r'\(?(полу)?финал[аы]?\)?',
                r'\b[а-яёa-z]+-?финал[аы]?\b',
                r'\bраунд\b', r'\bплей\s*[-–—]?\s*офф?\b',
                r'\b\d+\s*[-–—/\s]?\s*\d+\s+(?:финала?|полуфинала?|четвертьфинала?)\b' # 1/8 финала, 1/16 финала 
            ]
        }


        # Стоп-слова для удаления по категориям
        self.STOP_WORDS = {
            # для неспортивных трансляций
            'not_sport': {
                'спецрепортаж', 'дайджест'
            },
            # для спортивных трансляций
            'sport': {
                'сезон', 'дайджест', 'место', 'лига ставок', 'betboom', 'olimpbet', 
                'winline', 'fonbet', 'фонбет', 'раунд', 'матч', 'товарищеский', 'озон', 'ozon',
                'фосагро', 'технониколь', 'online', 'on line', 'открытый чемпионат россии', 'кубок гагарина'
            },
            # для фильмов
            'films': {
                'фильм', 'документальный', 'сериал', 'fonbet'
            }
        }
    

    def divide_by_categories(self, data: pd.DataFrame, table_form: str):
        """
            Метод для разделения программ по 4 категориям: 
                - sport (спортивные програмы), 
                - not_sport (неспортивные програмы)
                - films (фильмы/сериалы/мультфильмы)
                - other
            
            Args:
                data: pd.DataFrame: исходная сетка, в которой хотим почистить названия программ от ненужных элементов
                table_form: pd.DataFrame: тип входной таблицы, в которой будем делать преобразования программ. Может быть либо 'vimb', либо 'palomars'.
        """
        if table_form == 'vimb':
            ###################### РАЗДЕЛЕНИЕ ПРОГРАММ ПО КАТЕГОРИЯМ ДЛЯ СЕТКИ VIMB ######################
            categories = {
            'films': ['анимационный', 'художественный фильм', 'документальный', 'худ. фильм', 'мультфильм'],
            'not_sport': ['лица страны', 'матч! парад', 'все на матч', 'новости', 'век нашего спорта', 
                        'титаны', 'география спорта', 'всё о главном', 'спецрепортаж', 'что за спорт', 
                        'непридуманные истории', '10 лет в спорте', 'команда мечты', 'непобедимый', 
                        'культовые', 'что по спорту'],
            'sport': ['автоспорт', 'аквабайк', 'акробатический рок-н-ролл', 'американский футбол', 
                    'айкидо', 'альпинизм', 'армрестлинг', 'бадминтон', 'баскетбол', 'биатлон', 
                    'бильярд', 'бобслей', 'бокс', 'борьба', 'боулинг', 'бейсбол', 'брейкинг', 
                    'бодибилдинг', 'банджо', 'балет', 'бег', 'велоспорт', 'водное поло', 'волейбол', 
                    'вейкбординг', 'виндсерфинг', 'вольная борьба', 'верховая езда', 'гандбол', 'гольф', 
                    'гребля', 'греко-римская борьба', 'гимнастика', 'гиревой спорт', 'горные лыжи', 
                    'дартс', 'дзюдо', 'дайвинг', 'дельтапланеризм', 'джиу-джитсу', 'драгрейсинг', 
                    'единоборства', 'карате', 'кёрлинг', 'конный', 'кхл', 'кикбоксинг', 'капоэйра', 'кудо',
                    'киберспорт', 'конькобежный спорт', 'легкая атлетика', 'лёгкая атлетика', 'лыжи', 
                    'лыжные гонки', 'лыжное двоеборье', 'лапта', 'мхл', 'марафон', 'маунтинбайк', 
                    'мотоспорт', 'мотокросс', 'метание диска', 'нхл', 'настольный теннис', 
                    'настольный футбол', 'ориентирование', 'олимпийские игры', 'падел', 'плавание', 
                    'прыжки в воду', 'прыжки на лыжах', 'пауэрлифтинг', 'парашютный спорт', 'паркур', 
                    'пейнтбол', 'регби', 'рпл', 'рукопашный бой', 'роллер спорт', 'рафтинг', 'реслинг', 
                    'самбо', 'санный спорт', 'скелетон', 'скоростной спуск на коньках', 
                    'смешанные единоборства', 'спортивная гимнастика', 'стрельба из лука', 'серфинг', 
                    'сноуборд', 'скалолазание', 'сквош', 'софтбол', 'теннис', 'триатлон', 
                    'тяжёлая атлетика', 'тхэквондо', 'тайский бокс', 'танцевальный спорт', 
                    'толкание ядра', 'ушу', 'универсальный бой', 'фехтование', 'фигурное катание', 
                    'формула-1', 'фрирайд', 'футбол', 'футзал', 'флорбол', 'фристайл', 'хоккей', 
                    'художественная гимнастика', 'хайдайвинг', 'хоккей на траве', 'чемпионат испании', 
                    'чемпионат италии', 'шахматы', 'шашки', 'шорт-трек', 'экстремальный спорт', 
                    'яхтинг', 'яхтенный спорт']
            }
            
            # Создаем паттерны одной строкой
            patterns = {k: '|'.join(v) for k, v in categories.items()}
            
            # Классификация одной строкой (создаем словарь с результатами)
            result = {}
            remaining = data
            for cat in ['films', 'not_sport', 'sport']:
                mask = remaining['Название программы'].str.contains(patterns[cat], case=False, na=False)
                result[cat] = remaining[mask]
                remaining = remaining[~mask]
            result['other'] = remaining
            
            # Распаковываем результаты
            VIMB_films, VIMB_not_sport, VIMB_sport, VIMB_other = result.values()

            # Списки из программ
            self.programs = {
                'films': list(set(VIMB_films['Название программы'])),
                'sport': list(set(VIMB_sport['Название программы'])),
                'not_sport': list(set(VIMB_not_sport['Название программы'])),
                'other': list(set(VIMB_other['Название программы']))
            }
            return self.programs

        elif table_form == 'palomars':
        
            ###################### РАЗДЕЛЕНИЕ ПРОГРАММ ПО КАТЕГОРИЯМ ДЛЯ РЕАЛЬНОЙ СЕТКИ ######################
            categories = {
            'sport': ['трансляция спортивного', 'репортаж', 'другая музыкально-танцевая'],
            'not_sport': ['передаче о спорте и спортсменах', 'новости', 'география'],
            'films': ['сериал', 'фильм']
            }
            
            # Создаем паттерны
            patterns = {k: '|'.join(v) for k, v in categories.items()}
            
            # Классификация
            plmrs_results = {}
            remaining = data
            
            for cat in ['sport', 'not_sport', 'films']:
                mask = remaining['Жанр'].str.contains(patterns[cat], case=False, na=False)
                plmrs_results[cat] = remaining[mask].reset_index(drop=True)
                remaining = remaining[~mask]
            
            plmrs_results['other'] = remaining.reset_index(drop=True)
            
            # Распаковываем (если нужны отдельные переменные)
            plmrs_sport, plmrs_not_sport, plmrs_films, plmrs_other = plmrs_results.values()
        
            # Списки из программ
            self.programs = {
                'films': list(set(plmrs_films['Название программы'])),
                'sport': list(set(plmrs_sport['Название программы'])),
                'not_sport': list(set(plmrs_not_sport['Название программы'])),
                'other': list(set(plmrs_other['Название программы']))
            }

            return self.programs
        
        else:
            print(f"Неизвестный вид таблицы! Выберите либо 'vimb', либо 'palomars'.")
        

    
    def _load_cities_by_country(self):
        """
        Загружает города для каждой страны из geonamescache
        """
        cities_by_country = {
            'russia': set(),
            'germany': set(),
            'spain': set(),
            'italy': set()
        }
        
        # Получаем все города из базы
        all_cities = self.gc.get_cities()
        
        # Словарь для сопоставления кодов стран с нашими ключами
        code_to_key = {v: k for k, v in self.country_codes.items()}
        
        for city_id, city_data in all_cities.items():
            country_code = city_data.get('countrycode')
            
            if country_code in code_to_key:
                country_key = code_to_key[country_code]
                city_name = city_data['name'].lower()
                
                # Добавляем только основное название, если оно не слишком короткое
                if len(city_name) > 2:
                    cities_by_country[country_key].add(city_name)
        
        # Добавляем крупные города для надежности
        major_cities = {
            'russia': ['москва', 'санкт-петербург', 'новосибирск', 'екатеринбург', 
                    'казань', 'нижний новгород', 'ростов-на-дону', 'самара'],
            
            'germany': ['берлин', 'гамбург', 'мюнхен', 'кёльн', 'франкфурт', 
                        'штутгарт', 'дюссельдорф'],
            
            'spain': ['мадрид', 'барселона', 'валенсия', 'севилья', 'малага',
                    'бильбао', 'гранада'],
            
            'italy': ['рим', 'милан', 'неаполь', 'турин', 'флоренция', 'венеция',
                    'болонья']
        }
        
        for country, cities in major_cities.items():
            for city in cities:
                cities_by_country[country].add(city.lower())
        
        return cities_by_country
    
    def _get_all_european_cities(self) -> Set[str]:
        """
        Объединяет все города для удобства
        """
        all_cities = set()
        for country_cities in self.cities_by_country.values():
            all_cities.update(country_cities)
        return all_cities
    
    def _remove_cities_by_country(self, 
                                  text: str, 
                                  countries: List[str] = None, 
                                  debug: bool = False) -> str:
        """
            Удаляет города указанных стран
            
            Args:
                text: исходный текст
                countries: список стран ('russia', 'germany', 'spain', 'italy')
                        если None, удаляет города всех стран
                debug: флаг отладки
        """
        if not text:
            return text
        
        if countries is None:
            cities_to_remove = self.all_european_cities
        else:
            cities_to_remove = set()
            for country in countries:
                if country in self.cities_by_country:
                    cities_to_remove.update(self.cities_by_country[country])
        
        if not cities_to_remove:
            return text
        
        result = text
        
        # Сортируем по длине (сначала длинные названия, чтобы избежать частичных совпадений)
        sorted_cities = sorted(cities_to_remove, key=len, reverse=True)
        
        # Фильтруем слишком короткие названия
        escaped_cities = [re.escape(city) for city in sorted_cities if len(city) > 2]
        
        if not escaped_cities:
            return text
        
        # Разбиваем на чанки для производительности (чтобы regex не был слишком длинным)
        chunk_size = 100
        for i in range(0, len(escaped_cities), chunk_size):
            chunk = escaped_cities[i:i+chunk_size]
            if chunk:
                pattern = r'\b(?:' + '|'.join(chunk) + r')\b'
                result = re.sub(pattern, ' ', result, flags=re.IGNORECASE | re.UNICODE)
        
        # Отдельно обрабатываем города с дефисами
        hyphen_cities = [city for city in sorted_cities if '-' in city and len(city) > 2]
        for city in hyphen_cities:
            parts = city.split('-')
            # Создаем гибкий паттерн для города с дефисом (может быть написан через пробел или дефис)
            flexible_pattern = r'\b' + r'\s*[-–—]?\s*'.join(re.escape(p) for p in parts) + r'\b'
            result = re.sub(flexible_pattern, ' ', result, flags=re.IGNORECASE | re.UNICODE)
        
        # Очищаем лишние пробелы
        result = re.sub(r'\s+', ' ', result).strip()
        
        if debug and result != text:
            countries_str = ', '.join(countries) if countries else 'все страны'
            print(f"  -> После удаления городов ({countries_str}): '{result}'")
        
        return result
    
    def _remove_cities_context_aware(self, 
                                     text: str, 
                                     program_type: str,
                                     debug: bool = False) -> str:
        """
        Удаляет города с учетом контекста программы
        
        Для спортивных программ удаляем города всех стран
        Для остальных программ удаляем только российские города
        """
        if not text:
            return text
        
        result = text
        
        # Для спортивных программ удаляем города всех европейских стран
        if program_type == 'sport':
            result = self._remove_cities_by_country(
                result, 
                countries=['russia', 'germany', 'spain', 'italy'],
                debug=debug
            )
            
            # Дополнительные паттерны для спортивного контекста
            # (матч в городе, турнир в городе и т.д.)
            if self.all_european_cities:
                # Берем первые 50 городов для примера (можно увеличить при необходимости)
                sample_cities = list(self.all_european_cities)[:100]
                cities_pattern = '|'.join(re.escape(city) for city in sample_cities if len(city) > 2)
                
                if cities_pattern:
                    context_patterns = [
                        r'\bв\s+(?:' + cities_pattern + r')\b',
                        r'\bиз\s+(?:' + cities_pattern + r')\b',
                        r'\b(?:' + cities_pattern + r')\s+(?:арена|стадион|дворец|сити)\b',
                    ]
                    
                    for pattern in context_patterns:
                        old_result = result
                        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE | re.UNICODE)
                        if debug and old_result != result:
                            print(f"    -> Контекстный паттерн сработал")
        
        # Для фильмов и других программ удаляем только российские города
        elif program_type in ['films', 'not_sport', 'other']:
            result = self._remove_cities_by_country(
                result,
                countries=['russia'],
                debug=debug
            )
        
        result = re.sub(r'\s+', ' ', result).strip()
        return result


    def final_cleaning(self, text):
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
            text = re.sub(pattern, ' ', text, flags = re.IGNORECASE)
            text = re.sub(r'^\s+', ' ', text)  # удаляем пробелы в начале после удаления
        
        # Применяем паттерны для конца строки
        for pattern in end_patterns:
            text = re.sub(pattern, ' ', text, flags = re.IGNORECASE)
            text = re.sub(r'\s+$', ' ', text)  # удаляем пробелы в конце после удаления
        
        # Удаляем точки, которые могли остаться между словами (но не внутри слов)
        # Например, "2 . тяжеловес" -> "2 тяжеловес" (точка уже убрана выше)
        
        # Знаки пунктуации - удаляем только в начале и конце
        punctuaction_marks = [
            '.', ',', '\\', '/', '|', ':', ';', '*', '?', '<', '>', '~', '`',
            '!', '@', '#', '$', '%', '^', '&', '(', ')', '+', 
            '=', '[', ']', '{', '}', '-', '№', '"', "'", '«', '»'
        ]
        escaped_marks = ' '.join(re.escape(mark) for mark in punctuaction_marks)
        punctuation_pattern = f'[{escaped_marks}]'
        
        # Удаляем знаки пунктуации в начале и конце
        text = re.sub(f'^{punctuation_pattern}+\\s*', ' ', text)
        text = re.sub(f'\\s*{punctuation_pattern}+$', ' ', text)
        
        # ДОПОЛНИТЕЛЬНО: удаляем одиночные точки с пробелами (как в "2 . тяжеловес")
        text = re.sub(r'\s+\.\s+', ' ', text)  # " . " -> " "
        text = re.sub(r'^\s*\.\s*', ' ', text)  # точка в начале
        text = re.sub(r'\s*\.\s*$', ' ', text)  # точка в конце
        
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Финальная проверка: если осталось что-то типа "2 тяжеловес", удаляем число в начале
        text = re.sub(r'^\d+\s+', ' ', text)

        text = re.sub(r'\.', '', text)  # точка в любом месте
        text = re.sub(r'\,', '', text)  # запятая в любом месте
        text = re.sub(r'\s+', ' ', text).strip() # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
        
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


    def _process_by_program_type(self, 
                            text: str,
                            text_lowered_init: str,
                            program_type: str,
                            stop_words_specific: set = None,
                            stop_patterns: list = None,
                            special_names: set = None,
                            debug: bool = False) -> str:
        """
            Обработка текста в зависимости от типа программы (ЭТАП 5)
            
            Args:
                text: исходный текст
                text_lowered_init: исходный текст в нижнем регистре
                program_type: тип программы ('films', 'sport', 'not_sport', 'other')
                stop_words_specific: специфичные стоп-слова для типа программы
                stop_patterns: паттерны для удаления
                special_names: специальные имена для сохранения
                debug: флаг отладки
                
            Returns:
                обработанный текст
        """
        result = text

        words_to_remove = stop_words_specific if stop_words_specific else set()
        
        if debug:
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
            
            # ВОЗВРАЩАЕМ ПОСЛЕ ВСЕХ ОПЕРАЦИЙ, А НЕ ВНУТРИ ЦИКЛА
            return result.strip()
        
        elif program_type == 'sport':
            # Удаление подстрок типа '3е место'
            result = re.sub(r'\b\d+\s*\-?\s*[еой]?\s*место\b', '', result, flags=re.IGNORECASE)
            
            # Для спортивных программ - только стоп-слова
            for stop_word in words_to_remove:
                pattern = r'\b' + re.escape(stop_word) + r'\b'
                old = result
                result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.UNICODE)
                result = re.sub(r'\s+', ' ', result).strip()
                if debug and old != result:
                    print(f"    Стоп-слово '{stop_word}' -> '{result}'")
            
            # ВОЗВРАЩАЕМ ПОСЛЕ ВСЕХ ОПЕРАЦИЙ, А НЕ ВНУТРИ ЦИКЛА
            return result.strip()
        
        elif program_type in ['not_sport', 'other']:
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
                print(f"    '(' in text_lower? {'(' in text_lowered_init}")
            
            if (not result.strip() or len(result.strip()) < 3) and '(' in text_lowered_init:
                if debug:
                    print(f"  -> Условие выполнено, ищем скобки в '{text_lowered_init}'")
                
                match = re.search(r'\(([^)]+)\)', text_lowered_init)
                if match:
                    result = match.group(1).strip()
                    if debug:
                        print(f"  -> Извлечено из скобок: '{result}'")
            
            # ВСЕГДА ВОЗВРАЩАЕМ РЕЗУЛЬТАТ (даже если не было извлечения из скобок)
            return result.strip()
        
        # На случай, если program_type не совпал ни с одним из условий
        return result.strip()


    def clean(
        self,
        text: str,
        program_type: str = 'other',  # 'films', 'sport', 'not_sport', 'other'
        stop_words_specific: set = None,
        stop_patterns: list = None,
        special_names: set = None,
        special_chars_to_remove: list = SPECIAL_CHARS_TO_REMOVE_GLOBAL,
        stop_patterns_general: list = None,
        special_patterns: set = None,
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
        
        # В теле функции подставляем значения по умолчанию
        if special_names is None:
            special_names = self.SPECIAL_NAMES
        
        if special_chars_to_remove is None:
            special_chars_to_remove = SPECIAL_CHARS_TO_REMOVE_GLOBAL
        
        if stop_patterns_general is None:
            stop_patterns_general = self.STOP_PATTERNS_GLOBAL
        
        if special_patterns is None:
            special_patterns = self.SPECIAL_PATTERNS_FOR_SAVE
        
        # ================================= ЭТАП 1: ПОДГОТОВКА =================================
        # Замена буквы ё на е, если требуется
        text_lower = text.lower().strip().replace('ё', 'е')
        result = text_lower

        # Удаление возрастных рейтингов
        old_result = result
        result = re.sub(r'\s*\d+\+', ' ', result)
        if debug and old_result != result:
            print(f"После удаления возрастных рейтингов: '{result}'")

        # Удаляем упоминания сезонов/частей в конце
        result = re.sub(
            r'\.?\s*(?:\d+[-]?[яи]?\s*(?:сез(?:он|я)?|часть|сери[яи]?)|(?:сез(?:он|я)?|часть|сери[яи]?)\s*\d+[-]?[яи]?)\s*', 
            ' ', 
            result, flags = re.IGNORECASE
        )
        if debug:
            print(f"После удаления номеров серий: '{result}'")


        # Удаляем упоминания финалов, полуфиналов и тд.
        result = re.sub(
            r'\b\d+\s*[-–—/\s]?\s*\d+\s+(?:финала?|полуфинала?|четвертьфинала?)\b',
            ' ', 
            result, flags = re.IGNORECASE
        )

        # Удаление городов
        if hasattr(self, '_remove_cities_context_aware'):
            result = self._remove_cities_context_aware(result, program_type, debug)
        
        # Замена названия 'футбол чемпионат россии премьер-лига' на 'футбол рпл'
        result = re.sub(r'\.', '', result)
        result = result.strip().replace('футбол чемпионат россии премьер-лига', 'футбол рпл')
        
        if debug:
            print(f"ЭТАП 1 - После подготовки: '{result}'")
        
        # ================================= ЭТАП 2: ОЧИСТКА ОТ ГЛОБАЛЬНЫХ ПАТТЕРНОВ =================================
        if stop_patterns_general:
            for pattern in stop_patterns_general:
                result = re.sub(r'\s*' + pattern + r'\s*', ' ', result, flags = re.IGNORECASE | re.UNICODE)
                result = result.strip()

                if debug:
                    print(f"  -> После удаления стоп-слов: '{result}'")
        
        # =================================ЭТАП 3: ПРОВЕРКА НА СПЕЦИАЛЬНЫЕ ИМЕНА =================================
        # Сначала удаляем стоп-слова, используя новый метод
        result_without_stopwords = self._process_by_program_type(
            text = result,
            text_lowered_init = text_lower,
            program_type = program_type,
            stop_words_specific = stop_words_specific,
            stop_patterns = stop_patterns,
            special_names = special_names,
            debug = debug
        )

        # Теперь проверяем на специальные имена
        if special_names and result_without_stopwords and len(result_without_stopwords.strip()) > 2:
            # Подготавливаем строку для проверки
            result_for_check = re.sub(r'[^\w\s]', ' ', result_without_stopwords)
            result_for_check = re.sub(r'\.', ' ', result_for_check)
            result_for_check = re.sub(r'\s+', ' ', result_for_check).strip()

            # Удаляем специальные символы
            if special_chars_to_remove:
                for char in special_chars_to_remove:
                    result_for_check = result_for_check.replace(char, ' ')

            result_for_check = re.sub(r'\s+', ' ', result_for_check).strip()
            
            if debug:
                print(f"ЭТАП 3 - Проверка special_names после удаления стоп-слов: '{result_for_check}'")
            
            # Проверяем, содержит ли строка любое специальное имя
            for special_name in special_names:
                if special_name in result_for_check:
                    if debug:
                        print(f"  -> НАЙДЕНО специальное имя: '{special_name}' в строке")
                    
                    special_name = re.sub(r'\.', '', special_name)
                    special_name = re.sub(r'\,', '', special_name)
                    special_name = re.sub(r'\s+', ' ', special_name).strip()
                    return special_name.strip()
        
        # ================================= ЭТАП 4: ПРОВЕРКА НА СПЕЦИАЛЬНЫЕ ПАТТЕРНЫ =================================
        if special_patterns:
            # Удаляем от специальных символов
            if special_chars_to_remove:
                for char in special_chars_to_remove:
                    result = result.replace(char, ' ')

                for pattern in special_patterns:
                    match = re.search(pattern, result, flags = re.IGNORECASE | re.UNICODE)
                    if match:
                        found_text = match.group(0)  # получаем всю найденную подстроку
                        if debug:
                            print(f"ЭТАП 4 - Специальный паттерн '{pattern}' найден")
                            print(f"  -> Найденная подстрока: '{found_text}'")
                            print(f"  -> Возвращаем: '{found_text.strip()}'")
                        
                        found_text = re.sub(r'\.', '', found_text)  # точка в любом месте
                        found_text = re.sub(r'\,', '', found_text)  # запятая в любом месте
                        found_text = re.sub(r'\s+', ' ', found_text).strip() # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
                        return found_text.strip()
        
        # === ЭТАП 5: ОБРАБОТКА В ЗАВИСИМОСТИ ОТ ТИПА ===
        result = self._process_by_program_type(
                text = result,
                text_lowered_init = text_lower,
                program_type = program_type,
                stop_words_specific = stop_words_specific,
                stop_patterns = stop_patterns,
                special_names = special_names,
                debug = debug
        )
        
        # Проверяем специальные имена после обработки
        if special_names and result and result.strip() in special_names:
            if debug:
                print(f"  -> После обработки получено специальное имя: '{result}'")
            
            result = re.sub(r'\.', '', result)
            result = re.sub(r'\,', '', result)
            result = re.sub(r'\s+', ' ', result).strip()
            return result.strip()
        
        # ================================= ЭТАП 7-10: ОСТАЛЬНАЯ ОЧИСТКА =================================
        if debug:
            print(f"ЭТАП 7 - Удаление скобок и кавычек")
        result = result.replace('"', ' ').replace("'", '')
        result = result.replace('«', ' ').replace('»', '')
        result = result.replace('(', ' ').replace(')', '')
        result = result.replace('[', ' ').replace(']', '')
        result = result.replace('{', ' ').replace('}', '')
        
        if debug:
            print(f"  После ЭТАПА 7: '{result}'")
            print(f"ЭТАП 8 - Базовая очистка")
        
        result = re.sub(r'\s+', ' ', result).strip()
        
        if debug:
            print(f"  После ЭТАПА 8: '{result}'")
            print(f"ЭТАП 9 - Удаление спецсимволов")
        
        if special_chars_to_remove:
            for char in special_chars_to_remove:
                result = result.replace(char, ' ')
        
        result = re.sub(r'^\s*\.\s*', ' ', result)  # точка в начале
        result = re.sub(r'\s*\.\s*$', ' ', result)  # точка в конце
        result = re.sub(r'^\s*\,\s*', ' ', result)  # точка в начале
        result = re.sub(r'\s*\,\s*$', ' ', result)  # точка в конце
        result = re.sub(r'\s*\,\s*$', ' ', result)  # точка в конце

        if debug:
            print(f"  После ЭТАПА 9: '{result}'")
            print(f"ЭТАП 10 - Финальная очистка")
        
        result = self.final_cleaning(result)
        
        if debug:
            print(f"  После ЭТАПА 10: '{result}'")
        
         # ================================= ЭТАП 11: ОСТАЛЬНАЯ ОЧИСТКА =================================
        # Финальная проверка
        if not result or len(result) < 3:
            if debug:
                print(f"Результат пустой или короткий, проверяем special_names")
            if special_names and text_lower.strip() in special_names:
                if debug:
                    print(f"  -> Возвращаем специальное имя: '{text_lower.strip()}'")
                
                text_lower = re.sub(r'\.', '', text_lower)  # точка в любом месте
                text_lower = re.sub(r'\,', '', text_lower)  # запятая в любом месте
                text_lower = re.sub(r'\s+', ' ', text_lower).strip() # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
                return text_lower.strip()
        
        if debug:
            print(f"  После ЭТАПА 11: '{result}'")
            print('='*50)


        # Удаляем одиночные буквы, которые являются отдельными словами (окружены пробелами или границами слов)
        result = re.sub(r'\s+[а-яёa-z]\s+', ' ', result, flags = re.IGNORECASE)  # между словами
        result = re.sub(r'^\s*[а-яёa-z]\s+', '', result, flags = re.IGNORECASE)  # в начале строки
        result = re.sub(r'\s+[а-яёa-z]\s*$', '', result, flags = re.IGNORECASE)  # в конце строки
        
        result = re.sub(r'\.', '', result)  # точка в любом месте
        result = re.sub(r'\,', '', result)  # запятая в любом месте
        result = re.sub(r'\b\d{4,}\b', '', result) #удаление последовательности из 8ми и более цифр
        result = re.sub(r'\s+', ' ', result).strip() # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами


        # ================================= ЭТАП 12: УДАЛЕНИЕ ДУБЛИКАТОВ =================================
        result_splited = result.split(' ')
        # Разбиваем на слова

        # Удаляем слова, которые уже встречались как подстроки в предыдущих словах
        unique_words = []
        seen_substrings = set()

        for word in result_splited:
            # Проверяем, не содержится ли это слово полностью в уже встреченных словах
            is_duplicate = False
            for seen_word in seen_substrings:
                if word in seen_word:  # только проверка, что слово содержится в предыдущем
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_words.append(word)
                seen_substrings.add(word)

        result = ' '.join(unique_words)

        result = re.sub(r'\.', '', result)  # точка в любом месте
        result = re.sub(r'\,', '', result)  # запятая в любом месте
        result = re.sub(r'\s+', ' ', result).strip() # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами

        if debug:
            print(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: '{result}'")
            print('='*50)
        

        if result == '':
            # Если результат пустой, но исходный текст содержал специальное имя
            for special_name in special_names:
                if special_name in text_lower:
                    if debug:
                        print(f"  -> Пустой результат, но найдено специальное имя: '{special_name}'")
                    return special_name
                
            print(Color.BOLD + Color.RED + f'‼️ Для {self.channel} строка {text} оказалась пустой. Проверьте обработку текста!' + Color.END)

        return result
    

    def clean_programs(self, data, table_form: str, program_name_column: str = 'Название программы'):
        """
            Метод для зачистки названий программ для всевозможных категорий
        """
        # ЭТАП 1: Инициализация программ по категориям
        self.programs = self.divide_by_categories(data, table_form)
        
        # ЭТАП 2: Маппинг категорий
        category_mapping = {
            'sport': ('sport', self.STOP_WORDS['sport'], self.STOP_PATTERNS['sport']),
            'not_sport': ('not_sport', self.STOP_WORDS['not_sport'], self.STOP_PATTERNS['not_sport']),
            'films': ('films', self.STOP_WORDS['films'], self.STOP_PATTERNS['films']),
            'other': ('not_sport', self.STOP_WORDS['not_sport'], self.STOP_PATTERNS['not_sport'])
        }
        
        # ЭТАП 3: Создаем копию датафрейма
        result_df = data.copy()
        
        # ЭТАП 4: Создаем общий маппинг для всех программ
        all_mappings = {}
        
        for category, (prog_type, words, patterns) in category_mapping.items():
            if category in self.programs:
                # Получаем уникальные программы для категории
                unique_category_programs = list(set(self.programs[category]))
                
                # Создаем маппинг для уникальных программ категории
                category_mapping_dict = {
                    prog: self.clean(
                        text = prog, 
                        program_type = prog_type, 
                        stop_words_specific = words, 
                        stop_patterns = patterns
                    ) 
                    for prog in unique_category_programs
                }
                
                # Добавляем в общий маппинг
                all_mappings.update(category_mapping_dict)
        
        # ЭТАП 5: Применяем маппинг к датафрейму
        result_df['program_name'] = result_df[program_name_column].map(all_mappings)
        
        # ЭТАП 6: Получаем уникальные очищенные программы с сохранением порядка
        unique_programs = []
        seen = set()
        for prog in all_mappings.values():
            if prog not in seen:
                unique_programs.append(prog)
                seen.add(prog)
        
        return unique_programs, result_df



class MusicChannelCleaner:
    """
        Класс для предобработки названий программ на канале МузТВ
    """
    def __init__(self, channel):
        self.channel = channel

        self.STOP_PATTERNS = [
            r'\bдокументальный\b', r'\bфильм\b',
            r'\bспец\b', r'\bд\W*ф\b', r'\bдок\.?\s*', r'(?:нон|non)[\s\-]?(?:стоп|stop)', r'\bbest\b', r'\bлучшее\b'
        ]

    def divide_programs_by_categories(self, df: pd.DataFrame):
        """
            Метод для разделения программ на различные группы в зависимости от названия передачи
        """
        categories = {
            # Категория для программ, содержащих следующие слова в своем названии.
            # Названия будут зачищаться таким образом, чтобы на выходе оставались только словосочетания, указанные в скобках
            'charts': [
                '10 самых', 'хит сториз', 'битва поколений', 
                'новогодний чарт', 'приехали!', 'самый лучший день',
                'лихие хиты', 'моя волна', 'очень караочен',
                'звезда на замене', 'янамузтв', 'премия муз-тв', 'топ 30',
                'тор 30', 'концерт муз-тв в день города', 'концерт муз-тв ко дню города',
                'день всех влюбленных на муз-тв', 'праздничный концерт муз-тв'
            ],

            # Категория для программ, содержащих слово 'концерт' в своем названии
            'concert': ['концерт'],

            # Категория для программ, содержащих слово 'ВК' в своем названии
            'vk': [
                'громкий вопрос', 'контакты', 'меломан',
                'меломаны', 'натальная карта', 'фест', 'под шубой'
            ],

             # Категория для фильмов
            'films': [
                'документальный фильм', 'док. фильм'
            ]    
        }

        variants_pattern = '|'.join(categories['vk'])
        patterns_vk = rf'ВК\s*[\(\[{{]?\s*(?:{variants_pattern})\s*[\)\]}}]?'

        vk_mask = df['Название программы'].str.contains(patterns_vk, case = False, na = False, regex = True)
        df_vk = df[vk_mask]
        
        # Оставшийся вимб
        df_remained = df[~vk_mask]

        # Создаем паттерны одной строкой
        patterns = {k: '|'.join(v) for k, v in categories.items()}

        # Классификация одной строкой (создаем словарь с результатами)
        result = {}
        remaining = df_remained
        for cat in ['charts', 'concert', 'films']:
            mask = remaining['Название программы'].str.contains(patterns[cat], case = False, na = False)
            result[cat] = remaining[mask]
            remaining = remaining[~mask]
        result['other'] = remaining

         # Распаковываем результаты
        df_charts, df_concert, df_films, df_other = result.values()

        # Словарь с данными и названиями
        data_frames = {
            'concert': df_concert,
            'charts': df_charts,
            'vk': df_vk,
            'films': df_films,
            'other': df_other
        }
        
        # Добавляем только непустые DataFrame
        self.programs = {}
        for key, df in data_frames.items():
            if not df.empty:
                programs = list(set(df['Название программы']))
                self.programs[key] = programs
            else:
                self.programs[key] = []
                print(f"⚠️ Категория '{key}': список пуст (DataFrame пустой)")
                
        return self.programs
        


    def general_preprocess_text(self, text: str):
        """
            Метод по общей обработке текста
        """
        # Приведение к нижнему регистру
        #text_lowered  = text.lower().strip()
    
        # Замена буквы е на ё
        text_lower = text.lower().strip().replace('ё', 'е')
    
        # Удаление подстрок типа '№5', '№09'
        text_lower = re.sub(r'№\d+', '', text_lower)
    
        sorted_patterns = sorted(self.STOP_PATTERNS, key = len, reverse = True)
            
        for pattern in sorted_patterns:
            text_lower = re.sub(r'\s*' + pattern + r'\s*', ' ', text_lower, flags = re.IGNORECASE)
    
        stop_words_special = ['специальный выпуск', 'лучшее', 'best']
        for stop_word in stop_words_special:
            if stop_word in text_lower:
                pattern = r'\b' + re.escape(stop_word) + r'\b'
                text_lower = re.sub(pattern, '', text_lower, flags = re.IGNORECASE)
                
        return text_lower.strip()
    
    
    def clean_special_elements(self, text: str, debug = False):
        
        if debug:
            print(f'Исходная строка {text}')
            print(f"\n{'='*80}")
            
        result = text
        # Основная предобработка
        result = self.general_preprocess_text(result)

        if 'концерт муз-тв ко дню города' in result:
            return 'концерт муз-тв ко дню города'
        
        elif 'концерт муз-тв в день города' in result:
            return 'концерт муз-тв в день города'

        elif 'вк фест' in result:
            return 'вк фест'
        
        # Удаляем стоп-слова
        stop_words_special = ['концерт', 'вк']
        for stop_word in stop_words_special:
            if stop_word in result:
                pattern = r'\b' + re.escape(stop_word) + r'\b'
                result = re.sub(pattern, '', result, flags = re.IGNORECASE)
        if debug:
            print(f'После удаления стоп-слов: "{result}"')
    
        # Удаляем тире, чтобы '80-х' -> '80х'
        result = re.sub(r'[-—–]', '', result)
        
        # Удаляем всю пунктуацию, оставляем цифры и буквы
        result = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', result)
        if debug:
            print(f'После удаления пунктуации: "{result}"')
    
        # Удаляем номера частей 
        pattern = r'\b(?:\d+\s+(?:часть|части|ч)\b|\b(?:часть|части|ч)\s+\d+(?:\s+\d+)*)\b'
    
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
        # Удаление одиночных кейсов, например, "ч 1"
        result = re.sub(r'\b\d+\s+ч\b', '', result, flags=re.IGNORECASE)
        if debug:
            print(f'После удаления номеров частей: "{result}"')
    
        # Удаление последовательности из 4х и более цифр
        result = re.sub(r'\b\d{4,}\b', ' ', result)
        if debug:
            print(f'После удаления последовательности цифр: "{result}"')
    
        # Удаляем одиночные буквы между пробелами
        pattern_letters = r'\s+[a-zA-Zа-яА-ЯёЁ]\s+'
        result = re.sub(pattern_letters, ' ', result)
        result = re.sub(r'\s+', ' ', result).strip()
        if debug:
            print(f'После удаления одиночных букв: "{result}"')
    
        # Удаляем одиночные цифры между пробелами
        pattern = r'\b\d\b'  # Только одна цифра между границами слов
        result = re.sub(pattern, '', result)
        result = re.sub(r'\s+', ' ', result).strip()
        if debug:
            print(f'После удаления одиночных цифр: "{result}"')
        
        if 'премия музтв' in result:
            return 'премия музтв'
    
        
        # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
        result = re.sub(r'\s+', ' ', result).strip()
        if debug:
            print(f"\n{'='*80}")
            print(f'После ФИНАЛЬНОГО форматирования: "{result}"')
            print(f"\n{'='*80}")
    
        return result
    
    
    def leave_main(self, text: str, debug = False):
        """
            Метод, который вычленяет главное по ключевым словам.
        """
        if debug:
            print(f"\n{'='*80}")
            print(f'Исходная строка {text}')
            print(f"\n{'='*80}")
            
        result = text
        # Основная предобработка
        result = self.general_preprocess_text(result)
    
        # Удаляем всю пунктуацию, оставляем цифры и буквы
        result = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', result)
        
        result = re.sub(r'\s+', ' ', result).strip()
        
        if debug:
            print(f'После удаления пунктуации: "{result}"')
    
    
        special_words = [
            '10 самых', 'хит сториз', 'битва поколений', 'новогодний чарт', 'приехали',
            'вк меломан', 'вк громкий вопрос', 'вк натальная карта', 'вк контакты',
            'вк фест', 'вк под шубой', 'самый лучший день', 'лихие хиты', 'моя волна', 'очень караочен',
            'звезда на замене', 'янамузтв', 'премия музтв', 'топ 30', 'тор 30',
            'концерт музтв в день города', 'концерт музтв ко дню города', 'день всех влюбленных на музтв',
            'праздничный концерт музтв'
        ]
        for special_word in special_words:
            if special_word in result:
                if debug:
                    print(f'Нашел особенное слово. На выходе будет: {special_word}')

                return special_word.strip()
    
    
    def film_cleaner(self, text: str, debug = False):
        """
            Метод для предобработки фильмов
        """
        if debug:
            print(f"\n{'='*80}")
            print(f'Исходная строка {text}')
            print(f"\n{'='*80}")
            
        result = text
        # Основная предобработка
        result = self.general_preprocess_text(result)
    
        # Удаляем тире, чтобы '80-х' -> '80х'
        result = re.sub(r'[-—–]', '', result)
        # Удаляем всю пунктуацию, оставляем цифры и буквы
        result = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', result)
        if debug:
            print(f'После удаления пунктуации: "{result}"')
    
        stop_words = ['документальный', 'фильм', 'док']
        
        for stop_word in stop_words:
            if stop_word in result:
                pattern = r'\b' + re.escape(stop_word) + r'\b'
                result = re.sub(pattern, '', result, flags = re.IGNORECASE)
        if debug:
            print(f'После удаления стоп-слов: "{result}"')
    
        # Удаление последовательности из 4х и более цифр
        result = re.sub(r'\b\d{4,}\b', ' ', result)
        if debug:
            print(f'После удаления последовательности цифр: "{result}"')
    
        # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
        result = re.sub(r'\s+', ' ', result).strip()
        if debug:
            print(f"\n{'='*80}")
            print(f'После ФИНАЛЬНОГО форматирования: "{result}"')
            print(f"\n{'='*80}")
    
        return result
    
    
    def simple_cleaner(self, text: str, debug = False):
        """
            Метод для очистки программ с "простым" названием.
        """
        if debug:
            print(f"\n{'='*80}")
            print(f'Исходная строка {text}')
            print(f"\n{'='*80}")
            
        result = text
        # Основная предобработка
        result = self.general_preprocess_text(result)
    
        # Удаляем тире, чтобы '80-х' -> '80х'
        result = re.sub(r'[-—–]', ' ', result)
        # Удаляем всю пунктуацию, оставляем цифры и буквы
        result = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', result)
        if debug:
            print(f'После удаления пунктуации: "{result}"')
    
        # Удаление последовательности из 4х и более цифр
        result = re.sub(r'\b\d{4,}\b', ' ', result)
        if debug:
            print(f'После удаления последовательности цифр: "{result}"')


        words = result.lower().split()
        if 'вк' in words:
            return 'вк контакты'
    
        # Финальное форматирование пробелов. Оставляем ровно 1 пробел между словами
        result = re.sub(r'\s+', ' ', result).strip()
        if debug:
            print(f"\n{'='*80}")
            print(f'После ФИНАЛЬНОГО форматирования: "{result}"')
            print(f"\n{'='*80}")
    
        return result


    def process_program_by_type(self, text: str, category_type: str, debug = False):
        """
            Метод по предобработке текста для каждой из категорий 
        """
        result = text
        
        if category_type == 'concert':
            result = self.clean_special_elements(text, debug)

        elif category_type == 'charts':
            result = self.leave_main(text, debug)

        elif category_type == 'vk':
            result = self.leave_main(text, debug)

        elif category_type == 'films':
            result = self.film_cleaner(text, debug)

        elif category_type == 'other':
            result = self.simple_cleaner(text, debug)

        else:
            result = self.simple_cleaner(text, debug)

        if result == '':
            print(text)
            
        return result


    def clean_programs(self, df, program_name_column: str = 'Название программы'):

        # ЭТАП 1: Разбивка программ по категориям
        self.programs = self.divide_programs_by_categories(df)
    
        # ЭТАП 2: Создаем копию датафрейма
        result_df = df.copy()
    
        # ЭТАП 3: Создаем общий маппинг для всех программ
        all_mappings = {}
        all_cleaned_programs = []  # список для всех очищенных программ
    
        for category, programs in self.programs.items():
            
            category_maps = {}
            
            if len(programs) != 0:
                for program in programs:
                    cleaned_program = self.process_program_by_type(program, category)
                    
                    category_maps[program] = cleaned_program

                    if cleaned_program is None:
                        print(f"  ⚠️ ПРОБЛЕМА: программа '{program}' вернула None в категории '{category}'")
                    
                    all_cleaned_programs.append(cleaned_program)  # добавляем в общий список
                    
                all_mappings[category] = category_maps
    
        # ЭТАП 4: Создаем единый маппинг для всех программ
        # Объединяем все маппинги из разных категорий в один словарь
        unified_mapping = {}
        for category_maps in all_mappings.values():
            unified_mapping.update(category_maps)
    
        # ЭТАП 5: Применяем единый маппинг к датафрейму
        result_df['program_name'] = result_df[program_name_column].map(unified_mapping)
        
        # ЭТАП 6: Получаем уникальные очищенные программы
        # Убираем дубликаты, но сохраняем порядок
        unique_programs = []
        for program in all_cleaned_programs:
            if program not in unique_programs:
                unique_programs.append(program)
        
        return unique_programs, result_df
    
    
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
            small_df: pd.DataFrame
            ):
        """
            Args:
                preprocessor: экземпляр TextPreprocessor (или None для использования по умолчанию)
        """
        #self.preprocessor = preprocessor or TextPreprocessor()
        self.vectorizer = TfidfVectorizer()

        self.List = List
        self.small_list = small_list
        self.df_big = df_big
        self.small_df = small_df
    

    def compare_lists(self, preprocess: bool = True):
        """
            Сравнивает два списка текстов.
        """
        #if preprocess:
        #    processed_list1 = [self.preprocessor.preprocess_text(text) for text in self.List]
        #    processed_list2 = [self.preprocessor.preprocess_text(text) for text in self.small_list]
        #else:
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

        comparison_result = {
                'TF-IDF': len(tfidf_found),
                'fuzzy': len(fuzzy_found),
                'Справочник': len(vocabulary_found),
                'Не найдено': len(programs_not_found)
            }
        
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
                print(Color.BOLD + Color.GREEN + f'УСПЕХ: Для всех программ найдены соответствия' + Color.END)
            else:
                print(f"\n⚠ ВНИМАНИЕ: Не для всех программ найдены соответствия")            
        
        return result_df, programs_not_found, comparison_result
    


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