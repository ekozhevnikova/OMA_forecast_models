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


STOP_PATTERNS = {
    'films': [
                # Полные слова
                r'\b(анимационный|документальный|худ\.?|фильм|цикл|сериал|мультфильм)\b',

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
        r'\b\d+\s*[-–—/\s]?\s*\d+\s+(?:финала?|полуфинала?|четвертьфинала?)\b' # 1/8 финала, 1/16 финала 
    ]
}


# Стоп-слова для удаления по категориям
STOP_WORDS = {
    # для неспортивных трансляций
    'not_sport': {
        'спецрепортаж', 'дайджест'
    },
    # для спортивных трансляций
    'sport': {
        'сезон', 'дайджест', 'место', 'лига ставок', 'betboom', 'olimpbet', 
        'winline', 'fonbet', 'фонбет', 'раунд', 'матч', 'товарищеский'
    },
    # для фильмов
    'films': {
        'фильм', 'документальный', 'сериал', 'fonbet'
    }
}


class SportChannelParsing:
    """
    Класс для очистки названий телепередач от служебных пометок,
    стоп-слов и лишних символов в зависимости от канала.

    !!! ВАЖНО !!!
    Данный класс предназначен для работы с каналом МАТЧ ТВ
    """
    def __init__(self, channel):
        self.channel = channel
    

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
                    'единоборства', 'карате', 'кёрлинг', 'конный', 'кхл', 'кикбоксинг', 'капоэйра', 
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


    def clean(
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
        
        result = self.final_cleaning(result)
        
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

        if result == '':
            print(Color.BOLD + Color.RED + f'‼️ Для {self.channel} строка {text} оказалась пустой. Проверьте обработку текста!' + Color.END)

        return result
    

    def clean_programs(self, data, table_form: str):
        """
            Метод для зачистки названий программ для всевозможных категорий
        """
        # ЭТАП 1: Инициализация программ по категориям
        self.programs = self.divide_by_categories(data, table_form)
        
        # ЭТАП 2: Маппинг категорий
        category_mapping = {
            'sport': ('sport', STOP_WORDS['sport'], STOP_PATTERNS['sport']),
            'not_sport': ('not_sport', STOP_WORDS['not_sport'], STOP_PATTERNS['not_sport']),
            'films': ('films', STOP_WORDS['films'], STOP_PATTERNS['films']),
            'other': ('not_sport', STOP_WORDS['not_sport'], STOP_PATTERNS['not_sport'])
        }
        
        # ЭТАП 3: Зачистка
        cleaned_programs = []
        for category, (prog_type, words, patterns) in category_mapping.items():
            if category in self.programs:
                cleaned_programs.extend([
                    self.clean(text = p, program_type = prog_type, 
                            stop_words_specific = words, stop_patterns = patterns)
                    for p in self.programs[category]
                ])
        
        return list(set(cleaned_programs))  # убираем дубликаты

