import os
import numpy as np
import pandas as pd
from typing import Optional

import warnings
warnings.filterwarnings('ignore')

from io_data.colors import *
from federal.channel_forecast.grid_preprocessing import *


class ChannelAnalysisMaster:
    """
        Класс, в котором реализованы пайплайны для выгрузки данных из БД Mediascope, 
        а также обновление исторической сетки VIMB, освовываясь на отчете Размещение/Сводная таблица.

        !!! В А Ж Н О !!!
        Класс работает только для конкретного канала!
    """
    def __init__(
            self,
            channel: str,
            date_filter: list,
            company_filter: str,
            basedemo_filter: str,
            auedience_file: Optional[str] = None,           # опциональный параметр
            web_file: Optional[str] = None,                 # опциональный параметр
            weighted_share_file: Optional[str] = None,      # опциональный параметр
            new_vimb_grids: Optional[str] = None,           # опциональный параметр
            hist_vimb_file: Optional[str] = None            # опциональный параметр
    ):

        """
            Атрибуты класса (Можно передавать только нужные параметры)
                auedience_file: str: Полный путь к файлу с Total TV Auedience для какого-то конкретного канала.
                web_file: str: Полный путь к файлу с исторической сеткой Mediascope для какого-то конкретного канала.
                weighted_share_file: str: Полный путь к файлу со взвешенной долей и исторической сеткой Mediascope для какого-то конкретного канала.
                new_vimb_grids: Полный путь к файлам с новыми сетками для какого-то конкретного канала.
                hist_vimb_file: Полный путь к файлу с исторической сеткой VIMB для какого-то конкретного канала.
            
        """
        self.channel = channel
        self.date_filter = date_filter
        self.company_filter = company_filter
        self.basedemo_filter = basedemo_filter
        
        # Файлы опциональны - передаем только те, что нужны
        self.auedience_file = auedience_file
        self.web_file = web_file
        self.weighted_share_file = weighted_share_file
        self.new_vimb_grids = new_vimb_grids
        self.hist_vimb_file = hist_vimb_file
        
        # Кэшируем результаты
        self._total_tv_auedience = None
        self._web_df = None

        self.STOP_WORDS = ['погода', 'межпрограммные заставки']
        self.PATTERN = '|'.join(self.STOP_WORDS)
    

    def auedience_pipeline(self):
        """
            Пайплайн для выгрузки и записи Auedience в файл для какого-то конкретного Федерального канала и конкретной БЦА.
        """
        # Если файл не передан, то выгрузка Total Channels Auedience не произойдет.
        if not self.auedience_file:
            raise ValueError('🚨 Для выполнения Auedience пайплайна необходимо указать auedience_file')

        A_parser = AuedienceParser(self.auedience_file)

        # 1. Выгрузка новых данных по Total TV Auedience
        auedience_new = A_parser.auedience_by_slots(self.date_filter, self.company_filter, self.basedemo_filter)

        # 2. Обновление таблицы
        print('🔄 Обновляю файл с Total TV Auedience. Пожалуйста, подождите ...')
        self.total_tv_auedience = A_parser.update_table_auedience(auedience_new)

        # 3. Сохранение в файл
        print('✅ Сохраняю файл с Total TV Auedience. Пожалуйста, подождите ...')
        A_parser.make_style_of_auedience_table(self.total_tv_auedience, 'Sheet1')

        return self.total_tv_auedience
    

    def mediascope_web_pipeline(self):
        """
            Пайплайн для выгрузки и записи исторической сетки Mediascope в файл для какого-то конкретного Федерального канала и конкретной БЦА.
        """
        # Если файл не передан, то выгрузка сетки Palomers не произойдет.
        if not self.web_file:
            raise ValueError('🚨 Для выполнения web пайплайна необходимо указать web_file')

        plmrs_parser = MediascopeParser(self.channel, self.web_file)
        # 1. Выгрузка новых исторических данных
        web_new = plmrs_parser.make_web(self.date_filter, self.company_filter, self.basedemo_filter)

        web_new = web_new[~web_new['Название программы'].str.contains(self.PATTERN, case = False, na = False)]

        # 2. Обновление таблицы
        print('🔄 Обновляю файл с исторической сеткой Mediascope. Пожалуйста, подождите ...')
        self.web_df = plmrs_parser.update_web_table(web_new)

        # 3. Сохранение в файл
        print('✅ Сохраняю файл с исторической сеткой Mediascope. Пожалуйста, подождите ...')
        plmrs_parser.make_style_of_web_table(self.web_df, 'Sheet1')

        # Если нужно вернуть в строковый формат
        web_new['Дата'] =  web_new['Дата'].dt.strftime('%Y-%m-%d')
        web_new['Время выхода'] =  web_new['Время выхода'].dt.strftime('%H:%M:%S')
        web_new['Время окончания'] =  web_new['Время окончания'].dt.strftime('%H:%M:%S')

        return web_new, self.web_df
    

    def plmrs_web_pipeline(
                self,
                web_new,
                start_time_col: str = 'Время выхода', 
                end_time_col: str = 'Время окончания',
                date_col: str = 'Дата'):
        """
            Пайплайн для обновления и записис в файл рассчитанных взвешенных долей для какого-то конкретного Федерального канала и конкретной БЦА.
        """
        # Если файл не передан, то расчет взвешенных долей не будет реализован.
        if not self.weighted_share_file:
            raise ValueError('🚨 Для выполнения web пайплайна необходимо указать weighted_share_file')

        parser = TVPreprocessing(self.channel, self.weighted_share_file, web_new)

        print('📈 Считаю взвешенную долю. Пожалуйста, подождите ...')
        # 1. Расчет взвешенной доли
        new_df, shares = parser.process_daily_weighted_shares(self.total_tv_auedience)

        # 2. Обновление таблицы
        new_df['Дата'] = pd.to_datetime(new_df['Дата'])
        updated = MediascopeParser(self.channel, self.weighted_share_file).update_web_table(new_df)

        # 3. Сохранение в файл
        print('🔄 Обновляю файл со взвешенной долей и исторической сеткой Mediascope. Пожалуйста, подождите ...')
        parser.make_plmrs_style_of_table(updated, 'Sheet1')

        return updated
    

    def vimb_web_pipeline(self):
        """
            Пайплайн для обновления исторической сетки VIMB.
        """
        # Если файлы с новыми сетками, историческими данными не переданы, то пайплайн не запустится.
        if not self.new_vimb_grids:
            raise ValueError('🚨 Для выполнения VIMB пайплайна необходимо указать путь к файлам с новыми сетками VIMB.')

        if not self.hist_vimb_file:
            raise ValueError('🚨 Для выполнения VIMB пайплайна необходимо указать путь к файлу с исторической сеткой VIMB.')

        # 1. Составление таблицы с новой сеткой
        combined = VIMBGridProcessor(self.new_vimb_grids).parse_new_vimb_grids()

        # 2. Обновление файла с историческими данными
        VIMBGridProcessor(self.hist_vimb_file).update_vimb_file(combined)
    

    def unified_pipeline(self, run_all: bool = True, **kwargs):
        """
        Гибкий объединенный пайплайн.
        
        Args:
            run_all: Если True, запускает все доступные пайплайны
            **kwargs: Можно передать какие пайплайны запускать:
                      run_auedience=True/False, run_web=True/False и т.д.
        """
        results = {}
        
        # Определяем, какие пайплайны запускать
        run_auedience = kwargs.get('run_auedience', run_all or self.auedience_file)
        run_web = kwargs.get('run_web', run_all or self.web_file)
        run_plmrs = kwargs.get('run_plmrs', run_all or self.weighted_share_file)
        run_vimb = kwargs.get('run_vimb', run_all or (self.new_vimb_grids and self.hist_vimb_file))
        
        print(Color.BOLD + f'🚀 Начинаю расчет для канала {self.channel}' + Color.END)
        # 1. Audience пайплайн
        if run_auedience and self.auedience_file:
            print(Color.BOLD + Color.VIOLET + '=== 🎬 Запуск выгрузки Total Channels Auedience пайплайна ===' + Color.END)
            results['auedience'] = self.auedience_pipeline()
        
        # 2. Web пайплайн
        if run_web and self.web_file:
            print(Color.BOLD + Color.BLUE + '=== 🌐 Запуск выгрузки сетки Mediascope пайплайна ===' + Color.END)
            results['web'] = self.mediascope_web_pipeline()
        
        # 3. PLMRS пайплайн (требует audience и web)
        if run_plmrs and self.weighted_share_file:
            if 'web' in results and results['web'] is not None:
                if 'auedience' in results and results['auedience'] is not None:
                    print(Color.BOLD + Color.ORANGE + '=== ⚖️ Запуск расчета взвешенных долей ===' + Color.END)
                    web_new, _ = results['web']
                    results['plmrs'] = self.plmrs_web_pipeline(web_new)

                else:
                    # Проверяем существование файла с Total TV Auedience. Без этого не можем продолжить!
                    if not os.path.exists(self.auedience_file):
                        print(Color.BOLD + Color.RED + f'❌ Ошибка: файл c Total TV Auedience для канала {self.channel} не найден: {self.auedience_file}' + Color.END)
                        print('⏭️ Пропускаем PLMRS пайплайн: требуется файл аудитории')
                    
                    else:
                        self.total_tv_auedience = pd.read_excel(self.auedience_file)
                        print(Color.BOLD + Color.GREEN + f'Файл c Total TV Auedience для канала {self.channel} найден!' + Color.END)
                        print(Color.BOLD + Color.ORANGE + '=== ⚖️ Запуск расчета взвешенных долей ===' + Color.END)
                        web_new, _ = results['web']
                        results['plmrs'] = self.plmrs_web_pipeline(web_new)

            else:
                print('⏭️ Пропускаем PLMRS пайплайн: требуется выполнить web пайплайн')
        
        # 4. VIMB пайплайн
        if run_vimb and self.new_vimb_grids and self.hist_vimb_file:
            print(Color.BOLD + Color.GREEN + '=== 📺 Запуск VIMB пайплайна ===' + Color.END)
            results['vimb'] = self.vimb_web_pipeline()
        
        print(Color.BOLD + f'✅ 🏁 Данные для канала {self.channel} успешно выгружены! Спасибо за Ваше ожидание! 😊' + Color.END)
        print('\n')
        
        return results