import pandas as pd
import numpy as np

class Federal_Comments:
    def __init__(self, df, delta_df):
        self.df = df
        self.delta_df = delta_df

    
    def search_channel_and_info(self, month: str, channel: str):
        """
            Функция для выделения DataFrame с каналом, по которому собираемся искать изменения инвентаря. А также создание словарей 
            Было/Стало из ДатаСетов с Атрибутами, присущими каналу.
            Args:
                df: DataFrame со сравнением прогнозов от двух дат
                channel: Канал ('ПЕРВЫЙ КАНАЛ', 'РОССИЯ 1')
                month: Месяц ('Январь','Май')
            Return:
                old_data: dict, new_data:dict
        """
        #Проверка, что канал есть в списке Федеральный каналов
        if channel not in ['2X2', 'ДОМАШНИЙ', 'ЗВЕЗДА', 
                           'КАРУСЕЛЬ', 'МАТЧ ТВ', 'МИР', 
                           'МУЗ ТВ', 'НТВ', 'ПЕРВЫЙ КАНАЛ', 
                           'ПЯТНИЦА', 'ПЯТЫЙ КАНАЛ', 'РЕН ТВ', 
                           'РОССИЯ 1', 'РОССИЯ 24', 'СОЛНЦЕ', 
                           'СПАС', 'СТС', 'СТС LOVE', 
                           'СУББОТА', 'ТВ ЦЕНТР', 'ТВ-3', 
                           'ТНТ', 'ТНТ 4', 'ЧЕ', 'Ю']:
            raise ValueError('Канал не найден в списке каналов! Выберите другой.')
            
        #Выбор конкретного канала из всего DataFrame
        df_channel = self.df[self.df['Канал'] == channel]
        data_dict = dict.fromkeys(list(set(df_channel['Значения'])))
    
        #Создание словаря со всеми атрибутами, присущими каналу
        for atribute, values in data_dict.items():
            data_dict[atribute] = df_channel[df_channel['Значения'] == atribute]
    
        #Поиск старых значений по каждой из статистик
        old_data = dict.fromkeys(list(set(df_channel['Значения'])))
        for not_changed_atribute, old_values in old_data.items():
            old_data[not_changed_atribute] = float(data_dict[not_changed_atribute][data_dict[not_changed_atribute]['Канал'].isin([channel])][month])
    
        #Поиск новых значений по каждой из статистик
        new_data = dict.fromkeys(list(set(df_channel['Значения'])))
        for changed_atribute, new_values in new_data.items():
            new_data[changed_atribute] = float(data_dict[changed_atribute][data_dict[changed_atribute]['Канал'].isin([channel])][f'{month}.1'])
        return old_data, new_data


    def get_reasons(self, month: str, channel: str):
        """
            Функция для поиска причин, которые повлекли за собой изменение GRP.
            Args:
                df: DataFrame со сравнением прогнозов от двух дат
                channel: Канал ('ПЕРВЫЙ КАНАЛ', 'РОССИЯ 1')
                month: Месяц ('Январь','Май')
            Returns:
                reasons: список причин, согласно которым предположительно произошли изменения инвентаря.
            
        """
        reasons = []
        _month = f'{month}.2'
        df_channel = self.df[self.df['Канал'] == channel]
        data = df_channel[['Канал', 'Значения', _month]].dropna()
        
        atributes = list(data['Значения'])
        for i in range(len(atributes)):
            if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.005:
                reasons.append('Share')
                
            elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.005:
                reasons.append('TTV')
                
            elif atributes[i] == 'КУС' and np.abs(float(data.loc[data['Значения'] == 'КУС', _month])) >= 0.005:
                reasons.append('КУС')
                
            elif atributes[i] == 'Т Общие' and np.abs(float(data.loc[data['Значения'] == 'Т Общие', _month])) >= 0.011:
                reasons.append('Т Общие')
                
            elif atributes[i] == 'GRP ТП канала' and np.abs(float(data.loc[data['Значения'] == 'GRP ТП канала', _month])) >= 1e-5:
                reasons.append('GRP ТП канала')
                
            elif atributes[i] == 'GRP КСР' and np.abs(float(data.loc[data['Значения'] == 'GRP КСР', _month])) >= 0.01:
                reasons.append('GRP КСР')
                
            elif atributes[i] == 'GRP Телемагазины' and np.abs(float(data.loc[data['Значения'] == 'GRP Телемагазины', _month])) >= 1e-3:
                reasons.append('GRP Телемагазины')
        return reasons


    def calculate_delta_GRP(self, month :str, channel: str, changed_statistic: str):
        """
            Функция для расчета delta(GRP НРА), если изменилась Доля.
            При расчете используется GRP НРА (было), Т Общие (было), GRP СП (было), GRP ТП канал (было), TTV (было), КУС общ. (было), Share (стало).
            1. Рассчитывается новый-старый TVR (рейтинг эфира) TVR = TTV (было) * Share (стало) / 100
            2. Рассчитывается новый-старый tvr (рейтинг рекламы) tvr = КУС общ. (было) * TVR
            3. Рассчитывается новый-старый GRP (ОБЩИЙ ИНВЕНТАРЬ) GRP = Т Общие (было) * tvr / 20
            4. Рассчитывается новый-старый GRP НРА (GRP ТП НРА) GRP_NRA = GRP - GRP СП (было) - GRP ТП канал (было)
            5. Рассчитывается дельта между GRP НРА (было) и GRP_NRA
            Args:
                df: DataFrame со сравнением прогнозов
                channel: канал, для которого будем считать разницу
                month: месяц, по которому будем считать
                point_of_sales: ТП Канала. По дефолту считается, что ТП канала есть. Отсутствует на МИРе, СПАСе и Карусели
            Return:
                delta_GRP: функция возвращается разницу по GRP ТП НРА
        """
        old_data, new_data = self.search_channel_and_info(month, channel)
    
        GRP_NRA = 0
        GRP = 0
        TVR = 0
        #Если поменялась Доля
        if changed_statistic == 'Share':
            TVR = old_data['TTV'] * new_data['Share'] / 100
            tvr = old_data['КУС'] * TVR
            GRP = (old_data['Т Общие'] * tvr) / 20
    
        #Если поменялся TTV
        elif changed_statistic == 'TTV':
            TVR = new_data['TTV'] * old_data['Share'] / 100
            tvr = old_data['КУС'] * TVR
            GRP = (old_data['Т Общие'] * tvr) / 20
            
        #Если поменялся КУС
        elif changed_statistic == 'КУС':
            TVR = old_data['TTV'] * old_data['Share'] / 100
            tvr = new_data['КУС'] * TVR
            GRP = (old_data['Т Общие'] * tvr) / 20
    
        #Если поменялись Объемы
        elif changed_statistic == 'Т Общие':
            delta_V = (new_data['Т Общие'] / old_data['Т Общие']) - 1.0
            delta_GRP = old_data['GRP ТП НРА'] * delta_V
            return round(delta_GRP)
    
        #Если поменялись Телемагазины
        elif changed_statistic == 'GRP Телемагазины':
            delta_GRP = new_data['GRP Телемагазины'] - old_data['GRP Телемагазины']
            return round(delta_GRP)
    
        #Если поменялись ТП Канал
        elif changed_statistic == 'GRP ТП канала':
            delta_GRP = new_data['GRP ТП канала'] - old_data['GRP ТП канала']
            return round(delta_GRP)
    
        #Если поменялись ТП Канал
        elif changed_statistic == 'GRP КСР':
            delta_GRP = new_data['GRP КСР'] - old_data['GRP КСР']
            return round(delta_GRP)
    
        #Расчет спонсорства с новым TVR
        tvr_sp = old_data['КУС СП'] * TVR
        GRP_SP = (old_data['Т СП'] * tvr_sp) / 20
            
        #Если есть ТП Канала
        if channel in ['ПЕРВЫЙ КАНАЛ', 'РОССИЯ 1', 'ТНТ 4', 'ТВ-3', 'ТВ ЦЕНТР',
                       'СУББОТА', 'СТС LOVE', 'СТС', 'СОЛНЦЕ', 'РОССИЯ 24', 
                       'РЕН ТВ', 'ПЯТНИЦА', 'НТВ', 'МАТЧ ТВ', 'МУЗ ТВ', 
                       'ЗВЕЗДА', 'ДОМАШНИЙ', 'ПЯТЫЙ КАНАЛ', '2X2', 'ТНТ', 'ЧЕ', 'Ю'
                        ]:
            GRP_NRA = GRP - GRP_SP - old_data['GRP ТП канала']
        elif channel == 'СПАС':
            GRP_NRA = GRP - old_data['GRP Телемагазины'] - GRP_SP
        elif channel in ['КАРУСЕЛЬ', 'МИР']:
            GRP_NRA = GRP - GRP_SP
                
        delta_GRP = GRP_NRA -  old_data['GRP ТП НРА']
        
        return round(delta_GRP)


    def get_reasons_according_to_table_cubik(self, reasons_channels_in_grp, month: str, grp_limit = 30):
        """
            Функция для сопоставления каналов, по которым есть изменения из Федерального кубика (delta_df) и каналов, изменения
            по которым были сгенерированы, исходя их таблицы сравнение прогнозов (forecast_comparison). 
            !!!Изменения для конкретного месяца!!!
            Высчитывается суммарный вклад GRP по всем вылетевшим статистикам для каждого канала. Смотрится, сходится ли 
            это изменение с тем, что отражено в Федеральном кубике.
            Args:
                delta_df: Изменения по дням, исходя из данных Федерального кубика.
                reasons_channels_in_grp: изменения по статистикам в GRP, которые были сгенерированы с помощью forecast_comparison.
                month: месяц, по которому смотрим все изменения инвентаря.
            Returns:
                missing_channels: каналы, по которым есть изменения из федерального кубика, но нет в таблице со сравнением прогнозов.
                sorted_reasons_channels_in_grp: словарь словарей из каналов, статистик, а также значений GRP,
                отсортированных в порядке уменьшения влияния.
                channels_not_exist: каналы из delta_df, которые отсутствуют в sorted_reasons_channels_in_grp.
                channels_not_enough_reasons: каналы, для которыз не нашлось достаточно причин для объяснения изменений.
                
                
        """
        delta_df_ = self.delta_df[self.delta_df['Месяц'] == month]
        channels = list(delta_df_['Канал'])
        merged_channels = list(channels & reasons_channels_in_grp.keys())
        
        # Преобразуем списки в множества
        set_channels = set(channels)
        set_merged_channels = set(merged_channels)
        
        # Находим элементы, которые есть в channels, но отсутствуют в merged_channels
        missing_channels = set_channels - set_merged_channels
    
        res = {}
        for i in range(len(delta_df_)):
            #Выделяем канал, изменения по которому хотим объяснить
            Channel = delta_df_.iloc[i]['Канал']
            #Выделяем изменение
            delta_grp = delta_df_.iloc[i]['Изменение GRP']
            for channel, contributions in reasons_channels_in_grp.items():
                result_contributors = {}
                if Channel == channel:
                    for statistic, value in contributions.items():
                        if value * delta_grp > 0:
                            result_contributors[statistic] = value
                else:
                    continue
                res[channel] = result_contributors
    
        sorted_reasons_channels_in_grp = {}
        for channel, metrics in res.items():
            if metrics:  # Проверяем, что вложенный словарь не пустой
                # Сортируем значения по заданным условиям
                sorted_metrics = sorted(metrics.items(),
                    key = lambda item: item[1] if item[1] > 0 else -item[1],
                    reverse = True  # Убывание для положительных, возрастание для отрицательных
                )
                sorted_reasons_channels_in_grp[channel] = dict(sorted_metrics)
    
        channels_not_exist = []
        channels_not_enough_reasons = []
        for i in range(len(delta_df_)):
            #Выделяем канал, изменения по которому хотим объяснить
            channel = delta_df_.iloc[i]['Канал']
            delta_grp = delta_df_.iloc[i]['Изменение GRP']
            
            if channel not in list(sorted_reasons_channels_in_grp.keys()):
                channels_not_exist.append(channel)
            else:
                for statistic, grp in sorted_reasons_channels_in_grp[channel].items():
                    if np.abs(delta_grp) - np.abs(np.sum(list(sorted_reasons_channels_in_grp[channel].values()))) >= grp_limit:
                        channels_not_enough_reasons.append(channel)
                    elif np.abs(delta_grp) - np.abs(np.sum(list(sorted_reasons_channels_in_grp[channel].values()))) <= grp_limit:
                        continue
        if set(channels_not_exist):
            print(f'Не нашлось релевантных причин в {month} для: {set(channels_not_exist)}.')
        if set(channels_not_enough_reasons):
            print(f'Найдено недостаточно причин в {month} для {set(channels_not_enough_reasons)}.')
        return missing_channels, sorted_reasons_channels_in_grp


    def write_comments(self, dict_of_result_reasons):
        """
            Функция для генерации комментариев на основе изменений статистик в GRP и DataFrame со сравнением прогнозов.
            Args:
                df: DataFrame со сравнением прогнозов
                dict_of_result_reasons: словарь словарей, где KEY: месяц, VALUE: словарь из каналов (key: канал, value: статистики с GRP)
            Returns:
                general_comments: словарь словарей, в котором KEY: месяц, VALUE: словарь из каналов (key:канал, 
                value: словарь из статистик с комментариями.)
        """
        comments_per_month = {}
        for month, reasons_dict in dict_of_result_reasons.items():
            comment_per_channel = {}
            for channel, reasons_dict in reasons_dict.items():
                old_data, new_data = search_channel_and_info(self.df, channel, month)
                comments = {}
                for statistic, delta_grp in reasons_dict.items():
                    if statistic == 'Share':
                        old = np.round(old_data['Share'], 2)
                        new = np.round(new_data['Share'], 2)
                        if reasons_dict['Share'] > 0:
                            comments[statistic] = f'Рост доли с {old} до {new}.'
                        else:
                            comments[statistic] = f'Снижение доли с {old} до {new}.'
                            
                    elif statistic == 'TTV':
                        if reasons_dict['TTV'] > 0:
                            comments[statistic] = 'Рост телесмотрения.'
                        else:
                            comments[statistic] = 'Снижение телесмотрения.'
        
                    elif statistic == 'КУС':
                        if reasons_dict['КУС'] > 0:
                            comments[statistic] = 'Рост КУС.'
                        else:
                            comments[statistic] = 'Снижение КУС.'
        
                    elif statistic == 'GRP Телемагазины':
                        val = reasons_dict['GRP Телемагазины']
                        if val > 0:
                            comments[statistic] = f'Размещение телемагазинов {val} GRP.'
                        else:
                            comments[statistic] = f'Снятие телемагазинов {(-1) * val} GRP.'
        
                    elif statistic == 'Т Общие':
                        val = reasons_dict['Т Общие']
                        if val > 0:
                            comments[statistic] = f'Дооткрытие рекламных объемов {val} GRP.'
                        else:
                            comments[statistic] = f'Сокращение рекламных объемов {(-1) * val} GRP.'
                comment_per_channel[channel] = comments
                comments_per_month[month] = comment_per_channel
                
        #Генерация комментариев по шаблонам
        general_comments = {}
        for month, comment_per_channel in comments_per_month.items():
            channel_comments_new = {}
            for channel, channel_comments in comment_per_channel.items():
                comments_new = []
                for statistic, comments in channel_comments.items():
                    comments_new.append(comments)
                comment_final = ' '.join(comments_new)
                channel_comments_new[channel] = comment_final
                general_comments[month] = channel_comments_new
        return general_comments


    def generate_result_table(self, general_comments, df_limits):
        """
            Генерация выходной таблицы.
            Args:
                general_comments: словарь словарей с комментариями.
                df_limits: таблица с порогами.
                delta_df: Изменения по дням, исходя из данных Федерального кубика.
            Returns:
                data_output: DataFrame с итоговыми комментариями.
        """
        limits = df_limits.T
        limits.reset_index(inplace = True)
        res = []
        for month, month_comments in general_comments.items():
            #Извлечение необходимых данных из кубика
            df_from_cubik = self.delta_df[self.delta_df['Месяц'] == month]
            df_from_cubik = df_from_cubik[['Канал', 'Месяц', 'Дата', 'Изменение GRP']]
    
            df = pd.DataFrame(list(general_comments[month].items()), columns = ['Канал', 'Комментарий'])
            #Добавление столбца с месяцем
            df['Месяц'] = month
            
            #Добавление пустого столбца для комментариев руководителя
            df['Доп столбец'] = ''
            
            #Join комментариев с кубиком
            df_res = pd.merge(df_from_cubik, df, on = ['Канал', 'Месяц'], how = 'left')
            
            #Join комментариев с порогами
            df_res = pd.merge(df_res, limits, on = ['Канал'], how = 'inner')
            df_res = df_res[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
            res.append(df_res)
        data_output = pd.concat(res)
        return data_output


    def get_result(self):
        """
            Функция для получения full-result
            Args:
                df: DataFrame со сравнением прогнозов
                delta_df: DataFrame с изменениями по дням согласно Федеральному кубику
            Return:
                data_output: DataFrame с шаблонными комментариями
        """
        reasons_channels = {}
        result = {}
        self.delta_df = self.delta_df.sort_values(by = 'Месяц')
        months_init = list(self.delta_df['Месяц'])
        months_new = []
        for old_month in months_init:
            month_new = str(old_month).split('\'')[0].title()
            months_new.append(month_new)
            
        # Замена столбца дат на новый конвертированный столбец
        self.delta_df['Месяц'] = self.delta_df['Месяц'].replace(months_init, months_new)
        
        for i in range(len(self.delta_df)):
            #Выделяем канал, изменения по которому хотим объяснить
            channel = self.delta_df.iloc[i]['Канал']
            #Выделяем месяц, по которому наблюдаются существенные изменения
            month = self.delta_df.iloc[i]['Месяц']
        
            reasons = self.get_reasons(month, channel)
            
            contributions = {}
            for changed_statistic in reasons:
                delta_GRP = self.calculate_delta_GRP(month, channel, changed_statistic)
                contributions[changed_statistic] = delta_GRP
        
            #Словарь с изменения по каналам для конкретного месяца
            #reasons_channels[channel] = contributions
        
            #Отсортированный словарь с GRP
            #missing_channels, sorted_reasons_channels_in_grp = get_reasons_according_to_table_cubik(df_summ_need_comment_sorted, reasons_channels, month)
            
            if not month in reasons_channels.keys():
                reasons_channels[month] = {
                    channel: contributions
                }
            else:
                reasons_channels[month][channel] = contributions
            
            #Удаляем пустые элементы из словаря или каналы, по которым нет изменений
            #reasons_channels = {k:v for k,v in reasons_channels.items() if v}
        res = {}
        for month, reasons in reasons_channels.items():
            missing_channels, sorted_reasons_channels_in_grp = self.get_reasons_according_to_table_cubik(reasons, month)
            res[month] = sorted_reasons_channels_in_grp
        
        general_comments = self.write_comments(res)
        data_output = self.generate_result_table(general_comments, df_limits)
        # Замена столбца дат на старый столбец
        data_output['Месяц'] = data_output['Месяц'].replace(months_new, months_init)
        return data_output