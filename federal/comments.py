import pandas as pd
import numpy as np
import pymorphy3 as pmrph
import xlsxwriter
from OMA_tools.federal.fed_preprocessing import Federal_Preprocessing

class color:
   BOLD = '\033[1m'
   BLUE = '\033[94m'
   RED = '\033[91m'
   END = '\033[0m'


class Federal_Comments:
    """
        Класс для генерации Федеральных комментариев, исходя из Федерального Кубика
    """
    def __init__(self, df, delta_df):
        self.df = df
        self.delta_df = delta_df
    
    @staticmethod
    def change_channels_name(channel_names_init, df, column_name: str):
        """
            Функция для замена имени канала на нужный для файла Комментарии
            Args:
                channel_names_init: Словарь из каналов, где ключ: имя капсом, значение: нужное название канала для файла Комментарии
                df: Датафрейм, в котором нужно заменить названия каналов, список из каналов, нуждающихся в замене
                column_name: Название колонки, которую будем заменять
            Returns:
                обновленный df с измененными названиями каналов
        """
        channels = list(df[column_name])
        for i in range(len(channels)):
            for channel, name_init in channel_names_init.items():
                if channels[i] == channel:
                    df[column_name].replace({channels[i]: name_init}, inplace = True)
        return df
    
    @staticmethod
    def influence_out_house(kus_file):
        """
            Функция для чтения файла с коэффициентами внедома
            Args:
                kus_file: путь к файлу с прогнозом КУСа.
            Returns:
                KUS_koeff_cleaned: DataFrame c коэффициентами внедома
        """
        KUS_koeff = pd.read_excel(kus_file, sheet_name = 'коэф.внедом', skiprows = 2)
        KUS_koeff = KUS_koeff[['Канал', 'январь.2', 'февраль.2', 'март.2', 'апрель.2', 'май.2',
            'июнь.2', 'июль.2', 'август.2', 'сентябрь.2', 'октябрь.2', 'ноябрь.2',
            'декабрь.2']]
        KUS_koeff_cleaned = KUS_koeff.dropna() 
        
        KUS_koeff_cleaned.rename(columns = {
            'январь.2': 'Январь.2',
            'февраль.2': 'Февраль.2',
            'март.2': 'Март.2',
            'апрель.2': 'Апрель.2',
            'май.2': 'Май.2',
            'июнь.2': 'Июнь.2',
            'июль.2': 'Июль.2',
            'август.2': 'Август.2',
            'сентябрь.2': 'Сентябрь.2',
            'октябрь.2': 'Октябрь.2',
            'ноябрь.2': 'Ноябрь.2',
            'декабрь.2': 'Декабрь.2'
            },
            inplace = True)
        
        #KUS_koeff_cleaned = Federal_Comments.change_channels_name(channel_names_init, KUS_koeff_cleaned, 'Канал')
        KUS_koeff_cleaned['Канал'] = KUS_koeff_cleaned['Канал'].str.upper()
        
        #Изменение столбца с каналами
        channels_need_replace = {
                    '2Х2': '2X2',
                    '5 КАНАЛ': 'ПЯТЫЙ КАНАЛ',
                    'ПЕРВЫЙ': 'ПЕРВЫЙ КАНАЛ',
                    'СТС ЛАВ': 'СТС LOVE',
                    'ТВ3': 'ТВ-3',
                    'ТНТ4': 'ТНТ 4'
                }
        channels_old = list(KUS_koeff_cleaned['Канал'])
        channels_new = []
        for i in range(len(channels_old)):
            channels_new.append(channels_old[i].upper())
        KUS_koeff_cleaned['Канал'] = KUS_koeff_cleaned['Канал'].replace(channels_old, channels_new)
        KUS_koeff_cleaned['Канал'].replace(channels_need_replace, inplace = True)
        return KUS_koeff_cleaned


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
            raise ValueError(f'{channel} не найден в списке каналов! Выберите другой.')
            
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


    def get_reasons(self, month: str, channel: str, kus_df):
        """
            Функция для поиска причин, которые повлекли за собой изменение GRP.
            Args:
                df: DataFrame со сравнением прогнозов от двух дат
                channel: Канал ('ПЕРВЫЙ КАНАЛ', 'РОССИЯ 1')
                month: Месяц ('Январь','Май')
            Returns:
                reasons: список причин, согласно которым предположительно произошли изменения инвентаря.
            
        """
        #Если изменения по коэффициентам внедома составляют больше 0.5 общего изменения КУСа, то это внедом
        criteria = 0.5

        reasons = []
        _month = f'{month}.2'
        df_channel = self.df[self.df['Канал'] == channel]
        data = df_channel[['Канал', 'Значения', _month]].dropna()
        data_copy = df_channel[['Канал', 'Значения', _month]]

        outhouse = float(kus_df.loc[kus_df['Канал'] == channel, _month])
        
        atributes = list(data['Значения'])
        for i in range(len(atributes)):
            if data.iloc[0]['Канал'] in ['НТВ', 'РОССИЯ 1']:
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.001:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.001:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.001 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')

            elif data.iloc[0]['Канал'] in ['ПЯТЫЙ КАНАЛ', 'РЕН ТВ', 'ТНТ', 'СТС']:
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.0015:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.0015:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.0015 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')

            elif data.iloc[0]['Канал'] in ['ДОМАШНИЙ', 'МАТЧ ТВ', 'ПЕРВЫЙ КАНАЛ']:
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.0017:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.0017:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.0017 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')

            elif data.iloc[0]['Канал'] in ['СУББОТА', 'МУЗ ТВ', 'ПЯТНИЦА']:
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.0024:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.0024:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.0024 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')

            elif data.iloc[0]['Канал'] in ['РОССИЯ 24', 'КАРУСЕЛЬ', 'СОЛНЦЕ', 'ЗВЕЗДА']:
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.00265:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.00265:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.00265 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')
            
            elif data.iloc[0]['Канал'] in ['Ю', 'ТВ ЦЕНТР', 'ТВ-3']:
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.003:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.003:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.003 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')
            
            elif data.iloc[0]['Канал'] == 'СПАС':
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.008:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.008:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.008 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')
            
            elif data.iloc[0]['Канал'] in ['ЧЕ', 'ТНТ 4']:
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.0038:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.0038:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.0038 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')
            
            elif data.iloc[0]['Канал'] == '2X2':
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.0067:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.0067:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.0067 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')
            
            elif data.iloc[0]['Канал'] == 'СТС LOVE':
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.0055:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.0055:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.0055 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')
            
            elif data.iloc[0]['Канал'] == 'МИР':
                if atributes[i] == 'Share' and np.abs(float(data.loc[data['Значения'] == 'Share', _month])) >= 0.0045:
                    reasons.append('Share')
                elif atributes[i] == 'TTV' and np.abs(float(data.loc[data['Значения'] == 'TTV', _month])) >= 0.0045:
                    reasons.append('TTV')  
                elif atributes[i] == 'КУС':
                    delta = np.abs(float(data.loc[data['Значения'] == 'КУС', _month]))
                    if delta >= 0.0045 and outhouse < criteria * delta:
                        reasons.append('КУС')
                    else:
                        reasons.append('КУС Внедом')
                
            if atributes[i] == 'Т Общие' and np.abs(float(data.loc[data['Значения'] == 'Т Общие', _month])) >= 0.011:
                reasons.append('Т Общие')
            
            elif atributes[i] == 'GRP ТП канала':
                if np.abs(float(data.loc[data['Значения'] == 'GRP ТП канала', _month])) > 0.025:
                    reasons.append('GRP ТП канала')
                elif np.isnan(float(data.loc[data['Значения'] == 'GRP ТП канала', _month])):
                    #month_2 = f'{month}.1'
                    if np.abs(float(data.loc[data['Значения'] == 'GRP ТП канала', month])) == 0.0 and np.abs(float(data.loc[data['Значения'] == 'GRP ТП канала', f'{month}.1'])) != 0.0:
                        reasons.append('GRP ТП канала')
                    elif np.abs(float(data.loc[data['Значения'] == 'GRP ТП канала', month])) != 0.0 and np.abs(float(data.loc[data['Значения'] == 'GRP ТП канала', f'{month}.1'])) == 0.0:
                        reasons.append('GRP ТП канала')
                
            elif atributes[i] == 'GRP КСР' and np.abs(float(data.loc[data['Значения'] == 'GRP КСР', _month])) >= 0.01:
                reasons.append('GRP КСР')
                
            elif atributes[i] == 'GRP Телемагазины':

                if np.abs(float(data_copy.loc[data_copy['Значения'] == 'GRP Телемагазины', _month])) >= 1e-3:
                    reasons.append('GRP Телемагазины')
                elif np.isnan(float(data_copy.loc[data_copy['Значения'] == 'GRP Телемагазины', _month])):
                    if np.abs(float(data_copy.loc[data_copy['Значения'] == 'GRP Телемагазины', month])) == 0.0 and np.abs(float(data_copy.loc[data_copy['Значения'] == 'GRP Телемагазины', f'{month}.1'])) != 0.0:
                        reasons.append('GRP Телемагазины')
                    elif np.abs(float(data_copy.loc[data_copy['Значения'] == 'GRP Телемагазины', month])) != 0.0 and np.abs(float(data_copy.loc[data_copy['Значения'] == 'GRP Телемагазины', f'{month}.1'])) == 0.0:
                        reasons.append('GRP Телемагазины')
            
            elif atributes[i] == 'GRP СП' and np.abs(float(data.loc[data['Значения'] == 'GRP СП', _month])) >= 1e-3:
                reasons.append('GRP СП')
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
        elif changed_statistic == 'КУС' or changed_statistic == 'КУС Внедом':
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
        
         #Если поменялись ТП Канал
        elif changed_statistic == 'GRP СП':
            delta_GRP = new_data['GRP СП'] - old_data['GRP СП']
            return round(delta_GRP)
    
        #Расчет спонсорства с новым TVR
        tvr_sp = old_data['КУС СП'] * TVR
        GRP_SP = (old_data['Т СП'] * tvr_sp) / 20
            
        #Если есть ТП Канала
        if channel in ['ТНТ 4', 'ТВ-3', 'ТВ ЦЕНТР',
                    'СУББОТА', 'СТС LOVE', 'СТС', 'СОЛНЦЕ', 'РОССИЯ 24', 
                    'РЕН ТВ', 'ПЯТНИЦА', 'МАТЧ ТВ', 'МУЗ ТВ', 
                    'ЗВЕЗДА', 'ДОМАШНИЙ', '2X2', 'ТНТ', 'ЧЕ', 'Ю'
                        ]:
            GRP_NRA = GRP - GRP_SP - old_data['GRP ТП канала']

        elif channel in ['ПЕРВЫЙ КАНАЛ', 'РОССИЯ 1']:
            GRP_KR = GRP - old_data['GRP КРМ'] - GRP_SP
            GRP_NRA = GRP_KR - old_data['GRP ТП канала']

        elif channel == 'НТВ' or channel == 'ПЯТЫЙ КАНАЛ':
            GRP_full_sp = old_data['GRP Телемагазины'] + GRP_SP
            GRP_KR = GRP - GRP_full_sp
            GRP_NRA = GRP_KR - old_data['GRP ТП канала']

        elif channel == 'СПАС':
            GRP_NRA = GRP - old_data['GRP Телемагазины'] - GRP_SP

        elif channel in ['КАРУСЕЛЬ', 'МИР']:
            GRP_NRA = GRP - GRP_SP
                
        delta_GRP = GRP_NRA -  old_data['GRP ТП НРА']
        return round(delta_GRP)


    def get_reasons_according_to_table_cubik(self, reasons_channels_in_grp, month: str, forecast_new, grp_limit = 50, write_warning = True):
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
                write_warning: Если True, то выводятся комментарии по каналам, для которых не нашлось достаточно причин. Если False, то сообщения не выводятся.
                По дефолту True.
                forecast_old: Дата прошлого обновления
                forecast_new: Дата нового обновления
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
            date = delta_df_.iloc[i]['Дата']
            #Выделяем канал, изменения по которому хотим объяснить
            Channel = delta_df_.iloc[i]['Канал']
            #Выделяем изменение
            delta_grp = delta_df_.iloc[i]['Изменение GRP']
            for channel, contributions in reasons_channels_in_grp.items():
                result_contributors = {}
                if Channel == channel and date == forecast_new:
                    for statistic, value in contributions.items():
                        #Рассматривается отдельно ситуация с СП. Если СП < 0 => КР растет; если СП > 0 => КР падает.
                        if statistic == 'GRP СП' or statistic == 'GRP ТП канала' or statistic == 'GRP Телемагазины':
                            if value * delta_grp < 0:
                                result_contributors[statistic] = value
                        else:
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
        channels_not_exist_dict = {}
        channels_not_enough_reasons = []
        channels_not_enough_reasons_dict = {}
        for i in range(len(delta_df_)):
            #Выделяем канал, изменения по которому хотим объяснить
            channel = delta_df_.iloc[i]['Канал']
            delta_grp = delta_df_.iloc[i]['Изменение GRP']
            
            if channel not in list(sorted_reasons_channels_in_grp.keys()):
                channels_not_exist.append(channel)
            else:
                for statistic, grp in sorted_reasons_channels_in_grp[channel].items():
                    #Получение списка из GRP
                    list_of_grp = list(sorted_reasons_channels_in_grp[channel].values())
                    #Замена отрицательных GRP на положительные
                    list_of_grp_updated = [x * (-1) if x < 0 else x for x in list_of_grp]
                    if np.abs(delta_grp) - np.abs(np.sum(list_of_grp_updated)) >= grp_limit:
                        channels_not_enough_reasons.append(channel)
                    elif np.abs(delta_grp) - np.abs(np.sum(list_of_grp_updated)) <= grp_limit:
                        continue
        
        if write_warning == True:
            if set(channels_not_exist):
                morph = pmrph.MorphAnalyzer()
                month_ = morph.parse(month)[0].inflect({'loct'}).word.capitalize()
                print(color.BOLD + color.RED + f'Требуется написать комментарий САМОСТОЯТЕЛЬНО для: {", ".join(set(channels_not_exist))} в {month_}.' + color.END, end = '\n\n')
            if set(channels_not_enough_reasons):
                morph = pmrph.MorphAnalyzer()
                month_ = morph.parse(month)[0].inflect({'loct'}).word.capitalize()
                print(color.BOLD + color.BLUE + f'Найдено НЕДОСТАТОЧНО причин в {month_} для: {", ".join(set(channels_not_enough_reasons))}.' + color.END, end = '\n\n')
            return sorted_reasons_channels_in_grp
        else:
            channels_not_exist_dict[month] = list(set(channels_not_exist))
            channels_not_enough_reasons_dict[month] = list(set(channels_not_enough_reasons))
            return sorted_reasons_channels_in_grp, channels_not_exist_dict, channels_not_enough_reasons_dict


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
        for month, reasons in dict_of_result_reasons.items():
            comment_per_channel = {}
            for channel, reasons_dict in reasons.items():
                old_data, new_data = self.search_channel_and_info(month, channel)
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

                    elif statistic == 'КУС Внедом':
                        if reasons_dict['КУС Внедом'] > 0:
                            comments[statistic] = 'Рост КУС за счет роста прогноза внедомашнего телесмотрения.'
                        else:
                            comments[statistic] = 'Снижение КУС за счет снижения прогноза внедомашнего телесмотрения.'
                    
                    elif statistic == 'GRP Телемагазины' and channel == 'СПАС':
                        val = reasons_dict['GRP Телемагазины']
                        if val > 0:
                            comments[statistic] = f'Размещение телемагазинов {val} GRP.'
                        else:
                            comments[statistic] = f'Снятие телемагазинов {(-1) * val} GRP.'
                    
                    elif statistic == 'GRP ТП канала':
                        if reasons_dict['GRP ТП канала'] > 0:
                            comments[statistic] = f'Рост ТП канала.'
                        else:
                            comments[statistic] = f'Снижение ТП канала.'
                    
                    elif statistic == 'GRP КСР':
                        if reasons_dict['GRP КСР'] > 0:
                            comments[statistic] = f'Рост КСР.'
                        else:
                            comments[statistic] = f'Снижение КСР.'

                    elif statistic == 'GRP СП':
                        if reasons_dict['GRP СП'] > 0:
                            comments[statistic] = f'Рост прогноза СП.'
                        else:
                            comments[statistic] = f'Снижение прогноза СП.'
        
                    elif statistic == 'Т Общие' and channel == 'КАРУСЕЛЬ':
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
            #df_res = pd.merge(df_res, limits, on = ['Канал'], how = 'inner')
            df_res = df_res[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Доп столбец', 'Комментарий']]
            res.append(df_res)
        if len(res) != 0:
            data_output = pd.concat(res)
            data_output_ = pd.merge(self.delta_df, data_output, on = ['Канал', 'Месяц', 'Дата', 'Изменение GRP'], how = 'left')
            result_df = pd.merge(data_output_, limits, on = ['Канал'], how = 'inner')
            return result_df[['Канал', 'Месяц', 'Дата', 'Изменение GRP', 'Порог', 'Доп столбец', 'Комментарий']]
        else:
            return []


    def get_result(self, df_limits, date_of_forecast, kus_file: str, flag = False):
        """
            Функция для получения full-result
            Args:
                df: DataFrame со сравнением прогнозов
                delta_df: DataFrame с изменениями по дням согласно Федеральному кубику
                flag: По дефолту False, отвечает за возвращение сетов из каналов, для которых не нашлось достаточно причин
            Return:
                data_output: DataFrame с шаблонными комментариями
        """
        #Чтение файла с коэффициентами внедома
        outhouse_koeffs = Federal_Comments.influence_out_house(kus_file)

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
            date = self.delta_df.iloc[i]['Дата']
            reasons = self.get_reasons(month, channel, outhouse_koeffs)
            
            contributions = {}
            for changed_statistic in reasons:
                delta_GRP = self.calculate_delta_GRP(month, channel, changed_statistic)
                contributions[changed_statistic] = delta_GRP
                    
            if not month in reasons_channels.keys():
                reasons_channels[month] = {
                    channel: contributions
                }
            else:
                reasons_channels[month][channel] = contributions
            
            #Удаляем пустые элементы из словаря или каналы, по которым нет изменений
            #reasons_channels = {k:v for k,v in reasons_channels.items() if v}
        #res = {}
        #Случай, когда каналы, для которых найдено недостаточно причин, на экран не выводятся. Выводятся в виде отдельной переменной
        if flag == True:
            res = {}
            for month, reasons in reasons_channels.items():
                sorted_reasons_channels_in_grp, channels_not_exist, channels_not_enough_reasons = self.get_reasons_according_to_table_cubik(reasons, 
                                                                                            month, 
                                                                                            date_of_forecast,
                                                                                            write_warning = False)
                res[month] = sorted_reasons_channels_in_grp
            
            general_comments = self.write_comments(res)
            data_output = self.generate_result_table(general_comments, df_limits)
            if len(data_output) != 0:
                return data_output.reset_index(drop = True), channels_not_exist, channels_not_enough_reasons
            else:
                return pd.DataFrame(), channels_not_exist, channels_not_enough_reasons
        
        #Случай, когда каналы, для которых найдено недостаточно причин выводятся на экран, но не выводятся в виде отдельной переменной.
        else:
            res = {}
            for month, reasons in reasons_channels.items():
                sorted_reasons_channels_in_grp = self.get_reasons_according_to_table_cubik(reasons, 
                                                                                            month
                                                                                            )
                res[month] = sorted_reasons_channels_in_grp
            
            general_comments = self.write_comments(res)
            data_output = self.generate_result_table(general_comments, df_limits)
            if len(data_output) != 0:
                return data_output.reset_index(drop = True)
            else:
                return pd.DataFrame()
    

    @staticmethod
    def combine_columns(row):
        if pd.isna(row['Комментарий_y']):
            return row['Комментарий_x']
        elif isinstance(row['Комментарий_x'], str) and isinstance(row['Комментарий_y'], str):
            return f"{row['Комментарий_y']} {row['Комментарий_x']}" 
        else:
            return row['Комментарий_x']