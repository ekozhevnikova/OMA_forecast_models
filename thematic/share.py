import time
import pandas as pd

class Share_Thematic:
    def __init__(self, dataframe_erk_drk, dataframe_mrk, dataframe_grk):
        self.dataframe_erk_drk = dataframe_erk_drk
        self.dataframe_mrk = dataframe_mrk
        self.dataframe_grk = dataframe_grk
        self.channels = {
            'ERK_DRK': 'ЕРК_ДРК',
            'MRK': 'МРК',
            'GRK': 'ЖРК'
        }
        
    @staticmethod
    def __get_output(dataframe):
        '''
        Static Private Method to get output DataFrame in new view from API
        '''
        dataframe['tvCompanyName'] = dataframe['tvCompanyName'].apply(lambda x: x.removesuffix(' (СЕТЕВОЕ ВЕЩАНИЕ)'))
        data_output = dataframe.rename(columns = {'prj_name': 'ЦА', 'tvCompanyName': 'Канал'})
        #tmp = pd.DataFrame.copy(data_output[data_output['Канал'] == 'BRIDGE CLASSIC'])
        data_output.replace({'VIJU HISTORY': 'VIASAT HISTORY', 
                             'VIJU NATURE': 'VIASAT NATURE', 
                             'VIJU TV1000 НОВЕЛЛА': 'TV 1000 НОВЕЛЛА',
                             'VIJU EXPLORE': 'VIASAT EXPLORE', 
                             'ТВ-21М': 'ТВ21', 
                             'АВТО ПЛЮС ТВ': 'АВТОПЛЮС',
                             'БОБЁР': 'БОБЕР', 
                             'VIJU TV1000': 'TV 1000',
                             'VIJU TV1000 ACTION': 'TV 1000 ACTION', 
                             'VIJU TV1000 РУССКОЕ': 'TV 1000 РУССКОЕ КИНО', 
                             'ЛЯ МИНОР. МОЙ МУЗЫКАЛЬНЫЙ': 'ЛЯ-МИНОР ТВ',
                             'BRIDGE CLASSIC': 'БРИДЖ ТВ CLASSIC', 
                             'BRIDGE HITS': 'БРИДЖ ТВ ХИТ',
                             'BRIDGE РУССКИЙ ХИТ': 'БРИДЖ ТВ РУССКИЙ ХИТ', 
                             'О!': 'О', 
                             'ПОЕХАЛИ!': 'ПОЕХАЛИ',
                             'ПОБЕДА': 'ПОБЕДА', 
                             'BRIDGE': 'БРИДЖ ТВ', 
                             'RU.TV': 'РУ ТВ'},
                             inplace = True)
        #data_output = pd.concat([tmp, data_output], ignore_index=True)
        #data_output.replace('BRIDGE', 'БРИДЖ ТВ', inplace = True)
        data_output.insert(loc = 2, column = 'Channel', value = data_output['Канал'] + ' ' + data_output['ЦА'])
        return data_output

    def __get_share(self, dataframe, channel_name: str, is_dropna: bool = True):
        '''
        Private Method to get Share statistic by using VLOOKUP and merging DataFrame API with each Thematic Channel
        dataframe - DataFRame in new view from API, which you received in previous function
        '''
        if channel_name == self.channels['ERK_DRK']:
            data = pd.merge(self.dataframe_erk_drk, dataframe, on = 'Channel', how = 'left')
            data = data.reindex(self.dataframe_erk_drk.index)
        elif channel_name == self.channels['MRK']:
            data = pd.merge(self.dataframe_mrk ,dataframe, on = 'Channel', how = 'left')
            data = data.reindex(self.dataframe_mrk.index)
        elif channel_name == self.channels['GRK']:
            data = pd.merge(self.dataframe_grk, dataframe, on = 'Channel', how = 'left')
            data = data.reindex(self.dataframe_grk.index)
        else:  
            print('This channel name does not exist' + channel_name)
        data = data.drop(['ЦА', 'Канал'], axis = 1)
        
        if is_dropna:
            data = data.dropna()
        return data

    def __to_file(self, filepath, channel_data: str, sheet_name: str):
        '''
        Private Method which helps you to write your received data to Excel file 
        '''
        with pd.ExcelWriter(filepath, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'replace') as writer:
            channel_data.to_excel(writer, sheet_name = sheet_name)
        #writer.save()

    def get_data(self, dataframe, filepath, is_dropna = True):
        '''
        This function helps you to get final result for each thematic channel
        dataframe - is the output DataFrame in the new view from API
        '''
        data_output = Share_Thematic.__get_output(dataframe)
        #data_output.to_excel('data_output.xlsx')        
        
        res = {}
        for key, value in self.channels.items():
            res[key] = self.__get_share(data_output, value, is_dropna)
            self.__to_file(filepath, channel_data = res[key], sheet_name = value)
        return data_output, res