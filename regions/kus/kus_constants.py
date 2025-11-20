import pandas as pd
import numpy as np

class KUS_Constants:

    def __init__(self, dict_data = None):
        """
        Конструктор класса Constants.
        
        Args:
            regions_dict: словарь с регионами для различных БЦА
        """
        self.dict_data = dict_data

    
    @property
    def regions_dict_list(self):
        """
            Список городов и их ID
        """
        if self.dict_data is None:
            raise ValueError("dict_data не инициализирован. Передайте словарь при создании объекта Constants.")
        else:
            return [
                self.dict_data['все 18+'], self.dict_data['все 14-59'], self.dict_data['все 10-45'], 
                self.dict_data['все 14-44'], self.dict_data['все 14-54'], self.dict_data['все 25-49'], self.dict_data['все 25-54'], 
                self.dict_data['все 4-45'], self.dict_data['все 6-54'], self.dict_data['ж 14-44'], self.dict_data['ж 25-59']
                ]
    

    @property
    def bca_list(self):
        return [
            'все 18+', 'все 14-59', 'все 10-45', 'все 14-44', 'все 14-54', 'все 25-49',
            'все 25-54', 'все 4-45', 'все 6-54', 'ж 14-44', 'ж 25-59'
            ]