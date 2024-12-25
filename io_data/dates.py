import pandas as pd
import pymorphy3 as pmrph
import datetime as dt
from datetime import datetime, timedelta
from pathlib import Path


class Dates_Operations:
    def __init__(self, date_filter = None, number_of_previous_days = []):
        if date_filter:
            self.date_filter = date_filter
        else:
            assert len(number_of_previous_days) > 0
            self.date_filter = Dates_Operations.set__date_filter(number_of_previous_days)
    
    @staticmethod
    def set__date_filter(number_of_previous_days = []):
        """
            Generate date_filter from today day
            Args:
                number_of_previous_days: list of the number of days that you want to minus from the today's date to generate start date and stop date
        """
        date_start = datetime.now() + timedelta(days = number_of_previous_days[0])
        start_date = date_start.strftime('%Y-%m-%d')

        date_stop = datetime.now() + timedelta(days = number_of_previous_days[1])
        stop_date = date_stop.strftime('%Y-%m-%d')
        
        print('start date: ' + start_date + '; ' + 'stop_date: ' + stop_date)
        return [(start_date, stop_date)]
    
    
    def get_date(self):
        """
            Get day start and stop from date_filter like: [('2024-10-01', '2024-10-31')]
            Output will be: 1, 31
        """
        start_date = self.date_filter[0][0]
        stop_date = self.date_filter[0][1]
        start = start_date[8:10]
        stop = stop_date[8:10]
        if start in ['01', '02', '03', '04', '05', '06', '07', '08', '09'] and stop in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
            start_res = start[1:]
            stop_res = stop[1:]
            return int(start_res), int(stop_res)
        elif start in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
            start_res = start[1:]
            return int(start_res), int(stop)
        elif stop in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
            stop_res = stop[1:]
            return int(start), int(stop_res)
        else:
            return int(start), int(stop)
    
    
    def get_month_from_date_filter(self):
        """
            Generates month num of start and stop date from date_filter like: [('2024-10-01', '2024-10-31')]
            Output will be: October
        """
        months = ['Январь', 'Февраль', 'Март', 
              'Апрель', 'Май', 'Июнь', 
              'Июль', 'Август', 'Сентябрь', 
              'Октябрь', 'Ноябрь', 'Декабрь']
        start_date = self.date_filter[0][0]
        stop_date = self.date_filter[0][1]
        morph = pmrph.MorphAnalyzer()
        month_num_1 = morph.parse(months[int(start_date[5:7]) - 1])[0].inflect({'gent'}).word.capitalize()
        month_num_2 = morph.parse(months[int(stop_date[5:7]) - 1])[0].inflect({'gent'}).word.capitalize()
        return month_num_1, month_num_2
    
    
    def generate_date_interval(self, statistic: str):
        """
            Generates date interval like 1 October - 11 November using date_filter like: [('2024-10-01', '2024-11-11')]
        """
        start_date, stop_date = self.get_date()
        month_num_1, month_num_2 = self.get_month_from_date_filter()

        sr = pd.Series([self.date_filter[0][0], self.date_filter[0][1]])
        idx = ['date_1', 'date_2']
        sr.index = idx 
        sr = pd.to_datetime(sr) 
        result = sr.dt.is_month_end 

        if start_date == 1 and stop_date in [28, 29, 30, 31] and month_num_1 == month_num_2 and result[1] == True:
            return str(statistic) + ' ' + str(month_num_2)
        elif month_num_1 == month_num_2 and result[1] == False:
            return str(statistic) + ' ' + str(start_date) + ' - ' + str(stop_date) + ' ' + str(month_num_1)
        else:
            return str(statistic) + ' ' + str(start_date) + ' ' + str(month_num_1) + ' - ' + str(stop_date) + ' ' + str(month_num_2)
    
    
    @staticmethod
    def get_date_and_month_from_filename(filename, substring_to_remove: str):
        """
            Generates day_num and month_num from filename
            Args:
                filename
                substring_to_remove: substring in the filename which your want to replace before date
                Example: file_01.01.2000; substring_to_remove here is 'file_'
        """
        path = Path(filename)
        name = path.name
        name = name.replace(substring_to_remove, '')
        date = name[0:2]
        month = name[3:5]
        if date in ['01', '02', '03', '04', '05', '06', '07', '08', '09'] and month in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
            date_res = date[1:]
            month_res = month[1:]
            return int(date_res), int(month_res)

        elif date in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
            date_res = date[1:]
            return int(date_res), int(month)

        elif month in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
            month_res = month[1:]
            return int(date), int(month_res)
        else:
            return int(date), int(month)
        
    @staticmethod
    def get_month(month_num: int, n: int, case: str):
        """
        Function to get month.
        Args:
            month_num: the number of month
            n: quantity of month that you want to minus. ATTENTION!!! To get current month you should specify n = 1.
            case: 'Именительный', 'Родительный', 'Дательный', 'Винительный', 'Творительный', 'Предложный'
        Return: desired month 
        """
        month_changes = ['Январь', 'Февраль', 'Март', 
                         'Апрель', 'Май', 'Июнь', 
                         'Июль', 'Август', 'Сентябрь', 
                         'Октябрь', 'Ноябрь', 'Декабрь']

        month = month_changes[month_num - n]
        morph = pmrph.MorphAnalyzer()
        month = morph.parse(month)[0]

        if case == 'Именительный' or case == 'Винительный':
            return [month.inflect({'nomn'}).word, month.inflect({'nomn'}).word.capitalize()]
        elif case == 'Родительный':
            return [month.inflect({'gent'}).word, month.inflect({'gent'}).word.capitalize()]
        elif case == 'Дательный':
            return [month.inflect({'datv'}).word, month.inflect({'datv'}).word.capitalize()]
        elif case == 'Творительный':
            return [month.inflect({'ablt'}).word, month.inflect({'ablt'}).word.capitalize()]
        elif case == 'Предложный':
            return [month.inflect({'loct'}).word, month.inflect({'loct'}).word.capitalize()]
        
    @staticmethod    
    def convert_dates_from_str_to_datetime_format(df, date_column_name: str, date_format_init: str):
        """
        Converts date from format 'Январь 2021' to '2021.01.01'.
        Args:
            df: DataFrame with Date column
            date_column_name: name of the column with Date
            date_format_init: init format of date. For example 'Январь 2021' is '%B %Y'
        Returns:
            df with converted to the new format date column
        """
        dates = list(df[date_column_name])
        dates_converted = []
        for i in dates:
            dates_converted.append(datetime.strptime(i, date_format_init).strftime('%Y.%m.%d'))
        df[date_column_name] = df[date_column_name].replace(dates, dates_converted)
        df[date_column_name] = df[date_column_name].apply(lambda x: pd.to_datetime(x))
        return df
    
    
    @staticmethod
    def make_difference_in_months(start_date, stop_date):
        """
            Args:
                start date in datetime format (for example: %d.%m.%Y)
                end date in datetime format (for example: %d.%m.%Y)
            Returns:
                returns the number of months between two dates
        """
        start_date = pd.to_datetime(start_date, format = '%d.%m.%Y')
        stop_date = pd.to_datetime(stop_date, format = '%d.%m.%Y')

        if start_date.year == stop_date.year:
            months = stop_date.month - start_date.month
        else:
            months = (12 - start_date.month) + (stop_date.month)
        if start_date.day > stop_date.day:
            months = months - 1
        return months