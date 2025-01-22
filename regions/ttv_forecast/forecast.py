import pandas as pd
from prophet import Prophet
import datetime as dt
from datetime import datetime, timedelta
from OMA_tools.regions.ttv_forecast.constants import Holidays, Prophet_Constants, Constants__Columns
#from regions_forecast.core import Main
from OMA_tools.regions.ttv_forecast.calculation import Calculation
from OMA_tools.io_data.operations import File, Table, Dict_Operations



class TTV_Regions_Forecast:
    """
    Class that makes Forecast procedure
    """
    def __init__(self, 
                 last_predict_date,
                 number_of_previous_days
                 ):
        self.last_predict_date = last_predict_date
        calc = Calculation()
        calc.set_date_filter(number_of_previous_days)
        self.calculation = calc
        self.proph_consts = Prophet_Constants()
    
    def get_predictions(self, df, key: str):
        """
        Generate the number of prediction days.
        
        Args:
            df: dict of total DataFrames.
        Returns:
            The number of predict days (int) and the last fact date (last_fact_date) in total DataFrame.
        """
        last_fact_date = df[key].date.max()
        #последняя дата предсказываемого периода
        self.last_predict_date = pd.to_datetime(self.last_predict_date, format = '%d.%m.%Y')
        last_fact_date = pd.to_datetime(last_fact_date, format = '%Y-%m-%d')
        predictions = self.last_predict_date - last_fact_date
        predictions = predictions.days
        return predictions, last_fact_date

    @staticmethod
    def convert_columns_to_prophet_format(dict_of_total):
        """
        This function is made to convert DataFrame to Prophet format with columns 'ds' and 'y'.
        
        Args:
            dict_of_total: dict with total DataFrames.
        Returns:
            dict_of_total: dict of total DataFrames in new format with new columns 'ds' and 'y'.
            ds_dict: dict of dates
        """
        ds_dict = {}
        for key, value in dict_of_total.items():
            new_cols = {'date': 'ds'}
            for col in value.columns[1:]:
                new_cols[col] = col
            dict_of_total[key] = value.rename(columns = new_cols)
            ds_dict[key] = pd.to_datetime(dict_of_total[key].ds, format = '%Y-%m-%d')
        return dict_of_total, ds_dict

    def get_cond__and__train_df(self, ds_dict, dict_of_total, last_fact_date, list_of_replacements = ['0', '1', '2', '3', '4', '5', '6']):
        """
        This function returns condition and datasets of training data which were selected by condition.
        
        Args:
            ds_dict: DataFrame with dates only
            dict_of_total: dict of total DataFrames, where column 'date' was renamed to 'ds' and column 'ГОРОД БЦА' was renamed to 'y'.
            last_fact_date: the last fact date in the datasets.
            
        Returns:
            cond_df: dict of condition on choosing the right history data
            train_df: dict of train_df DataFrames depending on condition (cond_df).
        """
        cond_df = {'All 4-45': [], 
                'All 6-54': [], 
                'All 14-54': [], 
                'All 18+': [], 
                'EKB_NN_KZN': [], 
                'Novosibirsk': [],
                'SaintPetersburg': []
                }

        train_df = {'All 4-45': [], 
                    'All 6-54': [], 
                    'All 14-54': [], 
                    'All 18+': [], 
                    'EKB_NN_KZN': [], 
                    'Novosibirsk': [],
                    'SaintPetersburg': []
                    }
        ds_dict_converted = Dict_Operations(ds_dict).replace_keys_in_dict(list(cond_df.keys()))
        dict_of_total_converted = Dict_Operations(dict_of_total).replace_keys_in_dict(list(train_df.keys()))
        #dict_cond_dates = proph_consts.cond_date_count
        
        for bca_num, dates in enumerate(self.proph_consts.cond_date_count.values()):
                for date_num, date in enumerate(dates):
                    idx = list(self.proph_consts.cond_date_count.keys())[bca_num]
                    cond_df[idx] = (ds_dict_converted[idx] <= pd.to_datetime(last_fact_date)) & (ds_dict_converted[idx] >= pd.to_datetime(date))
                    train_df[idx] = dict_of_total_converted[idx][cond_df[idx]]
        return cond_df, train_df


    def get_forecast(self, dict_of_total, train_df, predictions, last_fact_date):
        """
        This function returns forecast for each BCA.
        
        Args:
            dict_of_total - DataFrame of new total with converted columns to Prophet format.
            train_df - training data, which was selected by condition
            predictions - number of predict days, int
            last_fact_date - the last fact date in our dataset
        Returns:
            results_df: dict of forecast results.
        """
        results_df = {}
        for bca, df in train_df.items():
            results = pd.DataFrame()
            for icol, col in enumerate(df.columns[1:]):
                tmp_df = pd.concat([df['ds'], df.iloc[:, icol + 1]], axis = 1, keys = ['ds', 'y'])

                m = Prophet()
                if self.proph_consts.cond_holidays[bca]:
                    m = Prophet(holidays = Holidays().holidays)
                m.fit(tmp_df)

                future = m.make_future_dataframe(periods = predictions)
                forecast = m.predict(future)

                #во фрейме дата добавляем нижнюю и верхнюю границы 
                tmp_df.columns = ['ds', 'yhat']
                tmp_df['yhat_lower'] = tmp_df['yhat']
                tmp_df['yhat_upper'] = tmp_df['yhat']

                tmp = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                forecast_cut = tmp[tmp.ds > last_fact_date]
                tmp_df = tmp_df[tmp_df.ds >= dt.datetime(last_fact_date.year - 2, 1, 1)]
                result = pd.concat([tmp_df, forecast_cut], axis = 0)
                result['bca'] = df.columns[icol + 1]

                results = pd.concat([result, results])
            results_df[bca] = results
        return results_df

    def get_result(self, filename_fact_data, filename_result_data, path, file_names):
        """
        The sum function that concludes API calculation and forecast process using Prophet.
        
        Args:
            filename_data_fact: the full path to file with Fact Data.
            filename_result_data: the full path to file with Forecast Results.
        Returns:
            The result of forecast procedure goes to the file with Forecast Results.
        """
        data_api = self.calculation.get__data_through_API()
        total = File(filename_fact_data).update_file(data_api, 'date', ['All 4-45', 
                                                                        'All 6-54', 
                                                                        'All 14-54', 
                                                                        'All 18+', 
                                                                        'EKB_NN_KZN', 
                                                                        'Novosibirsk', 
                                                                        'SaintPetersburg']
                                                    )
        
        predictions, last_fact_date = self.get_predictions(total, 'All 4-45')
        df, ds_dict = TTV_Regions_Forecast.convert_columns_to_prophet_format(dict_of_total = total)
        cond_df, train_df = self.get_cond__and__train_df(ds_dict = ds_dict,
                                                    dict_of_total = df,
                                                    last_fact_date = last_fact_date)
        results_df = self.get_forecast(dict_of_total = df,
                                  train_df = train_df,
                                  predictions = predictions,
                                  last_fact_date = last_fact_date)  
        File(filename_result_data).to_file(results_df)
        File(filename_result_data).to_file_from_dict(path = path,
                                                               dict_data = results_df,
                                                               file_names = file_names)
        #return results_df