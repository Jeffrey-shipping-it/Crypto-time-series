import pandas as pd
import pandas_ta as ta
import os


class DataProcessor:
    """
    Class containing related data handling methods to make pipelines.
    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _fill_data_dict(fill_dir=os.listdir(), extension=".parquet", func=pd.read_parquet, splitter="-"):
        """
        Fills up data dictionary from given directory, given a file format and corresponding function
        :param fill_dir: Directory to read from.
        :param extension: file format.
        :param func: function needed to read files.
        :param splitter: splitter for name.
        :return: dictionary with ticker as key and corresponding df as value.
        """
        file_list = [file for file in fill_dir if file.endswith(extension)]
        data_dict = {}
        for file in file_list:
            data_dict[file.split(splitter)[0]] = func(file)
        return data_dict

    def _downsample_binance_data_dict(self, data_dict):
        """
        Samples all data in dictionary to frequency given in config file.
        :param data_dict: data dict.
        :return: Resampled data dictionary.
        """
        for ticker, ticker_data in data_dict.items():
            data_dict[ticker] = ticker_data.resample(self.config["freq"]).agg(
                self.config["features_config"]).reset_index()
        return data_dict

    @staticmethod
    def _extract_ta_ind(data_dict):
        """
        Computes technical indicators for all data. For more info on possible indicators,
        visit pandas-ta documentation.
        :param data_dict:data dict.
        :return:
        """
        for ticker, ticker_data in data_dict.items():
            df = data_dict[ticker]
            df.ta.rsi(close=df['close'], length=72, append=True)
            df.ta.fwma(close=df['close'], length=72, append=True)
            df.dropna(inplace=True)
            df.columns = df.columns.str.replace('.', '')
            data_dict[ticker] = df
        return data_dict

    @staticmethod
    def _filter_by_date(data_dict, date_start=None, date_range=None):
        """
        filters training data by date before fitting. Date start and date range can be specified both,
        in which case they will be applied both.
        :param data_dict: data dict.
        :param date_start: date from which training data is used.
        :param date_range: pandas date range from which training data is used.
        :return: filtered training data.
        """
        for ticker, ticker_data in data_dict.items():
            if date_start:
                data_dict[ticker] = ticker_data.query("open_time > @date_start")
            if date_range:
                data_dict[ticker] = ticker_data.query("open_time in @date_range")
            if date_start == date_range is None:
                raise ValueError("Either date_start or date_range must be specified!")
        return data_dict
