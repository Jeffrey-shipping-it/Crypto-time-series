from neuralprophet import NeuralProphet
import pandas as pd
import numpy as np
import logging
import itertools


class NeuralProphetFitting:
    """
    Neural Prophet class to tune AR-Networks for the purpose of price forecasting.
    """

    def __init__(self, data_dict, date, config):
        self.run_date = date
        self.data_dict = data_dict
        self.config = config

    @staticmethod
    def add_missing_dates_nan(df, freq):
        """Fills missing datetimes in 'ds', with NaN for all other columns
        Args:
            df (pd.Dataframe): with column 'ds'  datetimes
            freq (str):Data step sizes. Frequency of data recording,
                Any valid frequency for pd.date_range, such as 'D' or 'M'
        Returns:
            dataframe without date-gaps but nan-values
        """
        if df["ds"].dtype == np.int64:
            df.loc[:, "ds"] = df.loc[:, "ds"].astype(str)
        df.loc[:, "ds"] = pd.to_datetime(df.loc[:, "ds"])

        data_len = len(df)
        r = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq=freq)
        df_all = df.set_index("ds").reindex(r).rename_axis("ds").reset_index()
        num_added = len(df_all) - data_len
        return df_all, num_added

    def preprocess_prophet(self, data_dict):
        """
        Transforms Binance data into format required by NeuralProphet.
        :param data_dict: dictionary containing all data.
        :return: Transformed data dict.
        """
        for ticker, df in data_dict.items():
            df = df.reset_index().rename(columns={'close': 'y', 'open_time': 'ds'})
            df_new, num = self.add_missing_dates_nan(df, self.config['frequency'])
            df_new.interpolate(method='linear', limit_direction='forward', inplace=True)
            df_new = df_new[['ds', 'y'] + self.config['features']]
            data_dict[ticker] = df_new
        return data_dict

    def fit_neuralprophet(self, data_dict):
        """
        Main fitting method that for every ticker in the data dict tunes a model, refits with the best
        parameters and fills a model dictionary.
        :param data_dict:  dictionary containing all data.
        :return: dictionary containing all models and corresponding validation metrics.
        """

        model_dict = {}
        for ticker, df in data_dict.items():
            model_dict[ticker] = {}

            logging.info(f"Fitting neuralprophet model for {ticker}")

            gridsearch_df, metric_cols = self.tune_neuralprophet(df, self.config["param_grid"])

            best_params = self._fetch_best_params(gridsearch_df, metric_cols)

            m = NeuralProphet(
                **best_params,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoints_range=1,
                impute_missing=True,
            )

            for feature in self.config["features"]:
                m = m.add_lagged_regressor(name=feature)

            m.highlight_nth_step_ahead_of_each_forecast(step_number=m.n_forecasts)
            df = df[self.config['features'] + ['y', 'ds']]
            metrics = m.fit(df
                            , freq=self.config["frequency"]
                            , validate_each_epoch=False
                            , plot_live_loss=False
                            )
            logging.info(metrics)
            model_dict[ticker]['model'] = m
            model_dict[ticker]['metrics'] = gridsearch_df
            print(f"Finished {ticker}")
            logging.info(f"Finished {ticker}")
        logging.info(f"Training of {len(model_dict.items())} models complete!")
        return model_dict

    def tune_neuralprophet(self, df, param_grid):
        """
        Iterates over hyper parameter combinations as given in config to find best validating model.
        :param df: data for individual ticker being tuned.
        :param param_grid: parameter grid to test.
        :return: evaluation frame containing metrics and hyperparamters
        """
        eval_frame = pd.DataFrame()
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        logging.info(f"Fitting {len(all_params)} models")
        for params in all_params:
            logging.info(f"fitting model with the following parameters: {params}")
            m = NeuralProphet(**params,
                              yearly_seasonality=False,
                              weekly_seasonality=False,
                              daily_seasonality=False,
                              changepoints_range=1,
                              impute_missing=True)  # create model object
            for feature in self.config["features"]:
                m = m.add_lagged_regressor(name=feature)
            m.highlight_nth_step_ahead_of_each_forecast(step_number=m.n_forecasts)
            df = df[self.config['features'] + ['y', 'ds']]
            metrics = m.fit(df
                            , freq=self.config["frequency"]
                            , validate_each_epoch=True
                            , valid_p=0.1
                            , plot_live_loss=True
                            )
            new_row = pd.concat(
                [pd.DataFrame(params, index=[0]), metrics.tail(1).set_index(pd.DataFrame(params, index=[0]).index)],
                axis=1)
            eval_frame = pd.concat([eval_frame, new_row])
        return eval_frame, list(metrics.columns)

    def _fetch_best_params(self, eval_frame, metrics_columns):
        """
        Takes best hyperparameters from evluation frame and returns them for final fit.
        :param eval_frame: frame with values of metrics.
        :param metrics_columns: corresponding metrics names.
        :return: dictionary with best performing hyper parameters.
        """
        eval_frame.sort_values(f"MAE-{self.config['prediction_length']}_val", ascending=False, inplace=True)
        best_row = eval_frame.head(1)
        best_params = best_row.drop(metrics_columns, axis=1).to_dict('records')[0]
        return best_params
