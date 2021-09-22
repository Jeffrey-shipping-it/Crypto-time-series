from data.dataprocessor import DataProcessor
from modelling.ARtrainer import NeuralProphetFitting
from utils import _save_model_dict
import argparse
import json
import logging
import pandas as pd


def training_pipeline(config_dict):
    processor = DataProcessor(config=config_dict)
    data_dict = processor._fill_data_dict()
    data_dict = processor._downsample_binance_data_dict(data_dict=data_dict)
    data_dict = processor._filter_by_date(data_dict, '2019-01-01')
    data_dict = processor._extract_ta_ind(data_dict)
    logging.info("preprocessing done")
    trainer = NeuralProphetFitting(pd.to_datetime("now"), data_dict, config_dict)
    data_dict = trainer.preprocess_prophet(data_dict)
    model_dict = trainer.fit_neuralprophet(data_dict)

    _save_model_dict(model_dict)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Initiating training pipeline")
    parser = argparse.ArgumentParser()

    # config json sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--config', type=str)
    args, _ = parser.parse_known_args()
    config = json.load(open(args.config))

    training_pipeline(config)
