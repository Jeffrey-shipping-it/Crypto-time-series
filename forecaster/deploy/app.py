import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import argparse
import logging
import json

app = Flask(__name__)


@app.route('/')
def home():
    return True


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Initiating prediction API")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, _ = parser.parse_known_args()
    config = json.load(open(args.config))

    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    ticker = payload['ticker']
    data = payload['data']
    projection = forecast(ticker, data)
    return projection.to_json()


def forecast(ticker, data):
    models = pickle.load(open('models.pickle', 'rb'))
    model = models[ticker]['model']
    future = model.make_future_dataframe(data, periods=8, n_historic_predictions=False)
    forecast = model.predict(future)
    lookup_df = forecast.drop(['ds'], axis=1)
    projection = forecast.assign(y_pred=lookup_df.lookup(lookup_df.index, lookup_df.isnull().idxmin(1)))[
        ['ds', 'y_pred']]
    return projection
