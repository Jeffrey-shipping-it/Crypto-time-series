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

    model = pickle.load(open(config['models_dir'], 'rb'))

    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
