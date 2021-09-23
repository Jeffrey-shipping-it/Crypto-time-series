# Crypto-time-series

This repository contains code for a ML microservice to be integrated into trading engines.
It contains a training pipeline for Autoregressive Neural Networks.

Training pipeline can be booted by command `python .forecaster/train.py --config config.json`, where config.json contains settings that can be tuned.

Deployment is done with Flask for now.
