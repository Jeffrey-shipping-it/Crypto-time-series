import pickle5 as pickle
import os


def _save_model_dict(model_dict, save_dir=os.listdir()):
    with open('models.pickle', 'wb') as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_model_dict():
    with open('notebooks/models.pickle', 'rb') as handle:
        model_dict = pickle.load(handle)
        return model_dict
