import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_params(path_parms):
    # Load model
    modelPred = load_model(path_parms + "/model.h5")
    # Load word2idx
    with open(path_parms + "/word2index.pickle", 'rb') as handle:
        w2i = pickle.load(handle)
    # Load label encoding
    le = LabelEncoder()
    le.classes_ = np.load(path_parms + "/classes.npy", allow_pickle=True)
    return modelPred, w2i, le