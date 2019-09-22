from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from keras.models import load_model
from glob import glob
import numpy as np
import pickle


def load_params(path_parms, type_param="level"):
    if type_param == 'level':
	    # Load model
	    modelPred = load_model(path_parms + "/model.h5")
	    # Load word2idx
	    with open(path_parms + "/word2index.pickle", 'rb') as handle:
	        w2i = pickle.load(handle)
	    # Load label encoding
	    le = LabelEncoder()
	    le.classes_ = np.load(path_parms + "/classes.npy", allow_pickle=True)
	    return modelPred, w2i, le
   
    getListNeight = glob(path_parms + "/*")
    dictWithNeighs = dict() 
    for cl in getListNeight:
    	# Load cls
    	filename_neght_cls = cl + "/neigh_cls.joblib.pkl"
    	currCls = joblib.load(filename_neght_cls)
    	# Load dict for cls
    	filename_dict = cl + "/dict_val.pickle"
    	with open(filename_dict, 'rb') as handle:
    		currDict = pickle.load(handle)
		# Create dict with cls
    	dictWithNeighs[cl.split('/')[-1]] = {
			"cls": currCls,
			"dict": currDict	
		}
    return 	dictWithNeighs
