import flask, json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Concatenate, Average, Input
from keras.models import Model
from flask_cors import CORS, cross_origin
from  glob import glob
from textProcess import *
from load_param import *


path_models = "./models/*"

# Загрузка моделей
global graph
graph = tf.get_default_graph()
models = dict()
directModel = glob(path_models)
for direct in directModel:
    print("Load params ...")
    if 'level' in direct:
    	level = direct.split('/')[-1]
    	modelPred, w2i, le = load_params(direct, type_param="level")
    	models[level] = {
	        "model": modelPred,
	        "w2i": w2i,
	        "le": le
	    }
    	print("Params success from " + level)
    else:
    	dictWithNeighs = load_params(direct, type_param="neigh")
    	print("Params success loads from " + direct.split("/")[-1])

# Загрузка объединненой модели
model_curr_1 = Input(shape=(64,))
x1 = models["level_1"]["model"].layers[0](model_curr_1)
model_curr_2 = Input(shape=(64,))
x2 = models["level_2"]["model"].layers[0](model_curr_2)
average = Average()([x1, x2])
model_output = Model(inputs=[model_curr_1, model_curr_2], outputs=average)
print("All params success loaded")


def formationRSP(req, thresholder_conf=1.3):
    # Формирование ответа
    listLevel = ["level_1", "level_2"]
    output = dict()
    result = dict()
    for level in listLevel:
        print(level)
        dParams = models[level] 
        text_new = preprocessing_text(req["text"], dParams["w2i"])
        with graph.as_default():
            pred_proba = np.round(dParams["model"].predict_proba(text_new), 2)
            pred_class = dict(zip(
                dParams["le"].inverse_transform(range(len(pred_proba[0]))), 
                pred_proba[0].astype(str)))
        # clsPred = np.argmax(pred)
        result[level] = pred_class
        if level == "level_1" and np.argmax(pred_proba) == 0:
            break
        # if max(pred_proba[0]) > thresholder_conf*(sum(pred_proba[0]) - max(pred_proba[0])):
        #     break
    if "level_2" in result.keys():
    	vector_level_1 = preprocessing_text(req["text"], models["level_1"]["w2i"])
    	vector_level_2 = preprocessing_text(req["text"], models["level_2"]["w2i"])
    	currTensor = [vector_level_1, vector_level_2]
    	print("dictWithNeighs.keys()", dictWithNeighs.keys())
    	maximKey = max(result["level_2"], key=result["level_2"].get)
    	print("maximKey =", maximKey)
    	getListParams = dictWithNeighs["cls_" + str(maximKey)]
    	with graph.as_default():
	    	dictResult = getLabelAndDistance(currTensor, getListParams['dict'], 
	    		model_output, getListParams['cls'], number_neigh=10)
	    	print(dictResult)
	    	newDict = dict()
	    	for k, item in dictResult.items():
		    	if len(k.split(".")) == 3 and len(k.split(".")[-1]) != 0:
			    	newDict[k.split(".")[-1]] = item
		    	if len(k.split(".")) > 3 and len(k.split(".")[-2]) != 0 and len(k.split(".")[-1]) != 0:
			    	newDict[k.split(".")[-2] + "." + k.split(".")[-1]] = item
	    	result["level_other"] = newDict
    	# dictLevel_3 = dict()
    	# dictLevel_4 = dict()
    	# for k, item in newDict.items():
	    # 	currListKey = k.split(".")
	    # 	currItem = item
	    # 	if len(currListKey[0]) == 0:
		   #  	continue
	    # 	if (len(currListKey) >= 1):
		   #  	if (currListKey[0] in dictLevel_3.keys()) and (dictLevel_3[currListKey[0]] > item):
			  #   	currItem = dictLevel_3[currListKey[0]]
		   #  	dictLevel_3[currListKey[0]] = currItem
	    # 	if (len(currListKey) > 1) and (len(currListKey[-1]) != 0):
		   #  	if (currListKey[-1] in dictLevel_4.keys()) and (dictLevel_4[currListKey[-1]] > item):
			  #   	item = dictLevel_4[currListKey[-1]]
		   #  	dictLevel_4[currListKey[-1]] = item
    	# if len(dictLevel_3) != 0:
	    # 	result["level_3"] = dictLevel_3
    	# if len(dictLevel_4) != 0:
	    # 	result["level_4"] = dictLevel_4
    output["predictResult"] = result
    output["code"] = 0
    return output


def checkReq(req):
    # Проверка валидности запроса
    importantFileds = ["text"]
    print(flask.request.json)
    if not flask.request.json:
        print("[Error] checkReq - no JsonData")
        flask.abort(400)
    for field in importantFileds:
        if field not in req.keys():
            print("[Error] checkReq - no valid keys")
            flask.abort(400)
    return 0


app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def create_task():
    req = flask.request.json  
    # print("[req] = ", flask.request.json) 
    result_check = checkReq(req) 
    if result_check == 0:
        response = formationRSP(req)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True, threaded=False)

