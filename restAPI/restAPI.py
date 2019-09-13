import flask, json
import numpy as np
import pandas as pd
import tensorflow as tf

from flask_cors import CORS, cross_origin
from  glob import glob
from textProcess import *
from load_param import *


# Загрузка моделей
global graph
graph = tf.get_default_graph()
models = dict()
directModel = glob("./models/level*")
for direct in directModel:
    level = direct.split('/')[-1]
    print("Load params ...")
    modelPred, w2i, le = load_params(direct)
    models[level] = {
        "model": modelPred,
        "w2i": w2i,
        "le": le
    }
    print("Params success from " + level)


def formationRSP(req):
    # Формирование ответа
    listLevel = ["level_1", "level_2", "level_3", "level_4"]
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
        result["predict_" + level] = json.dumps(pred_class)
        if level == "level_1" and np.argmax(pred_proba) == 0:
            break
        if max(pred_proba) > 1.3*(sum(pred_proba) - max(pred_proba)):
            break
    output["predictResult"] = json.dumps(result)
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

