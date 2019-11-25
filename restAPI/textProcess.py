import string
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
nltk.download("punkt")



def getLabelAndDistance(data, dictValue, model_output, neigh, number_neigh=10):
    # Get predict
    result_predict = model_output.predict(data)
    result_predict = result_predict.reshape(result_predict.shape[0],  -1)
    print("result_predict =", len(result_predict[0]))

    # Computing distance 
    distance, index = neigh.kneighbors([result_predict[0]], number_neigh, return_distance=True)
    dictResult = dict()
    for i in range(len(index[0])):
        key = dictValue[index[0][i]]
        if key not in dictResult.keys():
            dictResult[key] = [] 
        dictResult[key].append(1 - distance[0][i])
    for key, item in dictResult.items():
        dictResult[key] = str(round(sum(item)/len(item), 2))
    # maximKey = max(dictResult, key=dictResult.get)
    # maximKey, dictResult[maximKey], 
    return dictResult



def process_text(text):
    # Function processing text for Word2Vec
    tokens = word_tokenize(text.lower())
    # delete punctuation
    table = str.maketrans('', '', string.punctuation)
    words_without_punct = [w.translate(table) for w in tokens]
    # delete all except text
    words = [word for word in words_without_punct if word.isalpha()]
    # delete stopword
    stop_words = set(stopwords.words("russian"))
    words_without_stop_words = [w for w in words if not w in stop_words]
    return ' '.join(words_without_stop_words)


def convert(texts, word2idx, max_text_len=64):
    data = np.zeros((len(texts), max_text_len), dtype=np.int)
    for i in range(len(texts)):
        string = texts[i].split()
        for j in range(len(string)):
            if (string[j] in word2idx) and (j < max_text_len):
                data[i, j] = word2idx[string[j]]
    return data


def preprocessing_text(text, word2idx):
    new_text = process_text(text)
    text_to_num = convert([new_text], word2idx)
    return text_to_num

