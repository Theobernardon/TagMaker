#### importations ####

import mlflow
import dill
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from classes import TextConcatWithWeightTransformer, NLPTextTransformer, tokenize_tag
import nltk
nltk.download('punkt')

#### importations des étapes de préprosessing ####

with open('prepro_pre_embed.plk', 'rb') as file:
    prepro_pre_embed = dill.load(file)
with open('prepro_post_embed.plk', 'rb') as file:
    prepro_post_embed = dill.load(file)
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
with open('y_prepro.plk', 'rb') as file:
    y_prepro = dill.load(file)

#### organisation des étapes de préprosessing ####

def prepro(x):
    x = prepro_pre_embed.transform(x)
    x = use(x)
    x = prepro_post_embed.transform(x)
    return x.astype(np.float32)

#### importations du model ####

with open('Modelfinal.plk', 'rb') as file:
    Modelfinal = dill.load(file)

#### fonction d'utilisation du model ####

app = Flask(__name__)

@app.route('/TagMaker', methods=['POST'])
def tag_maker():
    data = request.get_json()
    x = pd.read_json(data)
    tags_list_arr = y_prepro.inverse_transform(Modelfinal.predict(prepro(x)))
    tags = [result.tolist() for result in tags_list_arr]
    return jsonify({'Tags': tags})

if __name__ == '__main__':
    app.run(debug=True)