#### importations ####

import mlflow
import re
import dill
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import cloudpickle
nltk.download('punkt')
nltk.download('wordnet')

class TextConcatWithWeightTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns='all', poid=[1]):
        self.columns = columns
        self.poid = poid

    def fit(self, X, y=None):
        return self
    
    def _concat_serie_meker(self, df, columns, poids):
        serie_out = (df[columns[0]] + " ") * poids[0]
        for col, poid in zip(columns[1:], poids[1:]):
            serie_out += (df[col] + " ") * poid
        return serie_out
    
    def transform(self, X):
        # 
        if (self.columns == 'all') & (self.poid == [1]):
            self.columns = X.columns
            self.poid = self.poid * X.shape[1]
        elif len(self.columns) == len(self.poid):
            pass
        else:
            raise print("ERROR: Si toutes les colonnes n'ont pas le même poids veuillez renseigner les colonnes et les poids dans 2 listes")
        
        return self._concat_serie_meker(X, self.columns, self.poid)

class NLPTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=True, stopwords=None, lemmatizer=None, vocabulary=None, tokenizer=nltk.word_tokenize, verbose=False):
        """NLPTextTransformer est un objet permettant transformer un texte en permettant l'intégration à une pipeline sklearn.
        Cet objet peut appliquer lower sur le texte, retirer des stop words et lemmatizer. 
        ATTENTION : Malgré la présence d'un tokenizer, le return de la fonction transforme sera bien sous forme de array de texte.
        Le tokenizer ne servira que d'étapes à l'application des autres process.
        ATTENTION : La réduction du champ lexical d'après le vocabulaire se fait après la lemmatisation. Vous devez donc, sur le 
        vocabulaire, avoir également effectué une étape de lemmatisation pour que les mots correspondent bien au texte.
        
        Args:
            lower (bool, optional): Si oui ou non le texte doit être en minuscule. Defaults to True.
            stopwords (list, tuple, set, optional): liste des stop words à retirer. Defaults to None.
            lemmatizer (méthode, optional): Objet de lemmatisation intervenant à travers une méthode .lemmatizer(text). Defaults to None.
            vocabulary (set, optional): Réduction des mots du texte a un vocabulaire défini. Defaults to None.
            tokenizer (_type_, optional): Objet de tokenisation intervenant à travers une méthode .tokenizer(text). Defaults to nltk.word_tokenize.
        """
        self.lower = lower
        self.stopwords = stopwords
        self.lemmatizer = lemmatizer
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.verbose = verbose

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.lower:
            X = X.str.lower()
        
        if (self.stopwords is not None) or (self.lemmatizer is not None) or (self.vocabulary is not None):
            if self.verbose:
                print('Tokenization:')
                X = X.progress_apply(lambda x: self.tokenizer(x))
            else:
                X = X.apply(lambda x: self.tokenizer(x))
            if isinstance(self.stopwords, (list, tuple, set)):
                if self.verbose:
                    print('Suppression des stopwords:')
                    X = X.progress_apply(lambda x: [word for word in x if not word in self.stopwords])
                else:
                    X = X.apply(lambda x: [word for word in x if not word in self.stopwords])
            else:
                raise print('TypeError: stopwords doit être une liste, un tuple ou un set')
            
            if self.lemmatizer is not None:
                try :
                    if self.verbose:
                        print('Lemmatization:')
                        X = X.progress_apply(lambda x: [self.lemmatizer(word) for word in x])
                    else:
                        X = X.apply(lambda x: [self.lemmatizer(word) for word in x])
                except Exception as e:
                    print(e)
            
            if self.vocabulary is not None:
                from sklearn.feature_extraction.text import CountVectorizer
                # Toutes ces étapes permettent de réduire un champ lexical immense à un 
                # champ lexical propre à chaque ligne permettant de faire une vérification 
                # par rapport au set local beaucoup plus rapide
                if self.verbose:
                    print('Vocabulary selection:')
                    print(' Identification des champs lexicaux locaux:')
                count_vectorizer = CountVectorizer(
                    tokenizer=self.tokenizer, 
                    vocabulary=self.vocabulary,
                )
                dftemp = pd.DataFrame(X)
                dftemp.rename(columns={dftemp.columns[0]: 'Token_text'}, inplace=True)
                X = count_vectorizer.fit_transform(X.apply(lambda x: " ".join(x)))
                dftemp['token_select_final'] = count_vectorizer.inverse_transform(X)
                dftemp['token_select_final'] = dftemp['token_select_final'].apply(lambda x: set(x))
                if self.verbose:
                    print('     étape faite')
                    print(' Application des champs lexicaux locaux:')
                    X = dftemp.progress_apply(lambda x: [token for token in x.Token_text if token in x.token_select_final], axis=1)
                else:
                    X = dftemp.apply(lambda x: [token for token in x.Token_text if token in x.token_select_final], axis=1)
                del dftemp
            
            X = X.apply(lambda x: " ".join(x))
        return X

def tokenize_tag(text):
    return re.split('><', text[1:-1])

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

with open("model.pkl", "rb") as fichier:
    loaded_model = cloudpickle.load(fichier)

#### fonction d'utilisation du model ####

app = Flask(__name__)

@app.route('/TagMaker', methods=['POST'])
def tag_maker():
    data = request.get_json()
    x = pd.read_json(data)
    tags_list_arr = y_prepro.inverse_transform(loaded_model.predict(prepro(x)))
    tags = [result.tolist() for result in tags_list_arr]
    return jsonify({'Tags': tags})

if __name__ == '__main__':
    app.run(debug=True)