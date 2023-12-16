#### importations ####

import mlflow
import dill
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import os
from flask import Flask, request, jsonify
from classes import TextConcatWithWeightTransformer, NLPTextTransformer, tokenize_tag

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

def scan_directory(root_path, indent=0):
    """
    Scan and print the file and folder hierarchy of the specified directory.

    Parameters:
    - root_path (str): The root directory to start scanning.
    - indent (int): The current level of indentation for displaying the hierarchy.
    """
    try:
        # List all files and folders in the current directory
        items = os.listdir(root_path)

        for item in items:
            # Create the full path of the current item
            full_path = os.path.join(root_path, item)

            # Print the current item with appropriate indentation
            print("  " * indent + f"- {item}")

            # If the current item is a directory, recursively scan its contents
            if os.path.isdir(full_path):
                scan_directory(full_path, indent + 1)

    except Exception as e:
        print(f"Error scanning directory '{root_path}': {e}")

# Utilisation de la fonction pour scanner un répertoire spécifique (par exemple, le répertoire de travail actuel)

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