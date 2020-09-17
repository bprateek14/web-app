# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:49:39 2020

@author: bandi
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import string

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
CV = pickle.load(open('CV.pkl', 'rb'))
tf = pickle.load(open('tf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html') 

def text(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words = " ".join(words)
    return words

def field(x):
    
        if x==1:
            return "computer science"
        elif x==2:
            return "Physics"
        elif x==3:
            return "mathematics"
        elif x==4:
            return "Statistics"
        elif x==5:
            return "Quantitative Biology"
        elif x==6:
            return "Quantitative Finance"

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_feature = request.form.values()
    
    final_feature = [text(str(int_feature))]
    bow = CV.transform(final_feature)
    final = tf.transform(bow)

    prediction = model.predict(final)
    
    output = field(prediction)

    return render_template('index.html', prediction_text='Abstract belongs to :  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
