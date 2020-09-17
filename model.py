# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:55:47 2020

@author: bandi
i"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
train = pd.read_csv('train_tGmol3O.zip')

def field(col):
    cs = col[0]
    ph = col[1]
    Ma = col[2]
    st = col[3]
    qb = col[4]
    qf = col[5]
    
    if cs==1:
        return 1
    elif ph==1:
        return 2
    elif Ma==1:
        return 3
    elif st==1:
        return 4
    elif qb==1:
        return 5
    elif qf==1:
        return 6

train['cat'] = train[['Computer Science', 'Physics', 'Mathematics',
       'Statistics', 'Quantitative Biology', 'Quantitative Finance']].apply(field,axis=1)

train.drop(['Computer Science', 'Physics', 'Mathematics',
       'Statistics', 'Quantitative Biology', 'Quantitative Finance'],axis=1,inplace=True)

train['ABSTRACT'] = train['TITLE'] + train['ABSTRACT']

train.drop(['TITLE','ID'],axis=1,inplace=True)

def stpwrds(text):
    text_tokens = word_tokenize(text)
    return [word for word in text_tokens if not word in stopwords.words()]

xtrain = train['ABSTRACT']
ytrain = train['cat']

CV = CountVectorizer(analyzer=stpwrds)
bow = CV.fit_transform(xtrain)

x_train,y_train = SMOTE().fit_sample(tf,ytrain)

def model(model,x_train,y_train,tf,gp=None):
    model.fit(x_train,y_train)
    pred = model.predict(tf)
    return model,f1_score(ytrain,pred,average='micro')

nb = MultinomialNB(alpha=0.5).fit(x_train,y_train)

pickle.dump(nb,open('model.sav','wb'))`