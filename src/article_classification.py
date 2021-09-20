import pandas as pd
import numpy as np
import sys, io, os, errno, fileinput, csv
import collections as cl
from os import path
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from settings import mainDataSetPath
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import tensorflow as tf
from tensorflow import keras as kr
from sklearn.metrics.pairwise import  cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


C_STATUS = False
C_PRED = None
F_PRED = False
article_model = None
keras_model = None
vectorizer = None
VERDICT = None
COS_SIM_ACTIVE = False
onion_vectorizer = None

def init_params():

    global C_STATUS, C_PRED
    C_STATUS = False
    C_PRED = None
    return


def svm_load_model():
    global article_model
    print("LOG :: article_classification - load_model")
    if article_model is None:
        article_model = load('models/articleSVM.joblib')
    print("LOG :: article_classification - load_model Success")
    return


def svm_predict(input_article):

    global C_STATUS, C_PRED, F_PRED, article_model
    print("LOG :: article_classification - article Test ", len(input_article) )
    test = []
    test.append(input_article)
    predicted = article_model.predict(test)
    C_PRED = predicted
    print("LOG :: article_classification - article Test result: " + str(predicted))
    if predicted[0] == 1:
        F_PRED = True
    else:
        F_PRED = False
    C_STATUS = True
    return predicted


def kr_load_model():
    global keras_model, vectorizer

    print("LOG :: article_classification - keras_load_model")
    if vectorizer is None:
        vectorizer = pickle.load(open("models/tfidf.pickle", "rb"))
    if keras_model is None:
        keras_model = kr.models.load_model('models/keras_content_classifier.h5')
    print("LOG :: article_classification - keras_load_model Success")


def kr_predict(input_article):

    global C_STATUS, C_PRED, F_PRED, VERDICT ,keras_model, vectorizer, COS_SIM_ACTIVE
    print("LOG :: article_classification - article Test ", len(input_article))
    test = []
    test.append(input_article)
    test_x = vectorizer.transform(test)


    # COSINE SIMILARITY BEFORE CLASSIFICATION

    if COS_SIM_ACTIVE:
        sim_threshold = 0.9 # Similarity Threshold

        tfidf_matrix = vectorizer.fit_transform(test_x)

        cos_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        similarity_vector = []

        for i in range(tfidf_matrix.shape[0]):
            for c in range(len(cos_similarity_matrix[i])):
                if cos_similarity_matrix[i][c] > sim_threshold  and c != i:
                    F_PRED = True
                    VERDICT = "FAKE"
                    return 1
                    # if([c,i,cos_similarity_matrix[i][c]] not in similarity_vector):
                    #    similarity_vector.append([i,c,cos_similarity_matrix[i][c]]) #cos_similarity_matrix[i][c]

    predicted = keras_model.predict_classes(test_x)
    C_PRED = predicted
    print("LOG :: article_classification - article Test result: " + str(predicted))
    if predicted[0] == 1:
        F_PRED = True
        VERDICT = "FAKE"
    else:
        F_PRED = False
        VERDICT = "REAL"
    C_STATUS = True
    return predicted


""" 
    count_vect = CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer(smooth_idf=False)

    trainset_file = mainDataSetPath + "/train_1.csv"
    train_df = pd.read_csv(trainset_file, sep=',')

    kernel: linear
    misclassification penalty factor C: 1.1
    Multiple Class Approach decision function: ovo (one versus one ) || ovr (one versus rest)
    1) linear kernel
    2) Radial Basis Function (RBF) kernel: set gamma value (e.g. 2^-4)

    svm_bow = Pipeline([
    ('vect', CountVectorizer(analyzer='word', stop_words='english', lowercase=True)),
    ('tfidf', TfidfTransformer()),
    ('svm', svm.SVC(kernel='linear', C=0.92, decision_function_shape='ovr')), ])

    def train_model():
        print("LOG :: article_classification - train_model")
        train_X = train_df['text'][10:1607].astype('U')
        train_Y = train_df['label'][10:1607]
        _ = svm_bow.fit(train_X, train_Y)
        print("LOG :: article_classification - train_model Success")
        return

    def handle_acd_query(query):
        train_model()
        res = test_model(query)
        return res
"""
