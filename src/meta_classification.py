from joblib import dump, load

import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


MC_STATUS = False
MC_PRED = None
MF_PRED = False
MVERDICT = None
M_COS_SIM_ACTIVE = False

meta_vectorizer = None
meta_model = None
svm_model = None


def init_params():

    global MC_STATUS, MC_PRED
    MC_STATUS = False
    MC_PRED = None
    return


def svm_load_model():
    global svm_model, meta_vectorizer

    print("LOG :: meta_classification - svm_load_model")
    if meta_vectorizer is None:
        meta_vectorizer = pickle.load(open("models/meta_tfidf.pickle", "rb"))
    if svm_model is None:
        svm_model = load('models/metaSVM.joblib')

    print("LOG :: meta_classification - svm_load_model Success")


def svm_predict(title_input, domain_input):

    global MC_STATUS, MC_PRED, MF_PRED, MVERDICT, svm_model, meta_vectorizer, M_COS_SIM_ACTIVE
    print("LOG ::meta_classification - article Test ", len(title_input))
    test = []
    test.append(title_input)
    test_x = meta_vectorizer.transform(test)

    # COSINE SIMILARITY BEFORE CLASSIFICATION
    if M_COS_SIM_ACTIVE:
        sim_threshold = 0.9 # Similarity Threshold
        tfidf_matrix = meta_vectorizer.fit_transform(test_x)
        cos_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        for i in range(tfidf_matrix.shape[0]):
            for c in range(len(cos_similarity_matrix[i])):
                if cos_similarity_matrix[i][c] > sim_threshold and c != i:
                    MF_PRED = True
                    MVERDICT = "FAKE"
                    return 1

    predicted = svm_model.predict(test_x)
    MC_PRED = predicted
    print("LOG :: article_classification - article Test result: " + str(predicted))
    if predicted[0] == 1:
        MF_PRED = True
        MVERDICT = "FAKE"
    else:
        MF_PRED = False
        MVERDICT = "REAL"
    MC_STATUS = True
    return predicted
