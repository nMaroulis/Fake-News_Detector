import pandas as pd
import numpy as np
import sys, io, os, errno, fileinput, csv
import collections as cl
from os import path


trainset_file = "Datasets/the_onion.csv"
trainset_file1 = "Datasets/train_2.csv"

CALCULATED = False
UNTRUSTED = []


def load_untrusted():

    global UNTRUSTED, CALCULATED
    if not UNTRUSTED:
        train_df = pd.read_csv(trainset_file, sep=',')
        df = train_df['domain']
        df = df.drop_duplicates()
        train_df1 = pd.read_csv(trainset_file1, sep=',')
        df1 = train_df1[(train_df1['spam_score'] > 0.7)]
        df1 = df1['site_url']
        df1 = df1.drop_duplicates()
        df.head(3)

        UNTRUSTED = df.values.tolist()
        UNTRUSTED.extend(df1.values.tolist())

        UNTRUSTED.remove('google.com')
        UNTRUSTED.remove('youtube.com')
        UNTRUSTED.remove('analytics.twitter.com')
        UNTRUSTED.remove('docs.google.com')
        UNTRUSTED.remove('twitter.com')
        UNTRUSTED.remove('m.youtube.com')
        UNTRUSTED.remove('dailymail.co.uk')
        UNTRUSTED.remove('instagram.com')
        UNTRUSTED.remove('cnn.com')
        CALCULATED = True
    return

