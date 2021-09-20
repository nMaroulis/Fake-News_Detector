import json
from itertools import islice
import re
import os
import collections
from time import sleep
from settings import mainDataSetPath
from src.dataset_funcs import str_cleansing

TOP50_WORDS = collections.Counter()
total_articles = 0
total_words = 0
calculated = False
i = 0
