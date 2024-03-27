import json 
import csv
import gzip
import os
import glob
import sys
import logging
import re
import datetime as dt
import torch
import pandas as pd
import nltk
from simpletransformers.classification import ClassificationModel

def increment_occurrence(dict_to_update, occurrence, weight=1):
    if occurrence in dict_to_update:
        dict_to_update[occurrence] += weight
    else:
        dict_to_update[occurrence] = weight
    return dict_to_update

def intersection(lst1, lst2):
    # from https://www.geeksforgeeks.org/python-intersection-two-lists/
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def safeget(dct, keys):
    # AUTHOR: Diogo Pacheco
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct

def load_csv(filename):
    with open(filename, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        return reader

def load_tweets_json(filename):
    # AUTHOR: Diogo Pacheco
    with gzip.GzipFile(filename) as tw_file:
        data = []
        user = []
        for line in tw_file:
            try:
                tw = json.loads(line.decode('utf-8'))
                user.append(tw["user"])
                data.append(parse_tweet(tw))
            except Exception as e:
                print(filename,e)
        tweets = pd.json_normalize(data)
    
    tweets.drop_duplicates(subset='id_str',inplace=True)
    return tweets

def load_tweets_pkl(filename):
    # written with an eye to the load_tweets_json function
    init = dt.datetime.now()
    print(init, filename,)
    with gzip.GzipFile(filename) as tw_file:
        data = pd.read_pickle(tw_file)
    print(dt.datetime.now()-init, filename)
    return data

def parse_tweet(tw):
    # AUTHOR: Diogo Pacheco
    # has been *heavily* modified from the original version

    entities = {
        '.'.join(k): safeget(tw,k) for k in (
            ('id_str',),
            ("text",)
        )
    }
    return entities

def load_br_stopwords():
    print("Loading nltk corpus...")
    br_stopwords = nltk.corpus.stopwords.words("portuguese")
    print("Corpus loaded.")
    return br_stopwords

def load_tokeniser():
    print("Initialising tokeniser...")
    tknzr = nltk.TweetTokenizer(strip_handles=True, reduce_len=True)
    print("Setup complete.")
    return tknzr

def load_toxicity_model():
    silent = True
    if not torch.cuda.is_available():
        print("Running on fallback with no CUDA. This is not recommended and will take a really long time!")
        print("Enabling progress bars so you get the idea on the amount of work being done.")
        silent = False
    else:
        print("CUDA is available.")
    print("Loading toxicity classification model...")
    tox_model = ClassificationModel( \
        "distilbert", \
        "ToLD-Br/model/toxic_bert_model", \
        use_cuda=torch.cuda.is_available(), \
        args={"silent":silent}
    )
    print("Loaded model.")
    return tox_model

def remove_urls(text):
    pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    return re.sub(pattern, "", text)

def get_tweet_files(dir_path, pairs_only):
    # ORIG. AUTHOR: Diogo Pacheco, modified KBH
    # heavily modified to add to created PKL files
    # instead of creating new PKL files from raw JSON files
    if dir_path.endswith('/'):
        pkl_files = glob.glob(os.path.join(dir_path, '*.pkl.gz'))
        json_files = glob.glob(os.path.join(dir_path, '*.json.gz'))
    else:
        print("Please provide a directory containing pkl.gz and corresponding json.gz files.")
        exit()

    if pairs_only:
        return pkl_json_pairs(pkl_files, json_files)
    else:
        print("Returning %d pkl files and %d json files." % len(pkl_files), len(json_files))
        return sorted(pkl_files), sorted(json_files)

def pkl_json_pairs(pkl_files, json_files):
    raw_filenames_pkl = [value.split('.')[0] for value in pkl_files]
    raw_filenames_json = [value.split('.')[0] for value in json_files]
    intersection_complement = set(raw_filenames_pkl) ^ set(raw_filenames_json)
    if len(intersection_complement) != 0:
        print("%d files found that have a JSON or PKL file, but not both:" % len(intersection_complement))
        print(intersection_complement)
        print("It is recommended to manually check these files and figure out what's happening.")
        print("For now, we will remove them from processing...")
        pkl_files = [file for file in pkl_files if file.split('.')[0] not in intersection_complement]
        json_files = [file for file in json_files if file.split('.')[0] not in intersection_complement]
        print("Removed excess files.")
    if len(pkl_files) != len(json_files):
        print("File amounts mismatch after attempted correction.")
        print("PKL files: %d" % len(pkl_files))
        print("JSON files: %d" % len(json_files))
        exit()
    print("Returning %d matched pkl and json files." % len(pkl_files))
    return sorted(pkl_files), sorted(json_files)