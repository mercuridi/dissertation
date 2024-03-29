import json 
import csv
import gzip
import os
import glob
import sys
import logging
import string
import re
import datetime as dt
import torch
import pandas as pd
import nltk
import spacy
import pt_core_news_sm
from simpletransformers.classification import ClassificationModel

URL_PATTERN = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"

def preprocess_text(text_list, stopwords, pt_core):
    # pipeline from https://spotintelligence.com/2022/12/21/nltk-preprocessing-pipeline
    processed_list = []
    for text in text_list:
        text = text.strip()
        text = " ".join(text.split())
        cleaned_text = re.sub(URL_PATTERN, "", text)
        tokens = nltk.word_tokenize(cleaned_text)
        lowercased_tokens = [token.lower() for token in tokens]
        nopunc_tokens = [token for token in lowercased_tokens if token not in string.punctuation]
        filtered_tokens = [token for token in nopunc_tokens if token.lower() not in stopwords]
        rejoined_tokens = " ".join(filtered_tokens)
        doc = pt_core(rejoined_tokens)
        #print(doc.text)
        processed_list.append(doc.text)
    return processed_list
        
def process_nlp(text_list, pt_core):
    logging.info("Performing sentiment NLP via spaCy2...")
    nlp_processed = []
    for text in text_list:
        doc = pt_core(text)
        nlp_processed.append(doc)
        #logging.info([(w.text, w.pos_) for w in doc])
    logging.info("NLP processing complete: %d records.", len(nlp_processed))
    return nlp_processed

def analyse_sentiment(text_list, sentilex_dataframe, pt_core):
    logging.info("Beginning sentiment analysis...")
    processed_text_list = process_nlp(text_list, pt_core)
    sentiments = []
    for text in processed_text_list:
        text_sentiment = 0
        for word in text:
            try:
                text_sentiment += int(sentilex_dataframe.at[word.text, "POL:N0"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                text_sentiment += int(sentilex_dataframe.at[word.text, "POL:N1"])
            except (KeyError, ValueError, TypeError):
                continue
        sentiments.append(text_sentiment)
    logging.info("Sentiment analysis complete.")
    return sentiments

def load_pt_core():
    logging.info("Loading Portuguese NLP library.")
    nlp = spacy.load("pt_core_news_sm")
    logging.info("Portuguese NLP loaded.")
    return nlp

def load_sentilex(filename="data/sentilex.txt"):
    logging.info("Loading SentiLex-PT02 from file...")
    sentilex = pd.read_csv(filename, sep = "|", index_col=0, header=0)
    logging.info("Loaded SentiLex from txt file.")
    return sentilex

def nicetime(start_time, end_time):
    time_diff = int(end_time-start_time)
    if time_diff > 3600:
        hours   = str(time_diff // 3600)
        minutes = str((time_diff - (int(hours) * 3600)) // 60)
        seconds = str(time_diff % 60)
    elif time_diff > 60:
        hours   = "00"
        minutes = str(time_diff // 60)
        seconds = str(time_diff % 60)
    else:
        hours   = "00"
        minutes = "00"
        seconds = str(time_diff)
    if int(hours) < 10 and hours != "00":
        hours = "0"+hours
    if int(minutes) < 10 and minutes != "00":
        minutes = "0"+minutes
    if int(seconds) < 10:
        seconds = "0"+seconds
    return (hours+":"+minutes+":"+seconds).rjust(9)


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
    #print(init, filename,)
    with gzip.GzipFile(filename) as tw_file:
        data = pd.read_pickle(tw_file)
    #print(dt.datetime.now()-init, filename)
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

def load_toxicity_model(silent=True):
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
        print("%d files found that have a JSON or PKL file, but not both." % len(intersection_complement))
        #print(intersection_complement)
        #print("It is recommended to manually check these files and figure out what's happening.")
        #print("For now, we will remove them from processing...")
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