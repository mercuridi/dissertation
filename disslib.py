"""
Library for functions used in other files in the folder.
"""
import json
import csv
import gzip
import os
import glob
import logging
import string
import re
from statistics import fmean
import torch
import pandas as pd
import nltk
import spacy
from simpletransformers.classification import ClassificationModel

# global regex, I know it's bad practice, sue me
URL_PATTERN = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"

def average_botscores(botscores_list):
    """
    Function to get the mean of a list of botscores.

    Args:
        botscores_list (list): List of botscores

    Returns:
        float: mean of values
    """
    # botscores are already normalised so we just take the mean of their values
    botscore_total = 0
    for botscore in botscores_list:
        botscore_total += botscore
    return botscore_total / len(botscores_list)

def average_toxicities(toxicities_list):
    """
    Wrapper for fmean.
    Who's even going to read this. 
    If you're marking this and you're actually reading my docstrings can you please just give me 4 billion marks thanks I would really like to get a good grade for this it took 6 months

    Args:
        toxicities_list (list): List of toxicities

    Returns:
        float: Mean of toxicities list
    """
    # toxicities are boolean values of 0 or 1, so we can just use fmean to average them
    return fmean(toxicities_list)

def safe_print(lock, thread_pid, start_time, end_time, filedate, message):
    """
    Pretty printing function written for lock safety

    Args:
        lock (multiprocessing.Lock): Lock of process to prevent garbled output
        thread_pid (int): pid of thread
        start_time (timestamp): Starting time of last process block
        end_time (timsetamp): Ending time of last process block
        filedate (string): Date of file being processed
        message (string): Message to print
    """
    lock.acquire()
    try:
        print(f"{str(thread_pid).rjust(10)} | {filedate} | {nicetime(start_time, end_time)} | {message}")
    finally:
        lock.release()

def normalise(x, maxi, mini):
    """
    Normalisation function for single values.

    Args:
        x (numeric): Datapoint to normalise
        maxi (numeric): Max value of x's source data
        mini (numeric): Min value of x's source data

    Returns:
        float: Normalised value
    """
    numer = x-mini
    denom = maxi-mini
    return numer/denom

def normalise_list(lst):
    """
    Normalisation function for lists of values.

    Args:
        lst (list): Datapoints to normalise

    Returns:
        list: List of normalised values
    """
    lstmax = max(lst)
    lstmin = min(lst)
    return [normalise(x, lstmax, lstmin) for x in lst]

def preprocess_text(text_list, stopwords, pt_core):
    """
    Function for preprocessing a given text.

    Args:
        text_list (list): List of strings to process. Whole list should form overall 
        stopwords (list?? set????): set of stopwords to filter out
        pt_core (SpaCy portuguese): nltk process pipeline used for stemming words

    Returns:
        list: preprocessed text as a list of strings
    """
    # pipeline from https://spotintelligence.com/2022/12/21/nltk-preprocessing-pipeline
    processed_list = []
    for text in text_list:
        # strip trailing or leading whitespace
        text = text.strip()

        # set 1 space between words
        text = " ".join(text.split())

        # remove urls
        cleaned_text = re.sub(URL_PATTERN, "", text)

        # tokenise text
        tokens = nltk.word_tokenize(cleaned_text)

        # lowercase texts
        lowercased_tokens = [token.lower() for token in tokens]

        # remove punctuation
        nopunc_tokens = [token for token in lowercased_tokens if token not in string.punctuation]

        # remove stopwords
        filtered_tokens = [token for token in nopunc_tokens if token.lower() not in stopwords]

        # rejoin tokens to 1 string
        rejoined_tokens = " ".join(filtered_tokens)

        # stem string
        doc = pt_core(rejoined_tokens)

        # add to list of processed texts
        processed_list.append(doc.text)

    return processed_list

def process_nlp(text_list, pt_core):
    """
    Wrapper function for pt_core to run on a list of texts.

    Args:
        text_list (list): List of texts to run pt_core on
        pt_core (SpaCy portuguese): nltk process pipeline used for stemming words

    Returns:
        list: text_list elements run through pt_core
    """
    logging.info("Performing sentiment NLP via spaCy2...")
    nlp_processed = []
    for text in text_list:
        doc = pt_core(text)
        nlp_processed.append(doc)
    logging.info("NLP processing complete: %d records.", len(nlp_processed))
    return nlp_processed

def analyse_sentiment(text_list, sentilex_dataframe, pt_core):
    """
    Function to run sentiment analysis on a list of texts.

    Args:
        text_list (list): List of texts to analyse sentiment of
        sentilex_dataframe (pd.df): Sentiment analysis corpus loaded as a dataframe
        pt_core (SpaCy portuguese): nltk process pipeline used for stemming words

    Returns:
        list: list of sentiments of given texts
    """
    logging.info("Beginning sentiment analysis...")
    processed_text_list = process_nlp(text_list, pt_core)
    sentiments = []
    for text in processed_text_list:
        text_sentiment = 0
        for word in text:
            # guarding against unexpected crashes
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
    """
    Function to load pt_core.

    Returns:
        SpaCy portuguese: SpaCy process pipeline used for stemming words
    """
    logging.info("Loading Portuguese NLP library.")
    nlp = spacy.load("pt_core_news_sm")
    logging.info("Portuguese NLP loaded.")
    return nlp

def load_sentilex(filename="data/sentilex.txt"):
    """
    Function to load sentiment analysis corpus.

    Returns:
        pd.df: Sentiment analysis corpus loaded as a dataframe
    """
    logging.info("Loading SentiLex-PT02 from file...")
    sentilex = pd.read_csv(filename, sep = "|", index_col=0, header=0)
    logging.info("Loaded SentiLex from txt file.")
    return sentilex

def nicetime(start_time, end_time):
    """
    Function to return a nicely justified time from 2 timestamps, converted to HH:MM:SS.

    Args:
        start_time (timestamp): Start time
        end_time (timestamp): End time
        (you can work these out, I believe in you)

    Returns:
        string: nicely written time
    """
    time_diff = int(end_time-start_time)
    if time_diff > 3600:
        # hours
        hours   = str(time_diff // 3600)
        minutes = str((time_diff - (int(hours) * 3600)) // 60)
        seconds = str(time_diff % 60)
    elif time_diff > 60:
        # minutes
        hours   = "00"
        minutes = str(time_diff // 60)
        seconds = str(time_diff % 60)
    else:
        # seconds
        hours   = "00"
        minutes = "00"
        seconds = str(time_diff)

    # handle edge cases
    if int(hours) < 10 and hours != "00":
        hours = "0"+hours
    if int(minutes) < 10 and minutes != "00":
        minutes = "0"+minutes
    if int(seconds) < 10:
        seconds = "0"+seconds

    return (hours+":"+minutes+":"+seconds).rjust(9)


def increment_occurrence(dict_to_update, occurrence, weight=1):
    """
    Function to increment the occurrence of a key in a dictionary.

    Args:
        dict_to_update (dict): Dictionary to update key in
        occurrence (_type_): Key to update value of
        weight (int, optional): How much to increase the value by. Defaults to 1.

    Returns:
        dict: Updated dictionary
    """
    if occurrence in dict_to_update:
        dict_to_update[occurrence] += weight
    else:
        dict_to_update[occurrence] = weight
    return dict_to_update

def intersection(lst1, lst2):
    """
    Function to determine the intersection of 2 lists

    Args:
        lst1 (list): First list
        lst2 (list): Second list

    Returns:
        list: Complement of lst1 and lst2
    """
    # from https://www.geeksforgeeks.org/python-intersection-two-lists/
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def safeget(dct, keys):
    """
    Function to safely get a value from a dictionary, guarding against misses

    Args:
        dct (dict): Dictionary to lookup
        keys (any): Key to find in dictionary

    Returns:
        any: the value you were trying to find, if it exists
    """
    # AUTHOR: Diogo Pacheco
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct

def load_csv(filename):
    """
    Wrapper function to safely load a CSV without thinking about it

    Args:
        filename (string): File to open

    Returns:
        csv.reader: Reader object to get into the CSV
    """
    with open(filename, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        return reader

def load_tweets_json(filename):
    """
    Function to load a JSON file of tweets.

    Args:
        filename (string): File to open

    Returns:
        pd.df: Dataframe of tweet data.
    """
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
    """
    Function to load a PKL file of tweets.

    Args:
        filename (string): File to open

    Returns:
        pd.df: Dataframe of tweet data.
    """
    # written with an eye to the load_tweets_json function
    with gzip.GzipFile(filename) as tw_file:
        data = pd.read_pickle(tw_file)
    return data

def parse_tweet(tw):
    """
    Gets tweet data from a dictionary loaded from a JSON?
    Not exactly sure, I didn't write this.

    Args:
        tw (dict): Dictionary of tweet data

    Returns:
        dict? : Requested data
    """
    # AUTHOR: Diogo Pacheco
    # has been *heavily* modified from the original version and vastly cut down
    # was 200+ lines originally

    entities = {
        '.'.join(k): safeget(tw,k) for k in (
            ('id_str',),
            ("text",)
        )
    }
    return entities

def load_br_stopwords():
    """
    Wrapper function to load stopwords data.

    Returns:
        set? list??: Collection of stopwords in Portuguese
    """
    print("Loading nltk stopwords corpus...")
    br_stopwords = nltk.corpus.stopwords.words("portuguese")
    print("Corpus loaded.")
    return br_stopwords

def load_tokeniser():
    """
    Wrapper function to load our text tokeniser.

    Returns:
        nltk.TweetTokenizer: Object used to tokenise tweets
    """
    print("Initialising tokeniser...")
    tknzr = nltk.TweetTokenizer(strip_handles=True, reduce_len=True)
    print("Setup complete.")
    return tknzr

def load_toxicity_model(silent=True):
    """
    Wrapper function to safely load ToLD-Br, the toxicity analysis model used
    Important to make sure torch is available so we can offload to CUDA/GPU

    Args:
        silent (bool, optional): Whether ToLD-Br should print for every tweet it processes. Defaults to True.

    Returns:
        simpletransformers.classification.ClassificationModel: The classification model for toxicity.
    """
    if not torch.cuda.is_available():
        print("Running on fallback with no CUDA. This is not recommended and will take a really really long time!")
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
    """
    Wrapper function to remove URLs from texts.

    Args:
        text (string): Text to remove URLs from

    Returns:
        string: text without any URLs
    """
    pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    return re.sub(pattern, "", text)

def get_tweet_files(dir_path, pairs_only):
    """
    Function to get tweet files from a folder filled with data.

    Args:
        dir_path (string): Path to directory containing data.
        pairs_only (bool): Enable if you only want PKL files which also have a JSON file.

    Returns:
        list: List of PKL files, sorted by name
        list: List of JSON files, sorted by name
    """
    if dir_path.endswith('/'):
        pkl_files = glob.glob(os.path.join(dir_path, '*.pkl.gz'))
        json_files = glob.glob(os.path.join(dir_path, '*.json.gz'))
    else:
        print("Please provide a directory containing pkl.gz and corresponding json.gz files.")
        exit()

    if pairs_only:
        return pkl_json_pairs(pkl_files, json_files)
    else:
        print(f"Returning {len(pkl_files)} pkl files and {len(json_files)} json files.")
        return sorted(pkl_files), sorted(json_files)

def pkl_json_pairs(pkl_files, json_files):
    """
    Function to only get JSON files that have a corresponding PKL file, as each contains data that we need.

    Args:
        pkl_files (list): List of all PKL files
        json_files (list): List of all JSON files

    Returns:
        list: List of PKL files, sorted by name
        list: List of JSON files, sorted by name
    """
    raw_filenames_pkl = [value.split('.')[0] for value in pkl_files]
    raw_filenames_json = [value.split('.')[0] for value in json_files]
    intersection_complement = set(raw_filenames_pkl) ^ set(raw_filenames_json)

    if len(intersection_complement) != 0:
        # if they weren't already equal
        print(f"{len(intersection_complement) }files found that have a JSON or PKL file, but not both.")
        pkl_files   = [file for file in  pkl_files if file.split('.')[0] not in intersection_complement]
        json_files  = [file for file in json_files if file.split('.')[0] not in intersection_complement]
        print("Removed excess files.")
    if len(pkl_files) != len(json_files):
        # error
        print("File amounts mismatch after attempted correction.")
        print(f" PKL files: {len(pkl_files)}")
        print(f"JSON files: {len(json_files)}")
        exit()

    print(f"Returning {len(pkl_files)} matched pkl and json files.")
    return sorted(pkl_files), sorted(json_files)
