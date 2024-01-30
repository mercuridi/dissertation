import json
import logging
import time
import pandas as pd
import spacy
import pt_core_news_sm

logging.basicConfig(filename='main.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def main():
    start_time = time.time()
    bot_id_strs = process_tweets_pkl('data/elections2018_tweets-20180830.pkl')
    raw_data = process_tweets_json('data/elections2018_tweets-20180830.json', bot_id_strs)
    sentilex = load_sentilex('data/sentilex.txt')
    print(sentilex)
    nlp_start = time.time()
    nlp = load_pt_core()
    nlp_processed = process_nlp(raw_data, nlp)
    analyse_sentiment(nlp_processed, sentilex)
    nlp_end = time.time()
    end_time = time.time()
    time_elapsed = end_time - start_time
    nlp_time = nlp_end - nlp_start
    logging.info("Time elapsed: %f", time_elapsed)
    logging.info("Of which NLP: %f", nlp_time)

def process_tweets_pkl(filename):
    logging.info("Processing pickle...")
    obj = pd.read_pickle(filename)
    bots = obj.loc[obj['botscore'] > 0.7]
    bot_id_strs = set(bots["id_str"])
    logging.info("Pickle processed. ID set size: %d", len(bot_id_strs))
    return bot_id_strs

def process_tweets_json(filename, bot_id_strs):
    logging.info("Processing raw JSON data...")
    raw_data = []
    with open(filename, encoding='utf8') as json_file:
        for line in json_file:
            line_data = json.loads(line)
            if line_data["id_str"] in bot_id_strs:
                raw_data.append(line_data["text"])
    logging.info("JSON processed.")
    return raw_data

def load_pt_core():
    logging.info("Loading Portuguese NLP library.")
    nlp = spacy.load("pt_core_news_sm")
    nlp = pt_core_news_sm.load()
    logging.info("Portuguese NLP loaded.")
    return nlp

def load_sentilex(filename):
    logging.info("Loading SentiLex-PT02 from file...")
    sentilex = pd.read_csv(filename, sep = "|", index_col=0, header=0)
    logging.info("Loaded SentiLex from txt file.")
    return sentilex

def process_nlp(raw_data, nlp):
    logging.info("Performing NLP via spaCy2...")
    nlp_processed = []
    for text in raw_data:
        doc = nlp(text)
        nlp_processed.append(doc)
        #logging.info([(w.text, w.pos_) for w in doc])
    logging.info("NLP processing complete: %d records.", len(nlp_processed))
    return nlp_processed

def analyse_sentiment(all_data, sentilex):
    sentiments = []
    for text in all_data:
        text_sentiment = 0
        for word in text:
            try:
                text_sentiment += int(sentilex.at(word, "POL:N0"))
            except (KeyError, ValueError):
                continue
            try:
                text_sentiment += int(sentilex.at(word, "POL:N1"))
            except (KeyError, ValueError):
                continue
    print(sentiments)
                
main()
