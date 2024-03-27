import json
import logging
import time
import pandas as pd
import spacy
import pt_core_news_sm

logging.basicConfig(filename='logs/sentiment.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def main():
    start_time = time.time()
    bot_id_strs = process_tweets_pkl(pkl_file)
    raw_data = process_tweets_json(json_file, bot_id_strs)
    sentilex = load_sentilex('data/sentilex.txt')
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

def analyse_sentiment(tweets, sentilex):
    logging.info("Beginning sentiment analysis...")
    sentiments = []
    for tweet in tweets:
        tweet_sentiment = 0
        for word in tweet:
            try:
                tweet_sentiment += int(sentilex.at[word.text, "POL:N0"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                tweet_sentiment += int(sentilex.at[word.text, "POL:N1"])
            except (KeyError, ValueError, TypeError):
                continue
        sentiments.append(tweet_sentiment)

    sample = sentiments[:10]
    mean_sentiment = sum(sentiments)/len(sentiments)
    opinionated = [x != 0 for x in sentiments]
    logging.info("Sentiment analysis complete. Sample of first 10: %s", sample)
    logging.info("Average: %d", mean_sentiment)
    logging.info("Opinionated average: %d", sum(opinionated)/len(opinionated))
    return sentiments