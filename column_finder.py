import logging
import pickle
import json
import pandas as pd

logging.basicConfig(filename='columns.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def read_pickle(filename):
    logging.info("Reading pickle %s...", filename)
    with open(filename, mode="rb") as f:
        return pd.read_pickle(f)

def read_json(filename):
    logging.info("Reading JSON %s...", filename)
    with open(filename, mode="r", encoding="utf8") as f:
        data = pd.DataFrame(json.loads(line) for line in f)
        return data

pkl_data_repro = pd.read_pickle("data/smalldata/elections2018_tweets-20180830_REPROCESSED.pkl.gz")
pkl_data = pd.read_pickle("data/smalldata/elections2018_tweets-20180830.pkl")
json_data = read_json("data/smalldata/elections2018_tweets-20180830.json")

logging.info("")
logging.info("PKL columns (REPRO):")
for col in pkl_data_repro.columns:
    logging.info(col)

logging.info("")
logging.info("PKL columns:")
for col in pkl_data.columns:
    logging.info(col)

logging.info("")
logging.info("JSON columns:")
for col in json_data.columns:
    logging.info(col)
