"""
Tiny script that returns the columns in each type of file

Returns:
    logs/columns.log: Logfile of columns in each file
"""

import logging
import json
import pandas as pd
import disslib

# set up logging
logging.basicConfig(filename='logs/columns.log',  \
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

pkl_data_repro = pd.read_pickle("data/testdata/elections2022_tweets-20220701_REPROCESSED.pkl.gz")
pkl_data = pd.read_pickle("data/testdata/elections2022_tweets-20220701.pkl.gz")
json_data = disslib.load_tweets_json("data/testdata/elections2022_tweets-20220806.json.gz")

print(pkl_data)
print(json_data)

print(pkl_data["quoted_status.id_str"])
print(pkl_data["retweeted_status.id_str"])

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
