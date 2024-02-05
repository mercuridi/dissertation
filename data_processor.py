import logging
import pickle
import json
import pandas as pd

logging.basicConfig(filename='reprocessor.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def read_pkl(filename):
    logging.info("Reading pickle %s...", filename)
    with open(filename, mode="rb") as f:
        return pd.read_pickle(f)

def read_json(filename):
    logging.info("Reading JSON %s...", filename)
    with open(filename, mode="r", encoding="utf8") as f:
        data = pd.DataFrame(json.loads(line) for line in f)
        return data

pkl_data = read_pkl("data/elections22/elections2022_tweets-20220701.pkl")
json_data = read_json("data/elections22/elections2022_tweets-20220701.json")

logging.info("")
logging.info("PKL columns:")
for col in pkl_data.columns:
    logging.info(col)

logging.info("")
logging.info("JSON columns:")
for col in json_data.columns:
    logging.info(col)

new_data = []
for index, entry in enumerate(pkl_data):
    corresponding = json_data.iloc[index]
    new_data = entry
    new_data.update({"text": corresponding.text})
    break

print(new_data)
