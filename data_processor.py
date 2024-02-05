import logging
import pickle
import json

logging.basicConfig(filename='main.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def read_pkl(filename):
    logging.info("Reading pickle %s...", filename)
    obj = pd.read_pickle(filename)
    #bots = obj.loc[obj['botscore'] > 0.7]
    id_strs = set(obj["id_str"])
    logging.info("Pickle processed. ID set size: %d", len(id_strs))
    return id_strs


def read_json(filename, bot_id_strs):
    logging.info("Processing raw JSON data...")
    raw_data = []
    with open(filename, encoding='utf8') as json_file:
        for line in json_file:
            line_data = json.loads(line)
            raw_data.append(line_data)
    logging.info("JSON processed.")
    return raw_data