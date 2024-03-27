# coding: utf-8
import sys
import logging
import datetime as dt
import pandas as pd
import disslib

# set up logging
logging.basicConfig(filename='logs/reprocessor.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def main(args):
    pkl_files, json_files = disslib.get_tweet_files(args[1], pairs_only=True)
    for i, pkl_file in enumerate(pkl_files):
        base_data = pd.read_pickle(pkl_file)
        json_file = json_files[i]
        new_data = disslib.load_tweets_json(json_file)

        #print(base_data)
        #print(new_data)
        
        reprocessed = pd.merge(base_data, new_data, how="left", on="id_str")
        print(reprocessed)
        
if __name__ == '__main__':
    main(sys.argv[1:])