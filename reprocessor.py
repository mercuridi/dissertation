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

# regex for removing urls in text processing

def main(args):
    pkl_files, json_files = disslib.get_tweet_files(args[1], pairs_only=True)
    for i, pkl_file in enumerate(pkl_files):
        pkl_init = dt.datetime.now()
        print(i, ":", pkl_init, pkl_file,)
        base_data = pd.read_pickle(pkl_file)
        print(i, ":", dt.datetime.now()-pkl_init, pkl_file)

        json_file = json_files[i]
        json_init = dt.datetime.now()
        print(i, ":", json_init, json_file)
        new_data = disslib.load_tweets_json(json_file)
        print(i, ":", dt.datetime.now()-json_init, json_file)

        #print(base_data)
        #print(new_data)
        
        reprocessed = pd.merge(base_data, new_data, how="left", on="id_str")
        print(reprocessed)
        
if __name__ == '__main__':
    main(sys.argv[1:])