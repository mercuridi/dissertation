"""
i just want to know how many datapoints there actually are
"""

import sys 
import glob
import os
import pandas as pd
import disslib

def main(args):
    pkl_counter()
    #csv_counter()

def csv_counter():
    csv_files = glob.glob(os.path.join("data/2_hashtag_stbm/", '*.csv'))
    total_rows = 0
    for file in csv_files:
        with open(file, "r", encoding="utf-8") as csv_handle:
            csv_as_df = pd.read_csv(csv_handle, delimiter=" ")
            total_rows += len(csv_as_df.index)
    
    print(f"Total datapoints: {total_rows}")

def pkl_counter():
    pkl_files, json_files = disslib.get_tweet_files("data/elections2022/", True)
    total_rows = 0
    no_hashtags_rows = 0
    no_retweet_rows = 0
    for file in pkl_files:
        tweets = disslib.load_tweets_pkl(file)
        #print(tweets)
        #print(tweets.columns)
        total_rows += len(tweets.index)
        no_nan = tweets[tweets["hashtags"].notna()]
        no_hashtags_rows += len(no_nan.index)
        no_retweets = no_nan[no_nan["retweeted_status.id_str"].isnull()]
        no_retweet_rows += len(no_retweets.index)
    print(f"Total datapoints: {total_rows}")
    print(f"Total with no hashtags filtered: {no_hashtags_rows}")
    print(f"Total with no retweets: {no_retweet_rows}")

if __name__ == '__main__':
    main(sys.argv[1:])