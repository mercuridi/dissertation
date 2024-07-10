import glob
import os
import datetime
import random
import numpy as np
import pandas as pd
import disslib

def main():
    day_by_day()
    day_combiner()

def day_combiner():
    all_filenames = glob.glob(os.path.join("data/POSTGRAD/SOROUSH/", '*.csv'))
    main_df = pd.read_csv(all_filenames.pop())
    for filename in all_filenames:
        main_df = pd.concat([main_df, pd.read_csv(filename)], ignore_index=True)
    
    main_df.to_csv("data/POSTGRAD/SOROUSH/requested_data.csv", sep=";", encoding="utf-8", index=False)

def day_by_day():
    with open("data/2_hashtag_stbm/2_H_STBM_TWEETS.csv", "r", encoding="utf-8") as mod_handle:
        # open tweets csv
        all_tweets = pd.read_csv(mod_handle, delimiter=";")
        
        all_tweets.drop(["timestamp_ms","quoted_status.id_str","hashtags","sentiment","modularity"], axis=1, inplace=True)
        all_tweets["id_str"] = pd.to_numeric(all_tweets["id_str"])
        print(all_tweets)
        print("Loaded all tweets with tox data.")

    all_filenames = glob.glob(os.path.join("data/elections2022/elections22/", '*.json.gz'))
    done_filenames = [str(x) for x in glob.glob(os.path.join("data/POSTGRAD/SOROUSH/", '*.csv'))]

    random.shuffle(all_filenames)
    for filename in all_filenames:
        filedate = filename.split(".")[0].split("-")[1]
        out_filename = "data/POSTGRAD/SOROUSH/SOROUSH" + filedate + ".csv"
        #print(out_filename)
        #print(done_filenames)
        if out_filename in done_filenames:
            print("Skipping finished file", out_filename)
            continue
        else:
            print("Loading file", filename)
            current_file = disslib.load_tweets_json(filename)
            current_file["id_str"] = pd.to_numeric(current_file["id_str"])
            #print(current_file)
            print("Merging dataframes...")
            updated_tweets = pd.merge(all_tweets, current_file, how="left", on="id_str")
            updated_tweets = updated_tweets.dropna()
            print(updated_tweets)
            print("^ New dataframe")
    
            updated_tweets["user.id_str"] = pd.to_numeric(updated_tweets["user.id_str"])
            updated_tweets.rename(columns={"id_str": "id", "user.id_str": "user.id"}, inplace=True)
            updated_tweets.to_csv(out_filename, sep=";", encoding="utf-8", index=False)
            print("Wrote to file")
    print("Done")
    
main()
