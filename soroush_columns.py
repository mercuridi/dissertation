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
    all_filenames = glob.glob(os.path.join("data/POSTGRAD/with_retweets/", '*.pkl.tar.gz'))
    main_df = pd.read_pickle(all_filenames.pop())
    for filename in all_filenames:
        main_df = pd.concat([main_df, pd.read_pickle(filename)], ignore_index=True)
    
    main_df.to_pickle("data/POSTGRAD/with_retweets/with_retweets.pkl.tar.gz", index=False)

def day_by_day():
    with open("data/2_hashtag_stbm/2_H_STBM_TWEETS.csv", "r", encoding="utf-8") as mod_handle:
        # open tweets csv
        all_tweets = pd.read_csv(mod_handle, delimiter=";")
        
        all_tweets.drop(["sentiment","modularity", "quoted_status.id_str"], axis=1, inplace=True)
        all_tweets["id_str"] = pd.to_numeric(all_tweets["id_str"])
        print(all_tweets)
        print("Loaded all tweets with tox data.")

    all_filenames = glob.glob(os.path.join("data/elections2022/elections22/", '*.json.gz'))
    done_filenames = [str(x) for x in glob.glob(os.path.join("data/POSTGRAD/with_retweets/", '*.pkl.tar.gz'))]

    random.shuffle(all_filenames)
    for filename in all_filenames:
        filedate = filename.split(".")[0].split("-")[1]
        out_filename = "data/POSTGRAD/with_retweets/rtws" + filedate + ".pkl.tar.gz"
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
            print("Handling retweets...")
            day_rtws = handle_retweets(current_file, all_tweets)
            current_file.dropna(subset="retweeted_status.id_str")
            current_file = pd.concat([current_file, day_rtws])
            del day_rtws
            #print(current_file)
            print("Merging dataframes...")
            current_file.update(all_tweets)
            #updated_tweets = pd.merge(all_tweets, current_file, how="left", on=["id_str"])
            #print(current_file)
            #print("^ New dataframe")
            current_file.to_pickle(out_filename)
            del current_file
            print("Wrote to file")
    print("Done")

def handle_retweets(file_frame, all_tweets):
    file_frame["toxicity"] = np.nan
    file_frame["botscore"] = np.nan
    day_retweets = file_frame.dropna(subset="retweeted_status.id_str")
    #print(day_retweets)
    for _, row in day_retweets.iterrows():
        row["toxicity"], row["botscore"] = get_tox_bot(row, all_tweets)
    return day_retweets

def get_tox_bot(row, all_tweets):
    orig = all_tweets.loc[all_tweets["id_str"] == row["id_str"]]
    return orig["toxicity"], orig["botscore"]
    
main()
