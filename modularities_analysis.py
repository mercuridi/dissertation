import logging
import sys
import pickle
import pandas as pd
import time
import datetime as dt
import disslib
import numpy as np
import nltk
import stanza

# set up logging
logging.basicConfig(filename='logs/modularities.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.CRITICAL)

def main(args):
    # open files
    modularity_data = pd.read_csv("data/collocations/2_hashtag_modularities_nodes_1000plus.csv", encoding="utf-8", sep=" ", header=0, names=["ID", "appearances", "modularity"])
    with open("data/collocations/2_hashtag_tweetIDs.pkl", "rb") as hashtags_ids_pickle:
        hashtags_tweetids = pickle.load(hashtags_ids_pickle)
    
    #print(modularity_data)
    #print(hashtags_tweetids)

    # convert modularity column to int
    modularity_data["modularity_class"] = pd.to_numeric(modularity_data["modularity"])
    
    # filter the hashtags we want to find down to only the ones which are in a certain set of modularity groups
    # 7: lula-focused
    # 9: bolsonaro-focused
    # change numbers in this list to adjust which groups we find
    modularities_to_find = [7, 9]
    filtered_data = modularity_data.loc[modularity_data["modularity"].isin(set(modularities_to_find))]
    
    # using the hashtags in each group, find the tweet IDs that contain those hashtags
    nodes_999_names = set(filtered_data["ID"])
    tweets_to_process = set()
    for key, tweetids in hashtags_tweetids.items():
        if key in nodes_999_names:
            for str_id in tweetids:
                tweets_to_process.add(np.int64(str_id))

    # get the pkl and json files
    pkl_files, json_files = disslib.get_tweet_files(dir_path="data/elections2022/", pairs_only=True)
    
    #pkl_files.reverse()
    #json_files.reverse()
    # set up sentiment and toxicity engines
    tox_model = disslib.load_toxicity_model()
    sentilex = disslib.load_sentilex()
    
    # set up text preprocessing
    br_stopwords = nltk.corpus.stopwords.words('portuguese')
    pt_core = disslib.load_pt_core()

    # setting up tracking variables for main loop
    total_tweets_checked = 0
    tweets_found = 0
    print("")
    print(dt.datetime.now())
    start = time.time()
    files_to_process = len(pkl_files)-1
    file_digits = len(str(files_to_process))

    # get amount of unique modularities and do some printing before main loop
    num_modularities = len(modularity_data["modularity_class"].unique())
    print(f"{str(-1).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, start)} | Number of modularities: {num_modularities}")
    print(f"{str(-1).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, start)} | Number of tweets to find: {len(tweets_to_process)}")

    # comb through all the json files which also have pkl files for each day,
    # collect the tweets in each modularity group, 
    # run toxicity and sentiment analysis on each collected tweet,
    # then append the results of each process into a collation dataframe
    # the final result will be written out to csv
    for index, json_file in enumerate(json_files):
        proc_start = time.time()
        
        print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, proc_start)} | Now loading file pair: {json_file.split('.')[0]}")
        pkl_file = pkl_files[index]
        raw_twts = disslib.load_tweets_json(json_file)
        pkl_data = disslib.load_tweets_pkl(pkl_file)
        raw_twts["id_str"] = pd.to_numeric(raw_twts["id_str"])
        filtered_json_twts = raw_twts[raw_twts["id_str"].isin(tweets_to_process)].copy()
        filtered_pkl_data = pkl_data[pkl_data["id_str"].isin(tweets_to_process)].copy()
        # TODO find a way to drop unnecessary columns if running out of memory becomes an issue
        #filtered_json_twts.drop(['A'], axis=1)
        total_tweets_checked += len(raw_twts)
        data_filtered = time.time()
        if len(filtered_json_twts) > 0:
            tweets_found += len(filtered_json_twts)
            collation_frame = pd.DataFrame(columns=["id_str", "timestamp_ms", "quoted_status.id_str", "hashtags", "botscore", "sentiment", "toxicity", "modularity"])
            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, data_filtered)} | Data loaded & filtered, {len(filtered_json_twts)} tweets to process")
            texts_to_process = disslib.preprocess_text(list(filtered_json_twts["text"]), br_stopwords, pt_core)
            text_preprocessed = time.time()
            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, text_preprocessed)} | Text preprocessed.")
            filtered_pkl_data["sentiment"] = disslib.analyse_sentiment(texts_to_process, sentilex, pt_core)
            sentiment_done = time.time()
            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, sentiment_done)} | Sentiment analysis complete")
            _, outputs = tox_model.predict(texts_to_process)
            filtered_pkl_data["toxicity"] = outputs
            tox_done = time.time()
            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, tox_done)} | Toxicity analysis complete")
            
            pd.merge(collation_frame, filtered_pkl_data, how="left")
            pd.merge(collation_frame, modularity_data, how="left")
            collation_done = time.time()
            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, collation_done)} | Data collation complete")
            
            filedate = pkl_file.split(".")[0].split("-")[1]
            filename = "data/2_hashtag_stbm_/2_hashtag_stbm_" + filedate + ".csv"
            collation_frame.to_csv(filename, sep=" ", encoding="utf-8", index=False)
            written_out = time.time()
            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, written_out)} | csv file {filename} written out")
            

            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, written_out)} | So far, found {tweets_found}/{len(tweets_to_process)} out of {total_tweets_checked} possible data entries")

        else:
            print(f"{str(index).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, data_filtered)} | 0 hits in day after load. Skipping sentiment analysis, toxicity analysis, and collation.")

if __name__ == '__main__':
    main(sys.argv[1:])