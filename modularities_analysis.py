import logging
import sys
import pickle
import pandas as pd
import time
import datetime as dt
import disslib
import numpy as np
import nltk
from multiprocessing import Pool, Lock, Manager, set_start_method

# set up logging
logging.basicConfig(filename='logs/modularities.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.WARNING)

def main(args, multiprocessed):
    # open files
    if len(args) == 0:
        start_from = 0
    else:
        start_from = int(args[0])
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
    
    pkl_files = pkl_files[start_from:]
    json_files = json_files[start_from:]
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
    print(f"{str(-1).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, start)} |          | Number of modularities: {num_modularities}")
    print(f"{str(-1).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, start)} |          | Number of tweets to find: {len(tweets_to_process)}")

    # comb through all the json files which also have pkl files for each day,
    # collect the tweets in each modularity group, 
    # run toxicity and sentiment analysis on each collected tweet,
    # then append the results of each process into a collation dataframe
    # the final result will be written out to csv
    if multiprocessed:
        set_start_method("forkserver")
        args_tuples = []
        with Manager() as manager:
            start = time.time()
            lock = manager.Lock()
            for index, json_file in enumerate(json_files):
                pkl_file = pkl_files[index]
                args_tuples.append(
                    (tweets_to_process, 
                    tox_model, 
                    sentilex, 
                    br_stopwords, 
                    pt_core, 
                    total_tweets_checked, 
                    tweets_found, 
                    start,
                    files_to_process, 
                    file_digits, 
                    index, 
                    json_file, 
                    pkl_file, 
                    lock)
                    )
            with Pool(2, initializer=init_pool_processes, initargs=(lock,)) as p:
                p.map(process_files, args_tuples)
                print("All processes complete.")
                p.close()
                p.join()

    else:
        for index, json_file in enumerate(json_files):
            pkl_file = pkl_files[index]
            process_files(
                (tweets_to_process, 
                tox_model, 
                sentilex, 
                br_stopwords, 
                pt_core, 
                total_tweets_checked, 
                tweets_found, 
                start,
                files_to_process, 
                file_digits, 
                index, 
                json_file, 
                pkl_file, 
                lock)
            )
def process_files(arg_tuple):
    (tweets_to_process, tox_model, sentilex, br_stopwords, pt_core, total_tweets_checked, tweets_found, start, files_to_process, file_digits, index, json_file, pkl_file, inner_lock) = arg_tuple
    
    proc_start = time.time()
        
    filedate = json_file.split(".")[0].split("-")[1]
    filename = "data/2_hashtag_stbm/2_hashtag_stbm_" + filedate + ".csv"
        
    disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, proc_start, filedate, f"Now loading file pair: {json_file.split('.')[0]}")

    raw_twts = disslib.load_tweets_json(json_file)
    pkl_data = disslib.load_tweets_pkl(pkl_file)

    raw_twts["id_str"] = pd.to_numeric(raw_twts["id_str"])
    pkl_data["id_str"] = pd.to_numeric(pkl_data["id_str"])

    pkl_data = pkl_data[["id_str", "timestamp_ms", "quoted_status.id_str", "hashtags", "botscore"]]
    pkl_data = pkl_data.reindex(columns=["id_str", "timestamp_ms", "quoted_status.id_str", "hashtags", "botscore", "sentiment", "toxicity"])

    filtered_json_twts = raw_twts[raw_twts["id_str"].isin(tweets_to_process)].copy()
    filtered_pkl_data = pkl_data[pkl_data["id_str"].isin(tweets_to_process)].copy()

        # TODO find a way to drop unnecessary columns if running out of memory becomes an issue
        #filtered_json_twts.drop(['A'], axis=1)
    total_tweets_checked += len(raw_twts)
    data_filtered = time.time()
    if len(filtered_json_twts) == 0:
        disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, data_filtered, filedate, "0 hits in day after load. Skipping sentiment analysis, toxicity analysis, and collation.")
    else:
        tweets_found += len(filtered_json_twts)
        disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, data_filtered, filedate, f"Now loading file pair: {json_file.split('.')[0]}")
            
        texts_to_process = disslib.preprocess_text(list(filtered_json_twts["text"]), br_stopwords, pt_core)
        text_preprocessed = time.time()
        disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, text_preprocessed, filedate, "Texts preprocessed via nltk and spacy")
            
        sentiments_list = disslib.analyse_sentiment(texts_to_process, sentilex, pt_core)
        filtered_pkl_data["sentiment"] = sentiments_list
        sentiment_done = time.time()
            #print(filtered_json_twts)
        disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, sentiment_done, filedate, "Sentiment analysis complete")
            
        predictions = []
            #print(texts_to_process)
        for text in texts_to_process:
            try:
                    #print(text)
                prediction, _ = tox_model.predict(text)
                predictions.append(prediction[0])
            except IndexError:
                logging.warning("Index out of range on text %s", text)
                predictions.append(0)
        filtered_pkl_data["toxicity"] = predictions
        tox_done = time.time()
        disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, tox_done, filedate, "Toxicity analysis complete")

            #print(filtered_json_twts)
            #print(filtered_pkl_data)
            #pd.concat(filtered_pkl_data, filtered_json_twts, how="left", on="id_str")
            #print(filtered_pkl_data)
            
        filtered_pkl_data.to_csv(filename, sep=" ", encoding="utf-8", index=False)
        written_out = time.time()
        disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, written_out, filedate, f"CSV file {filename} written out")
            
        disslib.safe_print(inner_lock, index, file_digits, files_to_process, start, written_out, filedate, f"So far, found {tweets_found}/{len(tweets_to_process)} out of {total_tweets_checked} possible data entries")

def init_pool_processes(the_lock):
    '''Initialize each process with a global variable lock.
    '''
    global lock
    lock = the_lock

if __name__ == '__main__':
    main(sys.argv[1:], multiprocessed=True)