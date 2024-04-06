import logging
import sys
import pickle
import pandas as pd
import time
import datetime as dt
import glob
import os
import disslib
import random
import numpy as np
import nltk
from multiprocessing import Pool, Lock, Manager, set_start_method
from torch import cuda
from waiting import wait

# set up logging
logging.basicConfig(filename='logs/modularities.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.WARNING)

def main(args, multiprocessed):
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
    
    pkl_files.reverse()
    json_files.reverse()
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
    print(cuda.get_device_name(cuda.current_device()))
    start = time.time()

    # get amount of unique modularities and do some printing before main loop
    num_modularities = len(modularity_data["modularity_class"].unique())
    print(f"{str(os.getpid()).rjust(10)} |          | {disslib.nicetime(start, start)} | Number of modularities: {num_modularities}")
    print(f"{str(os.getpid()).rjust(10)} |          | {disslib.nicetime(start, start)} | Number of tweets to find: {len(tweets_to_process)}")

    # comb through all the json files which also have pkl files for each day,
    # collect the tweets in each modularity group, 
    # run toxicity and sentiment analysis on each collected tweet,
    # then append the results of each process into a collation dataframe
    # the final result will be written out to csv
    
    # 2 paths for multiprocessing possibilities
    if multiprocessed:
        set_start_method("forkserver")
        args_tuples = []
        with Manager() as manager:
            start = time.time()
            lock = manager.Lock()
            already_done = glob.glob(os.path.join("data/2_hashtag_stbm", '*.csv'))
            done_set = set()

            for file in already_done:
                done_set.add(file.split(".")[0].split("_")[5])
            


            for index, json_file in enumerate(json_files):
                filedate = json_file.split(".")[0].split("-")[1]
                if filedate not in done_set:
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
                        json_file, 
                        pkl_file, 
                        lock)
                        )
            num_processes = 3
            with Pool(num_processes, initializer=init_pool_processes, initargs=(lock,)) as p:
                p.map(process_files, args_tuples)
                print("All processes complete.")
                p.close()
                p.join()

    else:
        start = time.time()
        already_done = glob.glob(os.path.join("data/2_hashtag_stbm", '*.csv'))
        done_set = set()
        lock = Lock()
        to_process = []
        for file in already_done:
            done_set.add(file.split(".")[0].split("_")[5])
        for file in json_files:
            filedate = file.split(".")[0].split("-")[1]
            if filedate not in done_set:
                #print(file)
                to_process.append(file)
        random.shuffle(to_process)
        print(f"{str(os.getpid()).rjust(10)} |          | {disslib.nicetime(start, start)} | Number of files to process: {len(to_process)}")
        for json_file in to_process:
            pkl_file = json_file.split(".")[0]+".pkl.gz"
            process_files(
                (tweets_to_process, 
                tox_model, 
                sentilex, 
                br_stopwords, 
                pt_core, 
                tweets_found, 
                start,
                json_file, 
                pkl_file, 
                lock)
            )
def process_files(arg_tuple):
    (tweets_to_process, tox_model, sentilex, br_stopwords, pt_core, tweets_found, start, json_file, pkl_file, inner_lock) = arg_tuple
    pid = os.getpid()
    proc_start = time.time()
        
    filedate = json_file.split(".")[0].split("-")[1]
    filename = "data/2_hashtag_stbm/2_hashtag_stbm_" + filedate + ".csv"
    
    disslib.safe_print(inner_lock, pid, start, proc_start, filedate, f"Now loading JSON file: {json_file}")
    disslib.safe_print(inner_lock, pid, start, proc_start, filedate, f"Corresponding PKL: {pkl_file}")

    raw_twts = disslib.load_tweets_json(json_file)
    raw_twts["id_str"] = pd.to_numeric(raw_twts["id_str"])
    filtered_json_twts = raw_twts[raw_twts["id_str"].isin(tweets_to_process)].copy()

    data_filtered = time.time()
    if len(filtered_json_twts) == 0:
        disslib.safe_print(inner_lock, pid, start, data_filtered, filedate, "0 hits in day after load. Skipping sentiment analysis, toxicity analysis, and collation.")
    else:
        tweets_found += len(filtered_json_twts)
        disslib.safe_print(inner_lock, pid, start, data_filtered, filedate, f"{len(filtered_json_twts)} texts found after filtering")
        if len(filtered_json_twts) > 50000:
            disslib.safe_print(inner_lock, pid, start, data_filtered, filedate, "ooh mama that's a big file!")
            return
            
        texts_to_process = disslib.preprocess_text(list(filtered_json_twts["text"]), br_stopwords, pt_core)
        text_preprocessed = time.time()
        disslib.safe_print(inner_lock, pid, start, text_preprocessed, filedate, "Texts preprocessed via nltk and spacy")
            
        sentiments_list = disslib.analyse_sentiment(texts_to_process, sentilex, pt_core)
        filtered_json_twts["sentiment"] = sentiments_list
        sentiment_done = time.time()
            #print(filtered_json_twts)
        disslib.safe_print(inner_lock, pid, start, sentiment_done, filedate, "Sentiment analysis complete")
            
        filtered_json_twts["toxicity"] = run_tox_model(texts_to_process, tox_model, inner_lock, start, filedate)
        tox_done = time.time()
        disslib.safe_print(inner_lock, pid, start, tox_done, filedate, "Toxicity analysis complete; now loading and constructing PKL file to dataframe")
        
        filtered_json_twts.drop(["text"], axis=1)

        pkl_data = disslib.load_tweets_pkl(pkl_file)
        pkl_data["id_str"] = pd.to_numeric(pkl_data["id_str"])
        pkl_data = pkl_data[["id_str", "timestamp_ms", "quoted_status.id_str", "hashtags", "botscore"]]
        pkl_data = pkl_data.reindex(columns=["id_str", "timestamp_ms", "quoted_status.id_str", "hashtags", "botscore", "sentiment", "toxicity"])
        filtered_pkl_data = pkl_data[pkl_data["id_str"].isin(tweets_to_process)].copy()
        filtered_pkl_data["sentiment"] = filtered_json_twts["sentiment"]
        filtered_pkl_data["toxicity"] = filtered_json_twts["toxicity"]
        filtered_pkl_data.to_csv(filename, sep=" ", encoding="utf-8", index=False)
        written_out = time.time()
        disslib.safe_print(inner_lock, pid, start, written_out, filedate, f"CSV file {filename} written out from constructed dataframe")

def init_pool_processes(the_lock):
    global lock
    lock = the_lock

def run_tox_model(texts_to_process, tox_model, model_lock, start_time, filedate):
    predictions = []
    verbose = False
    threshold = 1000
    times_to_print = 20
    times_printed = 0
    if len(texts_to_process) > threshold:
        verbose = True
        modulus = len(texts_to_process) // times_to_print
        started_tox = time.time()
        disslib.safe_print(model_lock, os.getpid(), start_time, started_tox, filedate, f"More than {threshold} texts to process; printing progress status.")
    for text_num, text in enumerate(texts_to_process):
        try:
            prediction, _ = tox_model.predict(text)
            predictions.append(prediction[0])
            if verbose:
                if text_num % modulus == 0:
                    percent_done = time.time()
                    disslib.safe_print(model_lock, os.getpid(), start_time, percent_done, filedate, f"{text_num}/{len(texts_to_process)} complete")
                    disslib.safe_print(model_lock, os.getpid(), start_time, percent_done, filedate, f"  ({times_printed}/{times_to_print})")
                    times_printed += 1

        except IndexError:
            logging.warning("Index out of range on text %s", text)
            predictions.append(0)
    return predictions

if __name__ == '__main__':
    main(sys.argv[1:], multiprocessed=False)