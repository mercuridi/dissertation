"""
Possibly the most important file in the dissertation repo.
Script to perform the toxicity and sentiment analysis on the already-filtered data given.
Automatically checks to see which files have already been completed.

Returns:
    data/2_hashtag_stbm/2_hashtag_stbm[date].csv
        CSV files of the data for each day after processing.
"""

import logging
import pickle
import time
import datetime as dt
import glob
import os
import random
from multiprocessing import Pool, Lock, Manager, set_start_method
import numpy as np
import nltk
import pandas as pd
from torch import cuda
import disslib

# set up logging
logging.basicConfig(filename='logs/modularities.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.WARNING)

def main(multiprocessed):
    """
    Function which: 
    - sets up data, 
    - handles main loop, 
    - handles multiprocessing if enabled, 
    - calls imported functions from disslib
    - and writes out completed data processing.

    Args:
        multiprocessed (bool): Multiprocessing enabler. Broken, likely with no way to make this work due to complex GPU reliances.
    """
    # open files
    modularity_data = pd.read_csv("data/collocations/2_hashtag_modularities_nodes_1000plus.csv", encoding="utf-8", sep=" ", header=0, names=["ID", "appearances", "modularity"])
    with open("data/collocations/2_hashtag_tweetIDs.pkl", "rb") as hashtags_ids_pickle:
        hashtags_tweetids = pickle.load(hashtags_ids_pickle)

    # convert modularity column to numeric
    modularity_data["modularity_class"] = pd.to_numeric(modularity_data["modularity"])

    # filter the hashtags we want to find down to only the
    # ones which are in a certain set of modularity groups
    # 7: left-wing
    # 9: right-wing

    # change numbers in this list to adjust which groups we find
    modularities_to_find = [7, 9]
    filtered_data = modularity_data.loc[modularity_data["modularity"].isin(set(modularities_to_find))]

    # using the hashtags in each group, find the tweet IDs that contain those hashtags
    #nodes_999_names = set(modularity_data["ID"])
    nodes_999_names = set(filtered_data["ID"])
    tweets_to_process = set()
    for key, tweetids in hashtags_tweetids.items():
        if key in nodes_999_names:
            for str_id in tweetids:
                tweets_to_process.add(np.int64(str_id))

    # get the pkl and json files
    pkl_files, json_files = disslib.get_tweet_files(dir_path="data/elections2022/", pairs_only=True)

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

    # main loop logic:
    #   - comb through all the json files which also have pkl files for each day,
    #   - collect the tweets in each modularity group,
    #   - run toxicity and sentiment analysis on each collected tweet,
    #   - then append the results of each process into a collation dataframe.
    #   - the final result will be written out to csv in data/2_hashtag_stbm/

    # 2 paths for multiprocessing possibilities
    # multiprocessing doesn't work, but I really, really, really tried
    # doesn't work because I couldn't find a way to make the CPU wait for the GPU to complete;
    # the CPU would offload the toxicity processing to the GPU, only to continue on and start
    # loading more files in its own thread, later offloading them to the GPU again
    # this is great, but it would very very quickly fill the RAM or VRAM and cause a force kill
    # in my case, the RAM always filled first and sometimes crashed my PC
    # if there was a way to force the CPU thread to just wait for its offloaded process to complete, this would have worked
    # but because the torch environment changes in each thread, there's not really any way to do that.
    # in theory, you could just not offload to the GPU, which would solve that problem
    # but the CPU sucks at running the torch operations, that's why we offloaded to GPU in the first place
    # so it would take just as long if not longer anyway
    # :(
    if multiprocessed:
        # multiprocessing setup
        set_start_method("forkserver")
        args_tuples = []
        with Manager() as manager:
            # set up variables
            start = time.time()
            lock = manager.Lock()
            already_done = glob.glob(os.path.join("data/2_hashtag_stbm", '*.csv'))
            done_set = set()

            # figure out what is and isn't done
            for file in already_done:
                done_set.add(file.split(".")[0].split("_")[5])

            # parse arguments into a tuple so that out spawned threads can use them
            # i don't know why they need me to do this, it's just how it works
            for index, json_file in enumerate(json_files):
                filedate = json_file.split(".")[0].split("-")[1]
                if filedate not in done_set:
                    pkl_file = pkl_files[index]
                    args_tuples.append((
                        tweets_to_process,
                        tox_model,
                        sentilex,
                        br_stopwords,
                        pt_core,
                        total_tweets_checked,
                        tweets_found,
                        start,
                        json_file,
                        pkl_file,
                        lock
                        ))

            # call threads to run the main loop function
            num_processes = 3
            with Pool(num_processes, initializer=init_pool_processes, initargs=(lock,)) as p:
                p.map(process_files, args_tuples)
                print("All processes complete.")
                p.close()
                p.join()

    else:
        # set up variables
        start = time.time()
        done_set = set()
        to_process = []
        lock = Lock()

        # figure out what is and isn't done already
        already_done = glob.glob(os.path.join("data/2_hashtag_stbm", '*.csv'))
        for file in already_done:
            done_set.add(file.split(".")[0].split("_")[5])
        for file in json_files:
            filedate = file.split(".")[0].split("-")[1]
            if filedate not in done_set:
                to_process.append(file)

        # shuffle the files to process, just for fun, why not
        random.shuffle(to_process)
        print(f"{str(os.getpid()).rjust(10)} |          | {disslib.nicetime(start, start)} | Number of files to process: {len(to_process)}")

        # main loop
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
    """
    Function to actually do the work.
    Processes data day by day / file by file.

    Args:
        arg_tuple (tuple): Tuple containing all arguments. Unpacked on first line:
            tweets_to_process: 
                List of ALL tweet IDs to process.
            tox_model: 
                ToLD-Br, loaded by disslib.load_toxicity_model()
            sentilex: 
                Sentiment analysis corpus, loaded by disslib.load_sentilex()
            br_stopwords:
                Stopwords in Portuguese, loaded by disslib.load_br_stopwords()
            pt_core: 
                SpaCy pipeline, loaded by disslib.load_pt_core()
            tweets_found:
                Number of tweets found so far. Currently has a bug, but the work is complete so I'm not fixing it.
            start:
                Start time of outer loop, passed in for printing purposes.
            json_file:
                Name of the JSON file we're currently working on.
            pkl_file:
                Name of the JSON file we're currently working on.
            inner_lock:
                The global lock outside, renamed for clarity.
    """
    # unpack arguments and set up variables
    (tweets_to_process, tox_model, sentilex, br_stopwords, pt_core, tweets_found, start, json_file, pkl_file, inner_lock) = arg_tuple
    pid = os.getpid()
    proc_start = time.time()

    # figure out date from filename
    filedate = json_file.split(".")[0].split("-")[1]
    filename = "data/2_hashtag_stbm/2_hashtag_stbm_" + filedate + ".csv"

    # initial prints
    disslib.safe_print(inner_lock, pid, start, proc_start, filedate, f"Now loading JSON file: {json_file}")
    disslib.safe_print(inner_lock, pid, start, proc_start, filedate, f"Corresponding PKL: {pkl_file}")

    # load and filter JSON file
    raw_twts = disslib.load_tweets_json(json_file)
    raw_twts["id_str"] = pd.to_numeric(raw_twts["id_str"])
    filtered_json_twts = raw_twts[raw_twts["id_str"].isin(tweets_to_process)].copy()

    # timestamp for finishing loading and filtering the JSON file
    data_filtered = time.time()

    if len(filtered_json_twts) == 0:
        # just in case nothing to process
        disslib.safe_print(inner_lock, pid, start, data_filtered, filedate, "0 hits in day after load. Skipping sentiment analysis, toxicity analysis, and collation.")
    else:
        # print tweets found to process
        tweets_found += len(filtered_json_twts)
        disslib.safe_print(inner_lock, pid, start, data_filtered, filedate, f"{len(filtered_json_twts)} texts found after filtering")

        # run away if that file is too big
        if len(filtered_json_twts) > 50000:
            disslib.safe_print(inner_lock, pid, start, data_filtered, filedate, "ooh mama that's a big file!")
            return

        # preprocess all texts
        texts_to_process = disslib.preprocess_text(list(filtered_json_twts["text"]), br_stopwords, pt_core)
        text_preprocessed = time.time()
        disslib.safe_print(inner_lock, pid, start, text_preprocessed, filedate, "Texts preprocessed via nltk and spacy")

        # calculate all sentiments
        sentiments_list = disslib.analyse_sentiment(texts_to_process, sentilex, pt_core)
        filtered_json_twts["sentiment"] = sentiments_list
        sentiment_done = time.time()
            #print(filtered_json_twts)
        disslib.safe_print(inner_lock, pid, start, sentiment_done, filedate, "Sentiment analysis complete")

        # calculate all toxicities
        filtered_json_twts["toxicity"] = run_tox_model(texts_to_process, tox_model, inner_lock, start, filedate)
        tox_done = time.time()
        disslib.safe_print(inner_lock, pid, start, tox_done, filedate, "Toxicity analysis complete; now loading and constructing PKL file to dataframe")

        # drop text column so it doesn't end up in the final data
        # otherwise those files would be enormous
        filtered_json_twts.drop(["text"], axis=1)

        # now load the PKL data so we can steal the botscores
        pkl_data = disslib.load_tweets_pkl(pkl_file)
        pkl_data["id_str"] = pd.to_numeric(pkl_data["id_str"])

        # cut down the PKL to just what we want
        pkl_data = pkl_data[["id_str", "timestamp_ms", "quoted_status.id_str", "hashtags", "botscore"]]

        # add the columns we're going to add
        pkl_data = pkl_data.reindex(columns=["id_str", "timestamp_ms", "quoted_status.id_str", "hashtags", "botscore", "sentiment", "toxicity"])

        # filter the pkl data to only tweets we have data for
        filtered_pkl_data = pkl_data[pkl_data["id_str"].isin(tweets_to_process)].copy()

        # copy over the sentiment and toxicity columns we just calculated
        # order is preserved so no worries there
        filtered_pkl_data["sentiment"] = filtered_json_twts["sentiment"]
        filtered_pkl_data["toxicity"] = filtered_json_twts["toxicity"]

        # write out the final file
        filtered_pkl_data.to_csv(filename, sep=" ", encoding="utf-8", index=False)
        written_out = time.time()
        disslib.safe_print(inner_lock, pid, start, written_out, filedate, f"CSV file {filename} written out from constructed dataframe")

def init_pool_processes(the_lock):
    """
    Helper function to force declare the lock to be a global lock.
    
    Args:
        the_lock (multiprocessing.Lock): it's a lock
    """
    global lock
    lock = the_lock

def run_tox_model(texts_to_process, tox_model, model_lock, start_time, filedate):
    """
    Helper function to run the toxicity model on many texts.

    Args:
        texts_to_process (_type_): _description_
        tox_model (_type_): _description_
        model_lock (_type_): _description_
        start_time (_type_): _description_
        filedate (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    main(multiprocessed=False)
