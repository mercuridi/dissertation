"""
Python script to iterate through the data given for the project and collate it all into a collocation matrix.
The collocation matrix is a record of every hashtag and the other hashtags it appears next to. Effectively, we can consider 
it an edge matrix complement to the node matrix of the hashtags themselves; these collocations represent connections we can analyse.

Returns:
    logs/collocations.log
        Logfile for the script. Doesn't contain much, most information is printed to terminal.
    data/collocations/x_hashtag_appearances.csv
        CSV file with all the hashtag nodes, including the number of times they appear.
    data/collocations/x_hashtag_collocations.csv
        CSV file for dataframe of collocated tweets for later processing.
        Effectively an edge matrix.
    data/collocations/x_hashtag_tweetIDs.pkl
        PKL files of every tweet ID we want to process later to get the toxicity/sentiment/whatever
        Basically just a list of everything that contains 2 or more tweets.
"""

import sys
import logging
import itertools
import time
import pickle
import csv
import datetime as dt
import pandas as pd
import disslib

# set up logging
logging.basicConfig(filename="logs/collocations.log",  \
                filemode = "w+",          \
                encoding="utf-8",         \
                level=logging.DEBUG)

def main(args):
    """
    Driver function to call other functions and set up variables used in them.
    Also handles the main loop.

    Args:
        args (list): List of given arguments from the command line.
    """
    # get all the pkl files
    # we use the pkl files to get the tweet IDs because the PKLs are much lighter to load than the JSON files
    # later we can load the JSON files and just request the IDs we already collected
    pkl_files, _ = disslib.get_tweet_files(args[0], pairs_only=True)

    # get the combination size from the command line
    # final analysis only requires 2-size collocations
    # higher numbers imply "stronger" connections
    if len(args) == 2:
        combination_size = int(args[1])
        print(f"Running hashtag collocations with a combination size of {combination_size}.")
    else:
        combination_size = 2
        print(f"Please provide a combination size as a second argument. Defaulting to {combination_size}.")

    # prepare terminal and time logging
    print("")
    print(dt.datetime.now())
    start = time.time()

    # set up some printing variables
    files_to_process = len(pkl_files)-1
    file_digits = len(str(files_to_process))
    print(f"{files_to_process} file pairs to be processed.")

    # initialise recording dicts
    collocation_dict = {}
    hashtag_data = {}

    # single variable to prevent any "possible undefined" later on
    final_i = 0

    # main loop
    # iterates over each file
    for i, pkl_file in enumerate(pkl_files):
        final_i = i

        # record loop start time
        proc_start = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, proc_start)} | Now loading file: {pkl_file}")

        # read pkl file
        base_data = pd.read_pickle(pkl_file)
        pkl_read = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, pkl_read)} | Read file containing {len(base_data)} entries")

        # process data:
        # - drop posts with no hashtags
        # - drop posts which are retweets
        no_nan = base_data[base_data["hashtags"].notna()]
        no_retweets = no_nan[no_nan["retweeted_status.id_str"].isnull()]
        #no_quotes = no_retweets[no_retweets["quoted_status.id_str"].isnull()]
        nans_dropped = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, nans_dropped)} | Dropped {len(base_data)-len(no_retweets)} bad rows, leaving {len(no_retweets)} rows to process.")
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, nans_dropped)} |→ Base data: {len(base_data)}, no NaN: {len(no_nan)}, no retweets: {len(no_retweets)}")

        # update the collocation & hashtag dictionaries
        update_collocations(collocation_dict, hashtag_data, no_retweets, combination_size)
        collocs_updated = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, collocs_updated)} | Collocations updated")

    # print collocations
    # should really be using disslib.niceprint() but I'm not changing that now, the work is finished
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Work finished.")
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Combination size: {combination_size}")
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Total unique collocations: {len(collocation_dict)}")

    # write the collocations CSV out
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Writing edges csv...")
    filename = "data/collocations/" + str(combination_size) + "_hashtag_collocations.csv"
    sorted_collocations = list(sorted(collocation_dict.items(), key=lambda x: x[1], reverse=False))
    with open(filename, "w+", encoding="utf-8") as csv_handle:
        appearances_dict = {}
        edges_writer = csv.writer(csv_handle, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # header row
        edges_writer.writerow(["Source", "Target", "Weight"])
        # writer loop for each row
        for (key, weight) in sorted_collocations:
            #if weight > 1000:
            u, v = tuple(key.split(", "))
            edges_writer.writerow([u, v, weight])
            disslib.increment_occurrence(appearances_dict, u, weight)
            disslib.increment_occurrence(appearances_dict, v, weight)
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Done.")

    # write the nodes CSV out
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Writing nodes csv...")
    filename = "data/collocations/" + str(combination_size) + "_hashtag_appearances.csv"
    with open(filename, "w+", encoding="utf-8") as csv_appearances_handle:
        appearances_writer = csv.writer(csv_appearances_handle, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        appearances_writer.writerow(["Label", "Appearances"])
        for node, appearances in appearances_dict.items():
            appearances_writer.writerow([node, appearances])
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Done.")

    # write the tweet ID pkl out
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Dumping hashtag:IDs dictionary to pkl...")
    filename = "data/collocations/" + str(combination_size) + "_hashtag_tweetIDs.pkl"
    with open(filename, "wb") as pkl_ids_handle:
        pickle.dump(hashtag_data, pkl_ids_handle)
    print(f"{str(final_i).rjust(file_digits)}/{files_to_process} | {disslib.nicetime(start, time.time())} | Done.")

def update_collocations(collocation_dict, hashtag_data, data, combination_size):
    """
    Subroutine to update the collocations and hashtag data

    Args:
        collocation_dict (dict): Dictionary of {collocation:appearances}
        hashtag_data (dict): Dictionary of {hashtag:set([tweet IDs])}
        data (pd.df): Dataframe of the data we want to extract the hashtags from
        combination_size (int): Size of combinations requested

    Returns:
        dict: Updated collocation dictionary
        dict: Updated appearance dictionary
    """
    for _, row in data.iterrows():
        # main loop
        hashtags_list = row["hashtags"]
        id_str = row["id_str"]
        if not isinstance(hashtags_list, list):
            # guard against a nan-dropping error
            continue
        else:
            num_hashtags = len(hashtags_list)
            if num_hashtags < combination_size:
                # skip tweets with less hashtags than the combination size
                continue
            else:
                # sort hashtags to prevent "backwards" combinations
                hashtags_list = sorted(hashtags_list)
                # generate the collocations
                combinations = itertools.combinations(hashtags_list, combination_size)
                for combination in combinations:
                    # add all collocations to the collocation dictionary
                    collocation_dict = disslib.increment_occurrence(collocation_dict, ", ".join(combination))
                for hashtag in hashtags_list:
                    if hashtag in hashtag_data:
                        # add the tweet ID to the hashtag's dictionary
                        current_set = hashtag_data[hashtag]
                        current_set.add(int(id_str))
                        hashtag_data[hashtag] = current_set
                    else:
                        hashtag_data[hashtag] = set([int(id_str)])

    return collocation_dict, hashtag_data

if __name__ == "__main__":
    main(sys.argv[1:])