import os
import glob
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

def update_collocations(collocation_dict, hashtag_data, data, combination_size):
    for _, row in data.iterrows():
        #print(row)
        hashtags_list = row["hashtags"]
        #timestamp = row.iloc["timestamp_ms"]
        id_str = row["id_str"]
        if not isinstance(hashtags_list, list):
            # guard against a nan-dropping error
            continue
        else:
            num_hashtags = len(hashtags_list)
            if num_hashtags < combination_size:
                continue
            else:
                hashtags_list = sorted(hashtags_list)
                combinations = itertools.combinations(hashtags_list, combination_size)
                for combination in combinations:
                    collocation_dict = disslib.increment_occurrence(collocation_dict, ", ".join(combination))
                for hashtag in hashtags_list:
                    if hashtag in hashtag_data:
                        current_list = hashtag_data[hashtag]
                        current_list.append(int(id_str))
                        hashtag_data[hashtag] = current_list
                    else:
                        hashtag_data[hashtag] = [int(id_str)]
                    return hashtag_data

    return collocation_dict, hashtag_data

def get_files(path_to_files):
    if path_to_files.endswith("/"):
        pkl_files = glob.glob(os.path.join(path_to_files, "*.pkl.gz"))
        json_files = glob.glob(os.path.join(path_to_files, "*.json.gz"))
        raw_filenames_pkl = [value.split(".")[0] for value in pkl_files]
        raw_filenames_json = [value.split(".")[0] for value in json_files]
        intersection_complement = set(raw_filenames_pkl) ^ set(raw_filenames_json)
        if len(intersection_complement) != 0:
            print("%d files found that have a JSON or PKL file, but not both:" % len(intersection_complement))
            print(sorted(intersection_complement))
            print("It is recommended to manually check these files and figure out what's happening.")
            print("For now, we will remove them from processing...")
            removed = 0
            for file in intersection_complement:
                json_name = file+".json.gz"
                pkl_name = file+".pkl.gz"
                if json_name in json_files:
                    json_files.remove(json_name)
                    removed += 1
                if pkl_name in pkl_files:
                    pkl_files.remove(pkl_name)
                    removed += 1
            print("Removed %d excess files." % removed)
        if len(pkl_files) != len(json_files):
            print("File amounts mismatch after attempted correction.")
            print("PKL files: %d" % len(pkl_files))
            print("JSON files: %d" % len(json_files))
            exit()
    else:
        print("Please provide a directory containing pkl.gz and corresponding json.gz files.")
        exit()

    return sorted(pkl_files)

def nicetime(start_time, end_time, just=True):
    if just:
        return (str(round(end_time-start_time, 3))+"s").rjust(9)
    else:
        return str(round(end_time-start_time, 3))+"s"

def main(args):
    # ORIG. AUTHOR: Diogo Pacheco, modified KBH
    # heavily, heavily modified to run driver code for collocation processing
    path_to_files = args[0]
    pkl_files = get_files(path_to_files)
    
    if len(args) == 2:
        combination_size = int(args[1])
        print(f"Running hashtag collocations with a combination size of {combination_size}.")
    else:
        combination_size = 2
        print(f"Please provide a combination size as a second argument. Defaulting to {combination_size}.")

    print("")
    print(dt.datetime.now())
    start = time.time()

    files_to_process = len(pkl_files)-1
    file_digits = len(str(files_to_process))
    print(f"{files_to_process} file pairs to be processed.")
    
    collocation_dict = {}
    hashtag_data = {}
    
    # the code after this point LOOKS LIKE A MESS
    # but I promise it makes the terminal output really pretty
    # if you look closely it's just the prints that look crazy
    
    for i, pkl_file in enumerate(pkl_files):
        proc_start = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, proc_start)} | Now loading file: {pkl_file}")
        base_data = pd.read_pickle(pkl_file)
        pkl_read = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, pkl_read)} | Read file containing {len(base_data)} entries in {nicetime(proc_start, pkl_read, False)}")
        
        no_nan = base_data[base_data["hashtags"].notna()]
        no_retweets = no_nan[no_nan["retweeted_status.id_str"].isnull()]
        no_quotes = no_retweets[no_retweets["quoted_status.id_str"].isnull()]
        nans_dropped = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, nans_dropped)} | Dropped {len(base_data)-len(no_quotes)} bad rows in {nicetime(pkl_read, nans_dropped, False)}, leaving {len(no_quotes)} rows to process.")
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, nans_dropped)} |→ Base data: {len(base_data)}, no NaN: {len(no_nan)}, no retweets: {len(no_retweets)}, no quotes: {len(no_quotes)}")
        
        update_collocations(collocation_dict, hashtag_data, no_quotes, combination_size)
        collocs_updated = time.time()
        print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, collocs_updated)} | Collocations updated in {nicetime(nans_dropped, collocs_updated, just=False)}")
    
    # print collocations
    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Work finished.") 
    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Combination size: {combination_size}")
    #print("Printing sample of most common collocations...")
    sorted_collocations = [(k, v) for k, v in sorted(collocation_dict.items(), key=lambda x: x[1], reverse=False)]
    #for i in range(1,6):
        #print(f"  {sorted_collocations[-i]}")
    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Total unique collocations: {len(collocation_dict)}")

    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Writing edges csv...")
    filename = "data/collocations/" + str(combination_size) + "_hashtag_collocations.csv"
    with open(filename, "w+", encoding="utf-8") as csv_handle:
        appearances_dict = {}
        edges_writer = csv.writer(csv_handle, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        edges_writer.writerow(["Source", "Target", "Weight"])
        for (key, weight) in sorted_collocations:
            if weight > 1000:
                u, v = tuple(key.split(", "))
                edges_writer.writerow([u, v, weight])
                disslib.increment_occurrence(appearances_dict, u, weight)
                disslib.increment_occurrence(appearances_dict, v, weight)
    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Done.")

    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Writing nodes csv...")
    filename = "data/collocations/" + str(combination_size) + "_hashtag_appearances.csv"
    with open(filename, "w+", encoding="utf-8") as csv_appearances_handle:
        appearances_writer = csv.writer(csv_appearances_handle, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        appearances_writer.writerow(["Label", "Appearances"])
        for node, appearances in appearances_dict.items():
            appearances_writer.writerow([node, appearances])
    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Done.")
    
    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Dumping hashtag:IDs dictionary to pkl...")
    filename = "data/collocations/" + str(combination_size) + "_hashtag_tweetIDs.pkl"
    with open(filename, "wb") as pkl_ids_handle:
        pickle.dump(hashtag_data, pkl_ids_handle)
    print(f"{str(i).rjust(file_digits)}/{files_to_process} | {nicetime(start, time.time())} | Done.")

if __name__ == "__main__":
    # AUTHOR: Diogo Pacheco
    main(sys.argv[1:])