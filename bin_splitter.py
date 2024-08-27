import glob
import os
import pandas as pd
import numpy as np
import disslib

def main():
    num_bins = 3
    thresholds = []
    for thres_num in range(1, num_bins):
        thresholds.append(thres_num/num_bins)
    thresholds.append(1)
    all_filenames = glob.glob(os.path.join("data/POSTGRAD/with_retweets/", '*.pkl.tar.gz'))
    num_files = len(all_filenames)
    results = [[] for _ in range(num_bins)]
    for file_index, filename in enumerate(all_filenames):
        print()
        print(f"{file_index}/{num_files}")
        file_frame = pd.read_pickle(filename)
        for i, thres in enumerate(thresholds):
            print(i, round(thres, 2))
            results[i].append(np.mean(file_frame["toxicity"].where(file_frame["botscore"] <= thres)))
            file_frame.drop(file_frame[file_frame.botscore <= thres].index, inplace=True)
        if file_index == 10:
            break
    thresholds.insert(0,0)
    print()
    for i, res_list in enumerate(results):
        print(f"Bin {i} toxicity for botscores between {round(thresholds[i], 2)}-{round(thresholds[i+1], 2)}:")
        print(np.mean(res_list))
        print()

main()