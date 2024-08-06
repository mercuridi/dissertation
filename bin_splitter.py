import glob
import os
import pandas as pd
import numpy as np
import disslib

def main():
    num_bins = 3
    thresholds = []
    for thres_denom in range(1, num_bins):
        thresholds.append(1/thres_denom)
    all_filenames = glob.glob(os.path.join("data/POSTGRAD/with_retweets/", '*.pkl.tar.gz'))
    for filename in all_filenames:
        file_frame = pd.read_pickle(filename)
        