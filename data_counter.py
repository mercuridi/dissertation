"""
i just want to know how many datapoints there actually are
"""

import sys 
import glob
import os
import pandas as pd

def main(args):
    csv_files = glob.glob(os.path.join("data/2_hashtag_stbm/", '*.csv'))
    total_rows = 0
    for file in csv_files:
        with open(file, "r", encoding="utf-8") as csv_handle:
            csv_as_df = pd.read_csv(csv_handle, delimiter=" ")
            total_rows += len(csv_as_df.index)
    
    print(f"Total datapoints: {total_rows}")

if __name__ == '__main__':
    main(sys.argv[1:])