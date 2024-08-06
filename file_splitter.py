import pandas as pd
import numpy as np
import disslib

def main():
    file_to_split = "data/elections2022/elections22/TOOBIG/elections2022_tweets-20221029.json"
    filedate = file_to_split.split(".")[0].split("-")[1]
    print("loading file")
    current_file = disslib.load_tweets_json(file_to_split)
    sz = len(current_file.index) // 2
    print("splitting dataframe 1")
    df1 = current_file.iloc[:, :sz]
    print("writing out df1")
    df1.to_json("data/elections2022/elections22/elections2022_tweets-"+filedate+"PART1.json.gz")
    del df1
    print("splitting dataframe 2")
    df2 = current_file.iloc[:, sz:]
    del current_file
    print("writing out df2")
    df2.to_json("data/elections2022/elections22/elections2022_tweets-"+filedate+"PART2.json.gz")
    del df2
    print("done")

main()