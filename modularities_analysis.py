import logging
import sys
import pickle
import pandas as pd
import disslib

# set up logging
logging.basicConfig(filename='logs/modularities.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def main(args):
    modularity_data = pd.read_csv("data/collocations/2_hashtag_modularities_nodes.csv")
    hashtags_ids = pickle.load("data/collocations/2_hashtag_tweetIDs.pkl")
    print(modularity_data)
    print(hashtags_ids)
    # get amount of unique modularities
    modularities = set(range(1, set(modularity_data["Modularity"].unique())+1))
    print(modularities)
        
    # comb through all the json files which also have pkl files,
    # for each day, collect the tweets in each modularity group, then run 
    # toxicity and sentiment analysis on each tweet

    pkl_files, json_files = disslib.get_tweet_files(args[1], pairs_only=True)
    for index, file in enumerate(json_files):
        pass
    
    # recombine that data along with botscores into a csv file to load back into gephi
    

if __name__ == '__main__':
    main(sys.argv[1:])