import logging
import sys
import pickle
import disslib

# set up logging
logging.basicConfig(filename='logs/modularities.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def main(args):
    modularities = disslib.load_csv("data/collocations/2_hashtag_modularities_nodes.csv")
    hashtags_ids = pickle.load("")
    print(modularities)
    # get amount of unique modularities
    
    # generate a list of tweet ids for each modularity
    
    # comb through all the json files which also have pkl files,
    # for each day, collect the tweets in each group, then run 
    # toxicity and sentiment analysis on each tweet
    
    # recombine that data along with botscores into a csv file to load back into gephi
    

if __name__ == '__main__':
    main(sys.argv[1:])