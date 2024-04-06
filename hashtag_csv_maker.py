import sys
from statistics import fmean
from collections import Counter
import pandas as pd
import numpy as np


def main(args):
    with open("data/2_H_STBM_TWEETS.csv", "r", encoding="utf-8") as master_handle:
        master_df = pd.read_csv(master_handle, delimiter=";", index_col="id_str")
        master_df["hashtags"] = master_df.hashtags.apply(lambda x: x[1:-1].replace("'", "").replace('"', "").split(', '))
    with open("data/collocations/2_hashtag_modularities_nodes_1000plus.csv", "r", encoding="utf-8") as mod_handle:
        hashtag_nodes = pd.read_csv(mod_handle, delimiter=" ", index_col="ID")
    
    #print(hashtag_nodes)
    valid_hashtags = set(hashtag_nodes.index)

    print("Processing data to hashtag dictionary")
    hashtag_data = {}
    global_min = 100000
    global_max = -100000
    for index, (_, data_row) in enumerate(master_df.iterrows()):
        current_botscore = data_row["botscore"]
        current_sentiment = data_row["sentiment"]
        current_toxicity = data_row["toxicity"]
        if current_sentiment > global_max:
            global_max = current_sentiment
        if current_sentiment < global_min:
            global_min = current_sentiment
        for hashtag in data_row["hashtags"]:
            if hashtag in valid_hashtags:
                update_hashtag_data(
                    hashtag_data, 
                    hashtag, 
                    current_botscore, 
                    current_sentiment, 
                    current_toxicity
                )
        if index % 100000 == 0:
            print(f"{index}/{len(master_df.index)} done")
    print("Finished processing to dictionary, reprocessing values to new dictionary")
    data_dict = {}
    
    for hashtag, values in hashtag_data.items():
        data_dict[hashtag] = [
            average_botscores(values["botscores"]), 
            average_sentiments_B(values["sentiments"]), 
            average_toxicities(values["toxicities"])
        ]
        #print(data_dict)
    print("Finished reprocessing, writing to csv")
    
    final_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['botscore', 'sentiment', 'toxicity'])
    final_df = final_df.merge(hashtag_nodes, how="outer", left_index=True, right_index=True)
    final_df.to_csv("data/2_H_STBM_HASHTAGS.csv", sep=";", encoding="utf-8", index=True, index_label="ID")

def update_hashtag_data(hashtag_data, hashtag, botscore, sentiment, toxicity):
    if hashtag in hashtag_data:
        hashtag_data[hashtag]["botscores"].append(botscore)
        hashtag_data[hashtag]["sentiments"].append(sentiment)
        hashtag_data[hashtag]["toxicities"].append(toxicity)
    else:
        hashtag_data[hashtag] = {
            "botscores": [botscore],
            "sentiments": [sentiment],
            "toxicities": [toxicity]
        }
    #return hashtag_data

def average_botscores(botscores_list):
    # botscores are already normalised so we just take the mean of their values
    botscore_total = 0
    for botscore in botscores_list:
        botscore_total += botscore
    return botscore_total / len(botscores_list)

def average_sentiments_A(sentiments_list, global_min, global_max):
    # method A for performing sentiment averaging
    # sentiments are integers which can be positive or negative, so we use minmax scaling to bring their values between 0 and 1
    # shift values by lowest sentiment globally so we can apply formula correctly
    
    # minimum is negative so we subtract it so the formula works properly
    sentiments_list = [(x-global_min)/(global_max-global_min) for x in sentiments_list]
    # get the mean of the list and return it
    return fmean(sentiments_list)

def average_sentiments_B(sentiments_list):
    neg_total = 0
    pos_total = 0
    zero_total = 0
    # calculate totalled positives and negatives for list
    for sentiment in sentiments_list:
        if sentiment == 0:
            zero_total += 1
        elif sentiment > 0:
            pos_total += sentiment
        elif sentiment < 0:
            neg_total += sentiment
    # calculate ratio of positive:negative
    print(pos_total, neg_total, zero_total)
    if neg_total == 0:
        return pd.NA
    elif pos_total == 0:
        return pd.NA
    elif neg_total == pos_total:
        return pd.NA
    else:
        return pos_total / abs(neg_total)
    

def average_toxicities(toxicities_list):
    # toxicities are boolean values of 0 or 1, so we can just use fmean to average them
    return fmean(toxicities_list)

if __name__ == '__main__':
    main(sys.argv[1:])
