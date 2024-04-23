"""
Script to generate a CSV file of all hashtags and their relevant data from raw tweet data.
"""
from statistics import fmean
import pandas as pd
import disslib

def main():
    """
    Driver function to load data and call functions.
    """
    with open("data/2_H_STBM_TWEETS.csv", "r", encoding="utf-8") as master_handle:
        master_df = pd.read_csv(master_handle, delimiter=";", index_col="id_str")
        master_df["hashtags"] = master_df.hashtags.apply(lambda x: x[1:-1].replace("'", "").replace('"', "").split(', '))
    with open("data/collocations/2_hashtag_modularities_nodes_1000plus.csv", "r", encoding="utf-8") as mod_handle:
        hashtag_nodes = pd.read_csv(mod_handle, delimiter=" ", index_col="ID")

    valid_hashtags = set(hashtag_nodes.index)

    data_dict = process_to_csv(master_df, valid_hashtags)
    write_hashtags(hashtag_nodes, data_dict)

def write_hashtags(hashtag_nodes, data_dict):
    """
    Wrapper function to create the final dataframe and write out the result.

    Args:
        hashtag_nodes (pd.df): Dictionary of data to be merged into data_dict's dataframe
        data_dict (dict): Dictionary to be converted to a dataframe
    """
    final_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['botscore', 'sentiment', 'toxicity'])
    final_df = final_df.merge(hashtag_nodes, how="outer", left_index=True, right_index=True)
    final_df.to_csv("data/2_H_STBM_HASHTAGS.csv", sep=";", encoding="utf-8", index=True, index_label="ID")

def process_to_csv(master_df, valid_hashtags):
    """
    Function to build a dictionar which will be written out as a csv from a dataframe.

    Args:
        master_df (pd.df): All data to be processed.
        valid_hashtags (pd.df): Hashtags we want to process.

    Returns:
        dict: Near-final dictionary to be converted to a dataframe
    """
    print("Processing data to hashtag dictionary")

    # loop variables
    hashtag_data = {}
    global_min = 100000
    global_max = -100000
    for index, (_, data_row) in enumerate(master_df.iterrows()):
        # get data from current row
        current_botscore = data_row["botscore"]
        current_sentiment = data_row["sentiment"]
        current_toxicity = data_row["toxicity"]

        # determine max and min for sentiment
        if current_sentiment > global_max:
            global_max = current_sentiment
        if current_sentiment < global_min:
            global_min = current_sentiment

        # process all the hashtags in each row
        for hashtag in data_row["hashtags"]:
            if hashtag in valid_hashtags:
                # update the data if we like the hashtag
                update_hashtag_data(
                    hashtag_data,
                    hashtag,
                    current_botscore,
                    current_sentiment,
                    current_toxicity
                )

        # "progress bar"
        if index % 100000 == 0:
            print(f"{index}/{len(master_df.index)} done")

    print("Finished processing to dictionary, reprocessing values to new dictionary")
    data_dict = {}

    # reprocess data to averages for each hashtag
    for hashtag, values in hashtag_data.items():
        data_dict[hashtag] = [
            disslib.average_botscores(values["botscores"]),
            average_sentiments_b(values["sentiments"]),
            disslib.average_toxicities(values["toxicities"])
        ]
    print("Finished reprocessing, writing to csv")
    return data_dict

def update_hashtag_data(hashtag_data, hashtag, botscore, sentiment, toxicity):
    """
    Helper function to handle the hashtag data dictionary.

    Args:
        hashtag_data (dict): Nested dict of all hashtags
        hashtag (string): Hashtag
        botscore (float): Botscore of hashtag
        sentiment (numeric): Sentiment of hashtag
        toxicity (float): Toxicity of hashtag
    """
    # safely add the data to the dictionary
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

    return hashtag_data

def average_sentiments_a(sentiments_list, global_min, global_max):
    """
    Method A for performing sentiment averaging after a normalisation
    Sentiments are integers which can be positive or negative, so we use minmax scaling to bring their values between 0 and 1
    Didn't work.

    Args:
        sentiments_list (list): List of sentiments to average
        global_min (int): Minimum sentiment value in list
        global_max (int): Maximum sentiment value in list

    Returns:
        float: mean of all sentiments
    """
    # minimum is negative so we subtract it so the formula works properly
    sentiments_list = [(x-global_min)/(global_max-global_min) for x in sentiments_list]
    # get the mean of the list and return it
    return fmean(sentiments_list)

def average_sentiments_b(sentiments_list):
    """
    Alternate method for averaging out sentiments.
    Didn't work either.

    Args:
        sentiments_list (list): List of sentiments to average

    Returns:
        float: mean of all sentiments
    """
    # set up loop variables
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

if __name__ == '__main__':
    main()
