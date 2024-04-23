"""
Script to generate a CSV file of all hashtags by date. Each hashtag can only appear once in any date, but can appear in multiple dates.
"""
import datetime
import pandas as pd
import disslib

def main():
    """
    Script to generate a CSV file of all hashtags by date. Each hashtag can only appear once in any date, but can appear in multiple dates.
    """
    # load files
    with open("data/2_hashtag_stbm/2_H_STBM_TWEETS.csv", "r", encoding="utf-8") as master_handle:
        master_df = pd.read_csv(master_handle, delimiter=";", index_col="id_str")
        master_df["hashtags"] = master_df.hashtags.apply(lambda x: x[1:-1].replace("'", "").replace('"', "").split(', '))
    with open("data/collocations/2_hashtag_modularities_nodes_1000plus.csv", "r", encoding="utf-8") as mod_handle:
        hashtag_nodes = pd.read_csv(mod_handle, delimiter=" ")

    # get valid hashtags
    valid_hashtags = set(hashtag_nodes["ID"])
    date_data = {}
    for index, (_, data_row) in enumerate(master_df.iterrows()):
        # main loop
        # get the variables for the current post being processed
        current_botscore = data_row["botscore"]
        current_toxicity = data_row["toxicity"]
        current_date = datetime.datetime.fromtimestamp(data_row["timestamp_ms"]/1000).date()
        for hashtag in data_row["hashtags"]:
            # for each hashtag in the post
            if hashtag in valid_hashtags:
                # if we want the hashtag, record its data
                if current_date in date_data:
                    if hashtag in date_data[current_date]:
                        date_data[current_date][hashtag]["toxicities"].append(current_toxicity)
                        date_data[current_date][hashtag]["botscores"].append(current_botscore)
                        date_data[current_date][hashtag]["appearances"] += 1
                    else:
                        date_data[current_date][hashtag] = { \
                            "toxicities": [current_toxicity], \
                            "botscores": [current_botscore], \
                            "appearances": 1
                        }
                else:
                    date_data[current_date] = {hashtag:{ \
                        "toxicities": [current_toxicity], \
                        "botscores": [current_botscore], \
                        "appearances": 1
                    }}
        # "progress bar"
        if index % 100000 == 0:
            print(f"{index}/{len(master_df.index)} done")

    # flatten dictionary into a 2D list so it loads directly to a dataframe for saving
    print("Finished processing to dictionary, reprocessing values to new dictionary")
    df_doublelist = []
    for date, hashtag_dict in date_data.items():
        for hashtag, attribute_list in hashtag_dict.items():
            df_doublelist.append([
                date,
                hashtag,
                list(hashtag_nodes.loc[hashtag_nodes["ID"] == hashtag]["modularity_class"])[0],
                disslib.average_botscores(attribute_list["botscores"]),
                disslib.average_toxicities(attribute_list["toxicities"]),
                attribute_list["appearances"]
            ])

    # load nested list to dataframe and write it out, sorted by date
    final_df = pd.DataFrame(df_doublelist, columns = ["date", "hashtag", "modularity", "botscore", "toxicity", "appearances"])
    final_df = final_df.sort_values("date")
    print("Finished reprocessing, writing to csv")
    final_df.to_csv("data/FINAL/2_H_STBM_BYDATES.csv", sep=";", encoding="utf-8", index=False)

if __name__ == '__main__':
    main()
