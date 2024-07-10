import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats
import disslib

def main():
    print("Reading data")
    tweets = pd.read_csv("data/FINAL/2_H_STBM_BYDATES.csv", sep=";")
    tweets["toxicity"]    = pd.to_numeric(tweets["toxicity"])
    tweets["botscore"]    = pd.to_numeric(tweets["botscore"])
    tweets["date"]        = pd.to_datetime(tweets["date"])
    
    # split data by date
    print("Splitting dates")
    data_groups = tweets.groupby("date")
    # convert to list so we can iterate over it
    groups = [data_groups.get_group(x) for x in data_groups.groups]
    day_toxes = []
    day_botscores = []
    dates = []
    print("Entering loop")
    for day in groups:
        means = day.mean(axis=1, numeric_only=True)
        day_toxes.append(day["toxicity"].mean())
        day_botscores.append(day["botscore"].mean())
        dates.append(day.iloc[0][0])
    
    print(dates)
    print(day_botscores)
    print(day_toxes)
    plt.plot(dates, day_botscores, label="Botscores")
    plt.plot(dates, day_toxes,  label="Toxicities")
    plt.xlabel("Date")
    plt.legend()
    plt.show()

def tox_bot_scatter():
    tweets = pd.read_csv("data/2_hashtag_stbm/2_H_STBM_TWEETS.csv", sep=";")
    r, p = stats.pearsonr(tweets["toxicity"], tweets["botscore"])
    print(round(r, 4))
    print(round(p, 4))
    plt.scatter(tweets.toxicity, tweets.botscore)
    plt.plot([], [], label = f"Correlation {round(r, 4)}")
    plt.plot([], [], label = f"P-value {round(p, 4)}")
    plt.xlabel("Toxicity")
    plt.ylabel("Botscore")
    plt.legend()
    plt.show()

main()