import sys
from statistics import fmean
import matplotlib.pyplot as plt
import pandas as pd

def main(args):
    xcol = "toxicity"
    ycol = "botscore"
    data = pd.read_csv("data/2_H_STBM_HASHTAGS.csv", delimiter=";", encoding="utf-8")
    
    data["appearances"] = pd.to_numeric(data["appearances"])
    app_max = data["appearances"].max()
    app_min = data["appearances"].min()
    print(app_max, app_min)
    
    app_med = []
    for _, row in data.iterrows():
        app_med.append(normalise(row["appearances"], app_max, app_min)*1000)
    data["appearances"] = app_med
    print(data["appearances"])

    lula_data = data[data.modularity_class == 7]
    bolso_data = data[data.modularity_class == 9]

    plt.scatter(lula_data[xcol], lula_data[ycol], c="Red", s=lula_data["appearances"], marker=".")
    plt.scatter(bolso_data[xcol], bolso_data[ycol], c="Blue", s=bolso_data["appearances"], marker=".")

    lula_tox = fmean(list(lula_data[xcol]))
    bolso_tox = fmean(list(bolso_data[xcol]))
    lula_bot= fmean(list(lula_data[ycol]))
    bolso_bot = fmean(list(bolso_data[ycol]))

    plt.axvline(x=lula_tox, c="Red", linestyle="dashed")
    plt.axhline(y=lula_bot, c="Red", linestyle="dashed")
    plt.axvline(x=bolso_tox, c="Blue", linestyle="dashed")
    plt.axhline(y=bolso_bot, c="Blue", linestyle="dashed")

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.show()

def normalise(x, max, min):
    numer = x-min
    denom = max-min
    return numer/denom

if __name__ == '__main__':
    main(sys.argv[1:])