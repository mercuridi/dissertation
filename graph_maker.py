import sys
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import pandas as pd
from collections import Counter
from scipy.ndimage.filters import gaussian_filter1d

OTHER_COLOUR    = "darkgoldenrod"
BATTLE_COLOUR   = "green"
LULA_COLOUR     = "royalblue"
BOLSO_COLOUR    = "firebrick"
NEWS_COLOUR     = "orange"

XCOL = "toxicity"
YCOL = "botscore"

SMOOTHING = 3
NUM_BLOCKS = 50

def main(args):
    # handle data nicely
    data = pd.read_csv("data/FINAL/2_H_STBM_HASHTAGS.csv", delimiter=";", encoding="utf-8")
    
    # cast columns
    #data.dropna(subset=["botscore", "toxicity", "appearances"])
    data = data.dropna()
    data["appearances"] = pd.to_numeric(data["appearances"])
    data["toxicity"] = pd.to_numeric(data["toxicity"])
    data["botscore"] = pd.to_numeric(data["botscore"])

    # normalise appearances to be drawn on scatters
    app_max = data["appearances"].max()
    app_min = data["appearances"].min()
    print(app_max, app_min)
    app_med = []
    for _, row in data.iterrows():
        app_med.append(normalise(row["appearances"], app_max, app_min)*1000)
    data["app_normalised"] = app_med

    # normalise y column
    app_sum = data["appearances"].sum()
    y_normalised = []
    for _, row in data.iterrows():
        y_normalised.append(
            (row[YCOL] * row["appearances"])/app_sum
        )
    data["y_normalised"] = y_normalised
    
    mods = [1, 7, 9, 14]
    #  1: battleground
    #  7: left wing
    #  9: right wing
    # 14: news

    #get_stats(data[~data.modularity_class.isin(mods)])

    draw_comms = False
    if draw_comms is True:
        draw_community(data[data.modularity_class == 1], BATTLE_COLOUR, "Battleground")
        draw_community(data[data.modularity_class == 7], LULA_COLOUR, "Left-wing")
        draw_community(data[data.modularity_class == 9], BOLSO_COLOUR, "Right-wing")
        draw_community(data[data.modularity_class == 14], NEWS_COLOUR, "News")
        draw_community(data[~data.modularity_class.isin(mods)], OTHER_COLOUR, "All other")
    #draw_community(data[data.modularity_class == 9], BOLSO_COLOUR, "Right-wing")
    #all_communities(data)
    #draw_scatter(data)
    #determine_relationship(data)
    #relation_graph(data, toxmed, botmed)
    #threshold_graph(data)
    #dev_apps(data)
    normaliser = False
    if normaliser is True:
        normalised = brute_force_normaliser(data)
        all_communities(normalised)
    print(np.std(data["botscore"]))
    print(np.std(data["toxicity"]))

def dev_apps(data):
    x, y, freqs = block_graph(data, 25, 1)
    plt.plot(x, y, label="Deviation")
    plt.plot(x, freqs, label="Block appearances")
    plt.legend()
    plt.xlabel(XCOL.title())
    plt.show()

def brute_force_normaliser(data):
    # worst normalisation technique of all time, this is bad code and i know it
    # do not do this
    dataframe_as_list = []
    print("Please wait, this might take a minute...")
    for _, data_row in data.iterrows():
        for _ in range(0, int(data_row["appearances"])):
            dataframe_as_list.append(list(data_row))
    normalised = pd.DataFrame(dataframe_as_list)
    normalised = normalised.rename(columns={0:"ID", 1:"botscore", 2:"sentiment", 3:"toxicity", 4:"appearances", 5:"modularity_class", 6:"app_normalised", 7:"y_normalised"})
    return normalised

def all_communities(data):
    block_mode = 1
    battle_x, battle_y, _ = block_graph(data[data.modularity_class == 1], NUM_BLOCKS, block_mode)
    lula_x, lula_y, _ = block_graph(data[data.modularity_class == 7], NUM_BLOCKS, block_mode)
    bolso_x, bolso_y, _ = block_graph(data[data.modularity_class == 9], NUM_BLOCKS, block_mode)
    news_x, news_y, _ = block_graph(data[data.modularity_class == 14], NUM_BLOCKS, block_mode)
    other_x, other_y, _ = block_graph(data[~data.modularity_class.isin([1, 7, 9, 14])], NUM_BLOCKS, block_mode)

    plt.plot(battle_x, gaussian_filter1d(battle_y, sigma=SMOOTHING), c=BATTLE_COLOUR, label="Battleground hashtags mean botscore")
    plt.plot(lula_x, gaussian_filter1d(lula_y, sigma=SMOOTHING), c=LULA_COLOUR, label="Left wing hashtags mean botscore")
    plt.plot(bolso_x, gaussian_filter1d(bolso_y, sigma=SMOOTHING), c=BOLSO_COLOUR, label="Right wing hashtags mean botscore")
    plt.plot(news_x, gaussian_filter1d(news_y, sigma=SMOOTHING), c=NEWS_COLOUR, label="News hashtags mean botscore")
    plt.plot(other_x, gaussian_filter1d(other_y, sigma=SMOOTHING), c=OTHER_COLOUR, label="All other hashtags mean botscore")
    
    if block_mode == 1:
        plt.xlabel(XCOL.title())
        plt.ylabel(YCOL.title() + " standard deviation")
    else:
        ax = plt.gca()
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([0.0038796620046618463, 0.7985270979021011])
        plt.xlabel(XCOL.title())
        plt.ylabel(YCOL.title())
    plt.legend(handles=[        
        mpatches.Patch(color=LULA_COLOUR, label="Left-wing hashtags"),
        mpatches.Patch(color=BOLSO_COLOUR, label="Right-wing hashtags"),
        mpatches.Patch(color=BATTLE_COLOUR, label="Battleground hashtags"),
        mpatches.Patch(color=NEWS_COLOUR, label="News hashtags"),
        mpatches.Patch(color=OTHER_COLOUR, label="All other hashtags"),
        ])
    plt.title("Standard deviation of major communities")
    plt.show()
    
def draw_community(community_data, argcolour, communityname):
    ax = plt.gca()
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([0.0038796620046618463, 0.7985270979021011])
    
    arglabel = communityname + " hashtags"
    argtitle = communityname + " hashtags: Toxicity and Botscore"
    
    community_data = community_data.sort_values(by=["toxicity"])
    community_data = community_data.reset_index(drop=True)

    x_line, y_line, _ = block_graph(community_data, NUM_BLOCKS, 0)
    sizes = community_data["app_normalised"]
    plt.scatter(community_data[XCOL], community_data[YCOL], c=argcolour, s=sizes, marker=".", alpha=0.9)
    plt.plot(x_line, gaussian_filter1d(y_line, sigma=SMOOTHING), c="black")
    plt.legend(handles=[
        mpatches.Patch(color=argcolour, label=arglabel),
        mpatches.Patch(color="grey", label="Mean line"),
    ]) 
    plt.xlabel(XCOL.title())
    plt.ylabel(YCOL.title())
    plt.title(argtitle)
    plt.show()

def block_graph(data, blocks=25, mode=0):
    data = data.sort_values(by=[XCOL])
    data = data.reset_index(drop=True)

    frames = np.array_split(data, blocks+1)
    yvals = []
    xvals = []
    freqs = []
    for frame in frames:
        if mode == 0:
            # mode 0 is mean
            # DO NOT pass data that has been through brute_force_normaliser()
            yvals.append(np.ma.average(frame[YCOL], weights=frame["appearances"]))
        elif mode == 1:
            # mode 1 is standard deviation
            # ONLY pass data that has been through brute_force_normaliser()
            yvals.append(np.std(frame["botscore"]))
        xvals.append(np.ma.average(frame[XCOL]))
    
    return xvals, yvals, freqs

def generate_parabola():
    numpoints = 100
    x = np.linspace(-50,50,numpoints)
    y = (x**2)/10000

    xpoints = np.linspace(0, 1, numpoints)
    return xpoints, y

def draw_scatter(data):
    # graphing calls from here down
    ax = plt.gca()
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([0.0038796620046618463, 0.7985270979021011])
    style = "dashed"
    local_alpha = 0.85
    median_lines = False

    other_data = data[~data.modularity_class.isin([1, 7, 9])]
    other_bot = np.nanmean(list(other_data[YCOL]))
    other_tox = np.nanmean(list(other_data[XCOL]))
    other_sizes = other_data["appearances"]
    print(other_tox, other_bot)
    plt.scatter(other_data[XCOL], other_data[YCOL], c=OTHER_COLOUR, s=other_sizes, marker=".", alpha=local_alpha, label="All other posts")

    battle_data = data[data.modularity_class == 1]
    battle_bot = np.nanmean(list(battle_data[YCOL]))
    battle_tox = np.nanmean(list(battle_data[XCOL]))
    battle_sizes = battle_data["appearances"]
    print(battle_tox, battle_bot)
    plt.scatter(battle_data[XCOL], battle_data[YCOL], c=BATTLE_COLOUR, s=battle_sizes, marker=".", alpha=local_alpha, label="Battleground posts")
    
    lula_data = data[data.modularity_class == 7]
    lula_bot = np.nanmean(list(lula_data[YCOL]))
    lula_tox = np.nanmean(list(lula_data[XCOL]))
    lula_sizes = lula_data["appearances"]
    print(lula_tox, lula_bot)
    plt.scatter(lula_data[XCOL], lula_data[YCOL], c=LULA_COLOUR, s=lula_sizes, marker=".", alpha=local_alpha, label="Left-wing posts")
    
    bolso_data = data[data.modularity_class == 9]
    bolso_bot = np.nanmean(list(bolso_data[YCOL]))
    bolso_tox = np.nanmean(list(bolso_data[XCOL]))
    bolso_sizes = bolso_data["appearances"]
    print(bolso_tox, bolso_bot)
    plt.scatter(bolso_data[XCOL], bolso_data[YCOL], c=BOLSO_COLOUR, s=bolso_sizes, marker=".", alpha=local_alpha, label="Right-wing posts")
    
    block_x, block_y = block_graph(data, 50)
    plt.plot(block_x, gaussian_filter1d(block_y, sigma=1.5), c="grey", label="Average botscore", alpha=local_alpha)

    
    xpara, ypara = generate_parabola()
    plt.plot(xpara, ((-ypara)+0.35)*0.4, c="grey", alpha = 0.2)
    plt.plot(xpara, ((ypara)+1.05)*0.4, c="grey", alpha = 0.2)
    
    if median_lines:
        plt.axvline(x=battle_tox, c=BATTLE_COLOUR, linestyle=style)
        plt.axhline(y=battle_bot, c=BATTLE_COLOUR, linestyle=style)
        plt.axvline(x=lula_tox, c=LULA_COLOUR, linestyle=style)
        plt.axhline(y=lula_bot, c=LULA_COLOUR, linestyle=style)
        plt.axvline(x=bolso_tox, c=BOLSO_COLOUR, linestyle=style)
        plt.axhline(y=bolso_bot, c=BOLSO_COLOUR, linestyle=style)

    plt.xlabel(XCOL.title())
    plt.ylabel(YCOL.title())
    plt.title("Toxicity, botscore, and modularity")
    plt.legend(handles=[
        mpatches.Patch(color=LULA_COLOUR, label="Left-wing hashtags"),
        mpatches.Patch(color=BOLSO_COLOUR, label="Right-wing hashtags"),
        mpatches.Patch(color=BATTLE_COLOUR, label="Battleground hashtags"),
        mpatches.Patch(color=OTHER_COLOUR, label="All other hashtags"),
        mpatches.Patch(color="grey", label="Mean botscore line"),
    ]) 
    plt.show()

def normalise(x, max, min):
    numer = x-min
    denom = max-min
    return numer/denom

def normalise_list(lst):
    lstmax = max(lst)
    lstmin = min(lst)
    return [normalise(x, lstmax, lstmin) for x in lst]

def relation_graph(data, toxmed, botmed):
    data = data.sort_values(by=[XCOL])
    plt.scatter(data[XCOL], data[YCOL], marker=".", c="darkgreen", alpha=0.5)
    #plt.plot(data["toxicity"], gaussian_filter1d(data["botscore"], sigma=10), label="Smoothed", c="pink")
    plt.xlabel(XCOL.title())
    plt.ylabel(YCOL.title())
    #plt.axhline(y=toxmed, linestyle="dashed", label="Network median toxicity", c="lightblue")
    #plt.axvline(x=botmed, linestyle="dashed", label="Network median botscore", c="lightgreen")
    #plt.legend()
    plt.show()

def determine_relationship(data):
    print("FILTERED FOR BOTSCORE")
    print(len(data.index))
    data = data.drop(data[data.botscore < 0.5].index)
    print(len(data.index))
    print(Counter(data["modularity_class"]))

    print("Mode toxicity: "     + str(statistics.mode(data["toxicity"])))
    print("Mean toxicity: "     + str(np.nanmean(data["toxicity"])))
    print("Median toxicity: "   + str(np.nanmedian(data["toxicity"])))
    print("Mode botscore: "     + str(statistics.mode(data["botscore"])))
    print("Mean botscore: "     + str(np.nanmean(data["botscore"])))
    print("Median botscore: "   + str(np.nanmedian(data["botscore"])))
    print()

def threshold_graph(data):
    means      = []
    medians    = []
    index_lens = []
    med_bot = np.nanmedian(data["botscore"])
    med_tox = np.nanmedian(data["toxicity"])

    # 0: botscore on x axis
    # 1: toxicity on x axis
    mode = 0
    if mode == 0:
        x_values = list(np.linspace(data["botscore"].min(), data["botscore"].max(), 1000))
        for x in x_values:
            index_lens.append(len(data.index))
            data = data.drop(data[data.botscore < x].index)
            means.append(np.nanmean(data["toxicity"]))
            medians.append(np.nanmedian(data["toxicity"]))

        plt.xlabel("Botscore threshold")
        plt.ylabel("Network toxicity")
        plt.axhline(y=med_tox, linestyle="dashed", label="Network median toxicity", c="blue")
        plt.axvline(x=med_bot, linestyle="dashed", label="Network median botscore", c="purple")

    elif mode == 1:
        x_values = list(np.linspace(data["toxicity"].min(), data["toxicity"].max(), 1000))
        for x in x_values:
            index_lens.append(len(data.index))
            data = data.drop(data[data.toxicity < x].index)
            means.append(np.nanmean(data["botscore"]))
            medians.append(np.nanmedian(data["botscore"]))

        plt.xlabel("Toxicity threshold")
        plt.ylabel("Network botscore")
        plt.axhline(y=med_bot, linestyle="dashed", label="Network median botscore", c="blue")
        plt.axvline(x=med_tox, linestyle="dashed", label="Network median toxicity", c="purple")

    smoothed = True
    if smoothed:
        plt.plot(x_values, gaussian_filter1d(medians, sigma=SMOOTHING), label="Median", c="green")
        plt.plot(x_values, gaussian_filter1d(means, sigma=SMOOTHING), label="Mean", c="red")
    else:
        plt.plot(x_values, medians, label="Median", c="green")
        plt.plot(x_values, means, label="Mean", c="red")

    #plt.plot(x_values, index_lens, label="Index lengths")

    plt.legend()
    plt.show()

def get_stats(data):
    print("RAW DATA")
    #print("Mode toxicity: "     + str(statistics.mode(data["toxicity"])))
    print("Mean toxicity: "     + str(np.average(data["toxicity"], weights=data["appearances"])))
    print("Median toxicity: "   + str(np.nanmedian(data["toxicity"])))
    #print("Mode botscore: "     + str(statistics.mode(data["botscore"])))
    print("Mean botscore: "     + str(np.average(data["botscore"], weights=data["appearances"])))
    print("Median botscore: "   + str(np.nanmedian(data["botscore"])))
    print()
    return \
        np.nanmean(data["toxicity"]), \
        np.nanmedian(data["toxicity"]), \
        np.nanmean(data["botscore"]), \
        np.nanmedian(data["botscore"])

if __name__ == '__main__':
    main(sys.argv[1:])