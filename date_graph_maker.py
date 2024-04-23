"""
Script for generating date-based graphs.
Some logic borrowed from hashtag_graph_maker.py

Returns:
    different graphs depending on which function you run
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import disslib

# set up global variables
OTHER_COLOUR    = "darkgoldenrod"
BATTLE_COLOUR   = "green"
LULA_COLOUR     = "royalblue"
BOLSO_COLOUR    = "firebrick"
NEWS_COLOUR     = "orange"

XCOL = "date"
YCOL = "botscore"
YNORM = "y_normalised"
COLOURCOL = "toxicity"
SIZECOL = "app_normalised"

LINEWIDTH = 2
SMOOTHING = 1
NUM_BLOCKS = 50

TOX_THRES = -0.5

def main():
    """
    Main routine for script. Collects data, sets it up nicely, and calls graphing functions.
    """
    # handle data nicely
    data = pd.read_csv("data/FINAL/2_H_STBM_BYDATES.csv", delimiter=";", encoding="utf-8")
    print(data)

    # cast columns
    data = data.dropna()
    data["appearances"] = pd.to_numeric(data["appearances"])
    data["toxicity"]    = pd.to_numeric(data["toxicity"])
    data["botscore"]    = pd.to_numeric(data["botscore"])
    data["date"]        = pd.to_datetime(data["date"])

    # filter to below TOX_THRES
    # only does anything if TOX_THRES > 0
    # not used in final report analysis
    data = data.loc[data["toxicity"] > TOX_THRES]

    # normalise appearances to be drawn on scatters
    app_max = data["appearances"].max()
    app_min = data["appearances"].min()
    print(app_max, app_min)
    app_med = []
    for _, row in data.iterrows():
        app_med.append(disslib.normalise(row["appearances"], app_max, app_min)*1000)
    data["app_normalised"] = app_med

    # function calls
    draw_scatter(data)
    #draw_appearances(data)

def draw_appearances(data):
    """
    Function to draw a graph of mean appearances in community by date.

    Args:
        data (pd.df): Full dataset to draw the graph from.
    """
    # local vars
    # change if you want
    local_alpha = 0.8
    local_linewidth = 2

    # get data we care about
    lula_data   = data[data.modularity_class == 7]
    bolso_data  = data[data.modularity_class == 9]
    battle_data = data[data.modularity_class == 1]
    other_data  = data[~data.modularity_class.isin([1, 7, 9])]

    # computationally wasteful but whatever, only run once to make the graph
    lula_x,     _, lula_y   = mean_calculator(lula_data,    XCOL, "appearances", "appearances")
    bolso_x,    _, bolso_y  = mean_calculator(bolso_data,   XCOL, "appearances", "appearances")
    battle_x,   _, battle_y = mean_calculator(battle_data,  XCOL, "appearances", "appearances")
    other_x,    _, other_y  = mean_calculator(other_data,   XCOL, "appearances", "appearances")
    all_x,      _, all_y    = mean_calculator(data,         XCOL, "appearances", "appearances")

    # plot lines
    plt.plot(lula_x,    gaussian_filter1d(lula_y, sigma=SMOOTHING),     c=LULA_COLOUR,      linewidth=local_linewidth, alpha=local_alpha, label="Left-wing hashtag usage by day")
    plt.plot(bolso_x,   gaussian_filter1d(bolso_y, sigma=SMOOTHING),    c=BOLSO_COLOUR,     linewidth=local_linewidth, alpha=local_alpha, label="Right-wing hashtag usage by day")
    plt.plot(battle_x,  gaussian_filter1d(battle_y, sigma=SMOOTHING),   c=BATTLE_COLOUR,    linewidth=local_linewidth, alpha=local_alpha, label="Battleground hashtag usage by day")
    plt.plot(other_x,   gaussian_filter1d(other_y, sigma=SMOOTHING),    c=OTHER_COLOUR,     linewidth=local_linewidth, alpha=local_alpha, label="All other hashtag usage by day")
    plt.plot(all_x,     gaussian_filter1d(all_y, sigma=SMOOTHING),      c="black",          linewidth=local_linewidth, alpha=local_alpha, label="Cumulative total hashtags used")

    # set titles and stuff
    plt.title("Hashtage usage per community by date")
    plt.xlabel("Date")
    plt.ylabel("Hashtags used")
    plt.legend()
    plt.show()


def draw_scatter(data):
    """
    Function to draw a graph of hashtags used in each community by date.
    Draws colours from COLOURCOL if comm_colour is True, sizes from SIZECOL

    Args:
        data (pd.df): Full dataset to draw the graph from.
    """
    # adapted from hashtag_graph_maker.py
    # local variables
    local_alpha = 0.8
    mean_colour = "black"
    local_cmap = matplotlib.cm.get_cmap("plasma")

    # define subplots
    fig, axs = plt.subplots(4, 1)

    # partition data
    lula_data   = data[data.modularity_class == 7]
    bolso_data  = data[data.modularity_class == 9]
    battle_data = data[data.modularity_class == 1]
    other_data  = data[~data.modularity_class.isin([1, 7, 9])]

    # calculate mean data for each community
    lula_means_x,   lula_means_y,   _   = mean_calculator(lula_data,    XCOL, YCOL, "appearances")
    bolso_means_x,  bolso_means_y,  _   = mean_calculator(bolso_data,   XCOL, YCOL, "appearances")
    battle_means_x, battle_means_y, _   = mean_calculator(battle_data,  XCOL, YCOL, "appearances")
    other_means_x,  other_means_y,  _   = mean_calculator(other_data,   XCOL, YCOL, "appearances")

    # set colours depending on whether they should be based on community
    # or a colourmap against COLOURCOL
    comm_colour = False
    if comm_colour:
        lul_c = LULA_COLOUR
        bol_c = BOLSO_COLOUR
        bat_c = BATTLE_COLOUR
        oth_c = OTHER_COLOUR
    else:
        lul_c = lula_data[COLOURCOL]
        bol_c = bolso_data[COLOURCOL]
        bat_c = battle_data[COLOURCOL]
        oth_c = other_data[COLOURCOL]

    # variable to change title of subplot graphs easily
    title =  " hashtags by date"

    # subplot plotting calls
    axs[0].scatter(lula_data[XCOL], lula_data[YCOL], c=lul_c, s=lula_data[SIZECOL], marker=".", alpha=local_alpha, cmap=local_cmap)
    axs[0].plot(lula_means_x, gaussian_filter1d(lula_means_y, sigma=SMOOTHING), c=mean_colour, alpha=local_alpha, linewidth=LINEWIDTH, label="Weighted mean botscore")
    axs[0].set_title("Left-wing"+title)

    axs[1].scatter(bolso_data[XCOL], bolso_data[YCOL], c=bol_c, s=bolso_data[SIZECOL], marker=".", alpha=local_alpha, cmap=local_cmap)
    axs[1].plot(bolso_means_x, gaussian_filter1d(bolso_means_y, sigma=SMOOTHING), c=mean_colour, alpha=local_alpha, linewidth=LINEWIDTH, label="Weighted mean botscore")
    axs[1].set_title("Right-wing"+title)

    axs[2].scatter(battle_data[XCOL], battle_data[YCOL], c=bat_c, s=battle_data[SIZECOL], marker=".", alpha=local_alpha, cmap=local_cmap)
    axs[2].plot(battle_means_x, gaussian_filter1d(battle_means_y, sigma=SMOOTHING), c=mean_colour, alpha=local_alpha, linewidth=LINEWIDTH, label="Weighted mean botscore")
    axs[2].set_title("Battleground"+title)

    axs[3].scatter(other_data[XCOL], other_data[YCOL], c=oth_c, s=other_data[SIZECOL], marker=".", alpha=local_alpha, cmap=local_cmap)
    axs[3].plot(other_means_x, gaussian_filter1d(other_means_y, sigma=SMOOTHING), c=mean_colour, alpha=local_alpha, linewidth=LINEWIDTH, label="Weighted mean botscore")
    axs[3].set_title("All other"+title)

    # set up stuff that goes on every axis
    for index, ax in enumerate(axs.flat):
        if index == 0:
            ylims = ax.get_ylim()
        else:
            ax.set_ylim(ylims)
        ax.set(xlabel=XCOL.title(), ylabel=YCOL.title())
        ax.legend(loc="upper left")

    # if colouring by colourmap, draw a colour bar under the graphs
    if not comm_colour:
        fig.subplots_adjust(bottom=0.15, top=0.95)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=local_cmap), cax=cbar_ax, label="Hashtag Toxicity", orientation="horizontal")

    # show the plots
    plt.show()

def mean_calculator(data, x_col, y_col, weight_col):
    """
    subroutine for calculating means and frequencies for each date

    Args:
        data (pd.df): all the data for the graph being plotted
        x_col (string): column being plotted on x axis
        y_col (string): column being plotted on y axis
        weight_col (string): column being used for average weighting

    Returns:
        list: list of x values
        list: list of means (y values)
        list: list of frequencies (y values)
    """
    # split data by date
    data_groups = data.groupby(x_col)
    # convert to list so we can iterate over it
    groups = [data_groups.get_group(x) for x in data_groups.groups]

    # loop variables
    weighted_means = []
    dates = []
    appearances = []
    for group in groups:
        # build loop variables
        weighted_means.append(np.average(group[y_col], weights=group[weight_col]))
        dates.append(group.iloc[0][x_col])
        appearances.append(group["appearances"].sum())

        # print for closer investigation of specific dates
        check_date = "2022-10-30"
        if group.iloc[0][x_col] == pd.Timestamp(ts_input=check_date):
            pd.set_option('display.max_rows', 1000)
            group = group[group.modularity_class == 7]
            print(group.sort_values("botscore", ascending=False))

    return dates, weighted_means, appearances

if __name__ == '__main__':
    main()
