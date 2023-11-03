#!/usr/bin/env python

import os
import sys
from pathlib import Path
import pickle
from pprint import pprint

from collections import defaultdict
from numpy import median, mean

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# =================================================
# =================================================
# =================================================

METHOD = "median"  # "mean"

RANDOM_NUMBERS = ["17", "36", "50", "142", "234", "777", "987"]
DATASETS = [("INSTANCE", "INSTANCE"), ("INSTANCE", "ETHZ")]
SIZES = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY", "SMALL", "MEDIUM", "LARGE"]
THRVAL = ["02", "05"]

#BASEFOLDER_START = "./FinalRetrain_DKPN_PN_PAPER___Rnd__"
#BASEFOLDER_END = "__TESTING-ULTIMATE_noDetrend"
BASEFOLDER_START = "Rnd_"
BASEFOLDER_END = ""

# =================================================
# =================================================
# =================================================


def __create_dict_mean_min_max__(dict_list, method="mean"):
    support_dict = defaultdict(list)

    for dictionary in dict_list:
        for key, value in dictionary.items():
            support_dict[key].append(value)

    if method.lower() in ("median", "med", "md"):
        mean_values = {key: median(values) for key, values in support_dict.items()}
    elif method.lower() in ("mean", "mn"):
        mean_values = {key: mean(values) for key, values in support_dict.items()}
    else:
        raise ValueError("Erroneous METHOD specified! Either 'mean' or 'median'!")
    min_values = {key: min(values) for key, values in support_dict.items()}
    max_values = {key: max(values) for key, values in support_dict.items()}

    return (mean_values, min_values, max_values)


def __make_figure__(DKPN_SCORES, PN_SCORES, TRAINNAME, TESTNAME, THRVAL,
                    store_file="DE.pdf", X_LABELS=SIZES, method="mean"):

    color_list = ["darkgoldenrod", "magenta", "blue"]

    def extract_dict_values(indict, what_list, order=["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY"]):
        value_lists = []
        for key in what_list:
            value_list = [indict[item][key] for item in order]
            value_lists.append(value_list)

        return value_lists

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # ===============================================================
    # ========================================  AX-1 --> scores P
    ax = axs[0, 0]

    # --- Get VALS
    key_list = ["P_FN", "P_FP", "P_TP"]

    VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=X_LABELS)
    VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=X_LABELS)

    for i, values in enumerate(VALUES_PN):
        sns.lineplot(x=X_LABELS, y=values, marker='o', linestyle="dashed",
                     label="PN_"+key_list[i], ax=ax, color=color_list[i])

    for i, values in enumerate(VALUES_DKPN):
        sns.lineplot(x=X_LABELS, y=values, marker='s', label="DKPN_"+key_list[i],
                     ax=ax, color=color_list[i])

    # --- Decorator
    ax.set_ylim([0, 5000])
    ax.set_xlabel('train size', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_ylabel('count', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_title('P-COUNT', fontstyle='italic', fontsize=16, fontname="Lato")   # Set the title of the plot
    ax.legend()  # Display the legend

    # ===============================================================
    # ========================================  AX-3 --> Scores P
    ax = axs[1, 0]

    # --- Get VALS
    key_list = ["P_recall", "P_precision", "P_f1"]

    VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=X_LABELS)
    VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=X_LABELS)

    for i, values in enumerate(VALUES_PN):
        sns.lineplot(x=X_LABELS, y=values, marker='o', linestyle="dashed",
                     label="PN_"+key_list[i], ax=ax, color=color_list[i])

    for i, values in enumerate(VALUES_DKPN):
        sns.lineplot(x=X_LABELS, y=values, marker='s', label="DKPN_"+key_list[i],
                     ax=ax, color=color_list[i])

    # --- Decorator
    # ax.set_ylim([0, 1])
    ax.set_ylim([0.7, 1])
    ax.set_xlabel('train size', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_ylabel('value', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_title('P-SCORES', fontstyle='italic', fontsize=16, fontname="Lato")   # Set the title of the plot
    ax.legend()  # Display the legend

    # ===============================================================
    # ========================================  AX-2 --> count S
    ax = axs[0, 1]

    # --- Get VALS
    key_list = ["S_FN", "S_FP", "S_TP"]

    VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=X_LABELS)
    VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=X_LABELS)

    for i, values in enumerate(VALUES_PN):
        sns.lineplot(x=X_LABELS, y=values, marker='o', linestyle="dashed",
                     label="PN_"+key_list[i], ax=ax, color=color_list[i])

    for i, values in enumerate(VALUES_DKPN):
        sns.lineplot(x=X_LABELS, y=values, marker='s', label="DKPN_"+key_list[i],
                     ax=ax, color=color_list[i])

    # --- Decorator
    ax.set_ylim([0, 5000])
    ax.set_xlabel('train size', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_ylabel('count', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_title('S-COUNT', fontstyle='italic', fontsize=16, fontname="Lato")   # Set the title of the plot
    ax.legend()  # Display the legend

    # ===============================================================
    # ========================================  AX-4 --> Count S
    ax = axs[1, 1]

    # --- Get VALS
    key_list = ["S_recall", "S_precision", "S_f1"]

    VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=X_LABELS)
    VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=X_LABELS)

    for i, values in enumerate(VALUES_PN):
        sns.lineplot(x=X_LABELS, y=values, marker='o', linestyle="dashed",
                     label="PN_"+key_list[i], ax=ax, color=color_list[i])

    for i, values in enumerate(VALUES_DKPN):
        sns.lineplot(x=X_LABELS, y=values, marker='s', label="DKPN_"+key_list[i],
                     ax=ax, color=color_list[i])

    # --- Decorator
    # ax.set_ylim([0, 1])
    ax.set_ylim([0.3, 1])
    ax.set_xlabel('train size', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_ylabel('value', fontstyle='italic', fontsize=14, fontname="Lato")
    ax.set_title('S-SCORES', fontstyle='italic', fontsize=16, fontname="Lato")   # Set the title of the plot
    ax.legend()  # Display the legend

    # ===============================================================
    plt.suptitle("%s Training - %s Testing - THR: %.1f - Method: %s" % (TRAINNAME, TESTNAME, float(THRVAL)*0.1, method.upper()),
                 fontweight='bold', fontsize=18, fontname="Lato")  #fontfamily='sans-serif')
    plt.tight_layout()  # Adjust the spacing between subplots
    fig.savefig(store_file)


if __name__ == "__main__":

    for (_ds1, _ds2) in DATASETS:  # InDomain CrossDomain

        DICT4FIGURES_DKPN_MEAN, DICT4FIGURES_DKPN_MIN, DICT4FIGURES_DKPN_MAX = {}, {}, {}
        DICT4FIGURES_PN_MEAN, DICT4FIGURES_PN_MIN, DICT4FIGURES_PN_MAX = {}, {}, {}

        for _thr in THRVAL:
            for _sz in SIZES:

                dict_list_DKPN, dict_list_PN = [], []

                for _rndm in RANDOM_NUMBERS:
                    _work_path = Path(BASEFOLDER_START+_rndm+BASEFOLDER_END) / ("Results_"+_ds1+"_"+_ds2+"_"+_sz+"_"+_thr)

                    pickle_DKPN = str(_work_path / "results_DKPN.pickle")
                    with open(pickle_DKPN, 'rb') as file:
                        loaded_data = pickle.load(file)
                    dict_list_DKPN.append(loaded_data)

                    pickle_PN = str(_work_path / "results_PN.pickle")
                    with open(pickle_PN, 'rb') as file:
                        loaded_data = pickle.load(file)
                    dict_list_PN.append(loaded_data)

                # --> End RANDOM
                assert len(dict_list_DKPN) == len(dict_list_PN) == len(RANDOM_NUMBERS)
                (_mean_dict_dkpn, _min_dict_dkpn, _max_dict_dkpn) = __create_dict_mean_min_max__(dict_list_DKPN, method=METHOD)
                (_mean_dict_pn, _min_dict_pn, _max_dict_pn) = __create_dict_mean_min_max__(dict_list_PN,  method=METHOD)
                #
                DICT4FIGURES_DKPN_MEAN[_sz] = _mean_dict_dkpn
                DICT4FIGURES_DKPN_MIN[_sz] = _min_dict_dkpn
                DICT4FIGURES_DKPN_MAX[_sz] = _max_dict_dkpn
                #
                DICT4FIGURES_PN_MEAN[_sz] = _mean_dict_pn
                DICT4FIGURES_PN_MIN[_sz] = _min_dict_pn
                DICT4FIGURES_PN_MAX[_sz] = _max_dict_pn

            # --> End SIZES
            __make_figure__(DICT4FIGURES_DKPN_MEAN, DICT4FIGURES_PN_MEAN, _ds1, _ds2, _thr, method=METHOD,
                            store_file=("%s_%s_%s_%s_scores.pdf" % (_ds1, _ds2, _thr, METHOD.lower())))

        # --> End THR

    # --> End DATASETs
