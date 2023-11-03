#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np
#
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
#
from pathlib import Path
import seaborn as sns
sns.set_theme(style="darkgrid")

WORKPATH = Path(sys.argv[1])
COLUMNS = ["RANDOM", "TESTMODEL", "TRAINSIZE", "THR", "PICKER", "P_f1", "S_f1"]
OUTDICT = {}

RANDOM_NUMBERS = ["17", "36", "50", "142", "234", "777", "987"]
THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DATASETS = ["INSTANCE", "ETHZ"]
# SIZES = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY", "SMALL", "MEDIUM", "LARGE"]
# SIZES = ["NANO2", "SMALL", "LARGE"]
# PICKERS = ("DKPN", "PN")

DATASETS = [sys.argv[2], ]  # "ETHZ", "PNW", "INSTANCE"
SIZES = [sys.argv[3], ]  # NANO2, MICRO, MEDIUM
PICKERS = ("DKPN", "PN")


def unpickle_me(inpath):
    with open(str(inpath), 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def main():
    for xx, pp in enumerate(WORKPATH.glob("*/*/results*.pickle")):
        # print(rnd_num, modeltest, trainsize, thresh, picker)
        # print(pp)
        fields = str(pp).strip().split(os.sep)
        assert len(fields) == 3
        #
        rnd_num = int(fields[0].split("_")[-1])
        modeltest = fields[1].split("_")[-3]
        trainsize = fields[1].split("_")[-2]
        thresh = fields[1].split("_")[-1]
        picker = fields[2].split("_")[-1].split(".")[0]
        #
        resdict = unpickle_me(pp)
        #
        OUTDICT[str(xx)] = [rnd_num, modeltest, trainsize, thresh, picker,
                            resdict["P_f1"], resdict["S_f1"]]

    df = pd.DataFrame.from_dict(OUTDICT, columns=COLUMNS, orient="index")
    # print(df.head())
    df.to_csv('ALL_DICT.csv', index=False)


    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================

    for _train in DATASETS:
        for _model in SIZES:
            plt_dict = {}
            plt_dict["PN"] = {}
            plt_dict["DKPN"] = {}

            for xx, _picker in enumerate(PICKERS):
                # COLUMNS = ["RANDOM", "TESTMODEL",
                #            "TRAINSIZE", "THR",
                #            "PICKER", "P_f1", "S_f1"]
                _plot_df = df.loc[(df.TESTMODEL == _train) &
                                  (df.TRAINSIZE == _model) &
                                  (df.PICKER == _picker)]
                _plot_df.head()

                Pf1_VALUES_MIN = []
                Pf1_VALUES_MAX = []
                Pf1_VALUES_MEAN = []
                Pf1_VALUES_MEDIAN = []
                Pf1_VALUES_MEDIAN_RND = []

                Sf1_VALUES_MIN = []
                Sf1_VALUES_MAX = []
                Sf1_VALUES_MEAN = []
                Sf1_VALUES_MEDIAN = []
                Sf1_VALUES_MEDIAN_RND = []

                for _thr in THRS:

                    Pf1_VALUES_MIN.append(np.min(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1']))
                    Pf1_VALUES_MAX.append(np.max(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1']))
                    Pf1_VALUES_MEAN.append(np.mean(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1']))
                    if len(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1']) == len(RANDOM_NUMBERS):
                        _median_value = np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1'])
                        _median_idx = _plot_df.loc[_plot_df['P_f1'] == _median_value, 'RANDOM'].values[0]
                    elif len(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1']) == len(RANDOM_NUMBERS)-1:
                        _series = tuple(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1'])
                        _series = pd.Series(_series)
                        _median_value = _series.median()
                        abs_diff = (_series - _median_value).abs()
                        #
                        _median_idx = abs_diff.idxmin()
                        _median_value = _series[_median_idx]
                    else:
                        raise ValueError("NOT ENOUGH DATA")
                    Pf1_VALUES_MEDIAN.append(_median_value)
                    Pf1_VALUES_MEDIAN_RND.append(_median_idx)

                    Sf1_VALUES_MIN.append(np.min(_plot_df.loc[_plot_df.THR == str(_thr)]['S_f1']))
                    Sf1_VALUES_MAX.append(np.max(_plot_df.loc[_plot_df.THR == str(_thr)]['S_f1']))
                    Sf1_VALUES_MEAN.append(np.mean(_plot_df.loc[_plot_df.THR == str(_thr)]['S_f1']))
                    if len(_plot_df.loc[_plot_df.THR == str(_thr)]['S_f1']) == len(RANDOM_NUMBERS):
                        _median_value = np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['S_f1'])
                        _median_idx = _plot_df.loc[_plot_df['S_f1'] == _median_value, 'RANDOM'].values[0]
                    elif len(_plot_df.loc[_plot_df.THR == str(_thr)]['S_f1']) == len(RANDOM_NUMBERS)-1:
                        _series = tuple(_plot_df.loc[_plot_df.THR == str(_thr)]['S_f1'])
                        _series = pd.Series(_series)
                        _median_value = _series.median()
                        abs_diff = (_series - _median_value).abs()
                        #
                        _median_idx = abs_diff.idxmin()
                        _median_value = _series[_median_idx]
                    else:
                        raise ValueError("NOT ENOUGH DATA")
                    Sf1_VALUES_MEDIAN.append(_median_value)
                    Sf1_VALUES_MEDIAN_RND.append(_median_idx)

                #
                plt_dict[_picker] = {}
                plt_dict[_picker]["P_f1"] = {}
                plt_dict[_picker]["P_f1"]["median"] = Pf1_VALUES_MEDIAN
                plt_dict[_picker]["P_f1"]["median_idx"] = Pf1_VALUES_MEDIAN_RND
                plt_dict[_picker]["P_f1"]["mean"] = Pf1_VALUES_MEAN
                plt_dict[_picker]["P_f1"]["min"] = Pf1_VALUES_MIN
                plt_dict[_picker]["P_f1"]["max"] = Pf1_VALUES_MAX

                plt_dict[_picker]["S_f1"] = {}
                plt_dict[_picker]["S_f1"]["median"] = Sf1_VALUES_MEDIAN
                plt_dict[_picker]["S_f1"]["median_idx"] = Sf1_VALUES_MEDIAN_RND
                plt_dict[_picker]["S_f1"]["mean"] = Sf1_VALUES_MEAN
                plt_dict[_picker]["S_f1"]["min"] = Sf1_VALUES_MIN
                plt_dict[_picker]["S_f1"]["max"] = Sf1_VALUES_MAX

    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # axs = axs.flatten()

    # _ax_idx = 0
    colors = {}
    colors["PN"] = "orange"
    colors["DKPN"] = "teal"

    print("%10s - %s" % (_train, _model))
    for _c, _picker in enumerate(PICKERS):
        for _r, _what in enumerate(("P_f1", "S_f1")):
            # ax = axs[_ax_idx]
            ax = axs[_r, _c]
            #
            sns.lineplot(x=THRS, y=plt_dict[_picker][_what]["min"],
                         marker='', linestyle="dashed",
                         ax=ax, color=colors[_picker])

            sns.lineplot(x=THRS, y=plt_dict[_picker][_what]["max"],
                         marker='', linestyle="dashed",  # alpha=0.4,
                         ax=ax, color=colors[_picker])

            sns.lineplot(x=THRS, y=plt_dict[_picker][_what]["mean"],
                         label=("%s_%s_mean" % (_picker, _what)),
                         ax=ax, marker='s', alpha=0.25, color="black")

            sns.lineplot(x=THRS, y=plt_dict[_picker][_what]["median"],
                         label=("%s_%s_median" % (_picker, _what)),
                         ax=ax, marker='s', color=colors[_picker])

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('threshold', fontstyle='italic', fontsize=14)
            ax.set_ylabel("%s" % _what, fontstyle='italic', fontsize=14)
            #
            # _ax_idx += 1
            max_value = max(plt_dict[_picker][_what]["median"])
            max_index = plt_dict[_picker][_what]["median"].index(max_value)
            rnd_exp = plt_dict[_picker][_what]["median_idx"][max_index]
            print("%4s" % _picker, " - ", _what, " ---> ",
                  "%.1f" % (max_index*0.1), " / %.3f / " % max_value, rnd_exp)
        #
    #
    plt.suptitle("%s - %s " % (_train, _model), fontweight='bold', fontsize=18)
    plt.tight_layout()
    #
    fig.savefig("%s_%s_f1.png" % (_train, _model))
    fig.savefig("%s_%s_f1.pdf" % (_train, _model))
    # plt.show()


if __name__ == "__main__":
    main()
