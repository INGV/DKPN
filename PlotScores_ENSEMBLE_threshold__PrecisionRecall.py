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
COLUMNS = ["RANDOM", "TESTMODEL", "TRAINSIZE", "THR", "PICKER",
           "P_precision", "S_precision", "P_recall", "S_recall"]
OUTDICT = {}

RANDOM_NUMBERS = ["17", "36", "50", "142", "234", "777", "987"]
THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DATASETS = ["INSTANCE", "ETHZ"]
# SIZES = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY", "SMALL", "MEDIUM", "LARGE"]
# SIZES = ["NANO2", "SMALL", "LARGE"]
# PICKERS = ("DKPN", "PN")

DATASETS = [sys.argv[2], ]  # INSTANCE, ETHZ, PNW
SIZES = [sys.argv[3], ]  # NANO2, MICRO , MEDIUM
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
                            resdict["P_precision"], resdict["S_precision"],
                            resdict["P_recall"], resdict["S_recall"]]

    df = pd.DataFrame.from_dict(OUTDICT, columns=COLUMNS, orient="index")
    print(df.head())

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

                Pprec_VALUES_MIN = []
                Pprec_VALUES_MAX = []
                Pprec_VALUES_MEAN = []
                Pprec_VALUES_MEDIAN = []

                Sprec_VALUES_MIN = []
                Sprec_VALUES_MAX = []
                Sprec_VALUES_MEAN = []
                Sprec_VALUES_MEDIAN = []

                for _thr in THRS:

                    Pprec_VALUES_MIN.append(np.min(_plot_df.loc[_plot_df.THR == str(_thr)]['P_precision']))
                    Pprec_VALUES_MAX.append(np.max(_plot_df.loc[_plot_df.THR == str(_thr)]['P_precision']))
                    Pprec_VALUES_MEAN.append(np.mean(_plot_df.loc[_plot_df.THR == str(_thr)]['P_precision']))
                    Pprec_VALUES_MEDIAN.append(np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['P_precision']))

                    # _mdn = np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1'])
                    # print(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1'].head())

                    Sprec_VALUES_MIN.append(np.min(_plot_df.loc[_plot_df.THR == str(_thr)]['S_precision']))
                    Sprec_VALUES_MAX.append(np.max(_plot_df.loc[_plot_df.THR == str(_thr)]['S_precision']))
                    Sprec_VALUES_MEAN.append(np.mean(_plot_df.loc[_plot_df.THR == str(_thr)]['S_precision']))
                    Sprec_VALUES_MEDIAN.append(np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['S_precision']))

                #
                plt_dict[_picker] = {}
                plt_dict[_picker]["P_precision"] = {}
                plt_dict[_picker]["P_precision"]["median"] = Pprec_VALUES_MEDIAN
                plt_dict[_picker]["P_precision"]["mean"] = Pprec_VALUES_MEAN
                plt_dict[_picker]["P_precision"]["min"] = Pprec_VALUES_MIN
                plt_dict[_picker]["P_precision"]["max"] = Pprec_VALUES_MAX

                plt_dict[_picker]["S_precision"] = {}
                plt_dict[_picker]["S_precision"]["median"] = Sprec_VALUES_MEDIAN
                plt_dict[_picker]["S_precision"]["mean"] = Sprec_VALUES_MEAN
                plt_dict[_picker]["S_precision"]["min"] = Sprec_VALUES_MIN
                plt_dict[_picker]["S_precision"]["max"] = Sprec_VALUES_MAX

    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================

    for _train in DATASETS:
        for _model in SIZES:
            for xx, _picker in enumerate(PICKERS):
                # COLUMNS = ["RANDOM", "TESTMODEL",
                #            "TRAINSIZE", "THR",
                #            "PICKER", "P_f1", "S_f1"]
                _plot_df = df.loc[(df.TESTMODEL == _train) &
                                  (df.TRAINSIZE == _model) &
                                  (df.PICKER == _picker)]
                _plot_df.head()

                Prec_VALUES_MIN = []
                Prec_VALUES_MAX = []
                Prec_VALUES_MEAN = []
                Prec_VALUES_MEDIAN = []

                Srec_VALUES_MIN = []
                Srec_VALUES_MAX = []
                Srec_VALUES_MEAN = []
                Srec_VALUES_MEDIAN = []

                for _thr in THRS:

                    Prec_VALUES_MIN.append(np.min(_plot_df.loc[_plot_df.THR == str(_thr)]['P_recall']))
                    Prec_VALUES_MAX.append(np.max(_plot_df.loc[_plot_df.THR == str(_thr)]['P_recall']))
                    Prec_VALUES_MEAN.append(np.mean(_plot_df.loc[_plot_df.THR == str(_thr)]['P_recall']))
                    Prec_VALUES_MEDIAN.append(np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['P_recall']))

                    # _mdn = np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1'])
                    # print(_plot_df.loc[_plot_df.THR == str(_thr)]['P_f1'].head())

                    Srec_VALUES_MIN.append(np.min(_plot_df.loc[_plot_df.THR == str(_thr)]['S_recall']))
                    Srec_VALUES_MAX.append(np.max(_plot_df.loc[_plot_df.THR == str(_thr)]['S_recall']))
                    Srec_VALUES_MEAN.append(np.mean(_plot_df.loc[_plot_df.THR == str(_thr)]['S_recall']))
                    Srec_VALUES_MEDIAN.append(np.median(_plot_df.loc[_plot_df.THR == str(_thr)]['S_recall']))

                #
                plt_dict[_picker]["P_recall"] = {}
                plt_dict[_picker]["P_recall"]["median"] = Prec_VALUES_MEDIAN
                plt_dict[_picker]["P_recall"]["mean"] = Prec_VALUES_MEAN
                plt_dict[_picker]["P_recall"]["min"] = Prec_VALUES_MIN
                plt_dict[_picker]["P_recall"]["max"] = Prec_VALUES_MAX

                plt_dict[_picker]["S_recall"] = {}
                plt_dict[_picker]["S_recall"]["median"] = Srec_VALUES_MEDIAN
                plt_dict[_picker]["S_recall"]["mean"] = Srec_VALUES_MEAN
                plt_dict[_picker]["S_recall"]["min"] = Srec_VALUES_MIN
                plt_dict[_picker]["S_recall"]["max"] = Srec_VALUES_MAX

    # =============================================================
    # =============================================================
    # =============================================================
    # =============================================================

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Define colors for cool blue and warm orange
    cool_blue_precision = '#007acc'  # Hexadecimal color code for cool blue
    warm_orange_recall = '#ff7f0e'  # Hexadecimal color code for warm orange

    for _c, _picker in enumerate(PICKERS):
        for _r, (_what1, _what2) in enumerate(
                zip(("P_precision", "S_precision"), ("P_recall", "S_recall"))
            ):
            ax = axs[_r, _c]
            # ---
            sns.lineplot(x=THRS, y=plt_dict[_picker][_what2]["min"],
                         marker='', linestyle="dashed",
                         ax=ax, color=warm_orange_recall)

            sns.lineplot(x=THRS, y=plt_dict[_picker][_what2]["max"],
                         marker='', linestyle="dashed",  # alpha=0.4,
                         ax=ax, color=warm_orange_recall)

            # sns.lineplot(x=THRS, y=plt_dict[_picker][_what2]["mean"],
            #              label=("%s_%s_mean" % (_picker, _what1)),
            #              ax=ax, marker='s', alpha=0.25, color="black")

            sns.lineplot(x=THRS, y=plt_dict[_picker][_what2]["median"],
                         label=("%s_%s_median" % (_picker, _what2)),
                         ax=ax, marker='s', color=warm_orange_recall)

            # ---
            sns.lineplot(x=THRS, y=plt_dict[_picker][_what1]["min"],
                         marker='', linestyle="dashed",
                         ax=ax, color=cool_blue_precision)

            sns.lineplot(x=THRS, y=plt_dict[_picker][_what1]["max"],
                         marker='', linestyle="dashed",  # alpha=0.4,
                         ax=ax, color=cool_blue_precision)

            # sns.lineplot(x=THRS, y=plt_dict[_picker][_what1]["mean"],
            #              label=("%s_%s_mean" % (_picker, _what1)),
            #              ax=ax, marker='s', alpha=0.25, color="black")

            sns.lineplot(x=THRS, y=plt_dict[_picker][_what1]["median"],
                         label=("%s_%s_median" % (_picker, _what1)),
                         ax=ax, marker='s', color=cool_blue_precision)
            #
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('threshold', fontstyle='italic', fontsize=14)
            ax.set_ylabel("score value", fontstyle='italic', fontsize=14)
            #
            # _ax_idx += 1
        #
    #
    plt.suptitle("%s - %s " % (_train, _model), fontweight='bold', fontsize=18)
    plt.tight_layout()
    #
    fig.savefig("%s_%s_PrecisionRecall.png" % (_train, _model))
    fig.savefig("%s_%s_PrecisionRecall.pdf" % (_train, _model))
    # plt.show()


if __name__ == "__main__":
    main()
