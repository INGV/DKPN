#!/usr/bin/env python

import os
import sys
from pathlib import Path
import pickle
from pprint import pprint

from collections import defaultdict
from numpy import median, mean, std, arange

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# =================================================
# =================================================
# =================================================

# ./PlotResiduals_NEW.py  DATASET  SIZE  RANDOM_NUMBER_DKPN_P  THRVAL_DKPN_P RANDOM_NUMBER_DKPN_S  THRVAL_DKPN_S  RANDOM_NUMBER_PN_P  THRVAL_PN_P  RANDOM_NUMBER_PN_S  THRVAL_PN_S


DATASET = sys.argv[1]   # ETHZ
SIZE = sys.argv[2]     # NANO2

RANDOM_NUMBER_DKPN_P = sys.argv[3]
THRVAL_DKPN_P = sys.argv[4]
RANDOM_NUMBER_DKPN_S = sys.argv[5]
THRVAL_DKPN_S = sys.argv[6]


RANDOM_NUMBER_PN_P = sys.argv[7]
THRVAL_PN_P = sys.argv[8]
RANDOM_NUMBER_PN_S = sys.argv[9]
THRVAL_PN_S = sys.argv[10]

# =================================================
# =================================================
# =================================================


def __extract_pickle__(inpath):
    with open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def create_residuals_plot_compare(resP_dkpn, resS_dkpn,
                                  resP_pn, resS_pn,
                                  add_title_p, add_title_s,
                                  axs, binwidth=0.025):

    bin_edges = arange(-11.05, 11.05 + binwidth, binwidth)

    print("BINWIDTH: %.3f" % binwidth)
    tpvl = [0.1, 0.2]  # tpvlp, tpvls
    ylim = [15000, 5000]  # tpvlp, tpvls
    ylim = [2000, 1000]  # tpvlp, tpvls
    _pidx = 0
    for (ax, data, title) in zip(
                    axs, ((resP_dkpn, resP_pn), (resS_dkpn, resS_pn)),
                         ('TP+FP P-residuals\n%s' % add_title_p,
                          'TP+FP S-residuals\n%s' % add_title_s)):

        ax.hist(data[0], bins=bin_edges, color="orange", edgecolor=None,
                label="DKPN: count=%d\n            mean=%.2f\n             std=%.2f" % (len(data[0]), mean(data[0]), std(data[0])))
        ax.hist(data[1], bins=bin_edges, facecolor=(.0, .0, .0, .0), edgecolor='blue',
                label="PN: count=%d\n    mean=%.2f\n      std=%.2f" % (len(data[1]), mean(data[1]), std(data[1])))

        ax.axvline(-tpvl[_pidx], color="darkgray", ls="dashed")
        ax.axvline(tpvl[_pidx], color="darkgray", ls="dashed")

        # Set labels and title
        ax.set_xlabel('residuals (s)')
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([0, ylim[_pidx]])
        ax.set_ylabel('count')
        ax.set_title(title)
        ax.legend()
        _pidx += 1
    #
    plt.tight_layout()  # Adjust the spacing between subplots


if __name__ == "__main__":

    fig, axs = plt.subplots(nrows=1, ncols=2,
                            figsize=(12, 4))

    plt.suptitle("%s - %s" % (
                DATASET, SIZE),
                fontweight='bold', fontsize=18)  # fontfamily='sans-serif', fontname="Lato")

    _work_path_DKPN_P = Path("./Rnd_" + RANDOM_NUMBER_DKPN_P) / (
            "Results_INSTANCE_"+DATASET+"_"+SIZE+"_"+THRVAL_DKPN_P)

    _work_path_DKPN_S = Path("./Rnd_" + RANDOM_NUMBER_DKPN_S) / (
            "Results_INSTANCE_"+DATASET+"_"+SIZE+"_"+THRVAL_DKPN_S)

    _work_path_PN_P = Path("./Rnd_" + RANDOM_NUMBER_PN_P) / (
            "Results_INSTANCE_"+DATASET+"_"+SIZE+"_"+THRVAL_PN_P)

    _work_path_PN_S = Path("./Rnd_" + RANDOM_NUMBER_PN_S) / (
            "Results_INSTANCE_"+DATASET+"_"+SIZE+"_"+THRVAL_PN_S)

    # ========================= P

    # TruePositive
    DKPN_P_TP = __extract_pickle__(
            str(_work_path_DKPN_P / "DKPN_TP_P_residuals.pickle")
            )

    PN_P_TP = __extract_pickle__(
            str(_work_path_PN_P / "PN_TP_P_residuals.pickle")
            )

    try:
        # FalsePositive
        DKPN_P_FP = __extract_pickle__(
                str(_work_path_DKPN_P / "DKPN_FP_P_residuals.pickle")
                )

        PN_P_FP = __extract_pickle__(
                str(_work_path_PN_P / "PN_FP_P_residuals.pickle")
                )

    except FileNotFoundError:
        DKPN_P_FP = []
        PN_P_FP = []

    # =========================  S

    # TruePositive
    DKPN_S_TP = __extract_pickle__(
            str(_work_path_DKPN_S / "DKPN_TP_S_residuals.pickle")
            )

    PN_S_TP = __extract_pickle__(
            str(_work_path_PN_S / "PN_TP_S_residuals.pickle")
            )

    try:
        # FalsePositive
        DKPN_S_FP = __extract_pickle__(
                str(_work_path_DKPN_S / "DKPN_FP_S_residuals.pickle")
                )

        PN_S_FP = __extract_pickle__(
                str(_work_path_PN_S / "PN_FP_S_residuals.pickle")
                )

    except FileNotFoundError:
        DKPN_S_FP = []
        PN_S_FP = []

    ALL_RES_DKPN_P = DKPN_P_TP.tolist() + DKPN_P_FP.tolist()
    ALL_RES_DKPN_S = DKPN_S_TP.tolist() + DKPN_S_FP.tolist()
    ALL_RES_PN_P = PN_P_TP.tolist() + PN_P_FP.tolist()
    ALL_RES_PN_S = PN_S_TP.tolist() + PN_S_FP.tolist()

    # --> End RANDOM -->  populate figure
    _ = create_residuals_plot_compare(
                    ALL_RES_DKPN_P, ALL_RES_DKPN_S,
                    ALL_RES_PN_P, ALL_RES_PN_S,
                    "DKPN thr: %.1f / PN thr: %.1f" % (float(THRVAL_DKPN_P), float(THRVAL_PN_P)),
                    "DKPN thr: %.1f / PN thr: %.1f" % (float(THRVAL_DKPN_S), float(THRVAL_PN_S)),
                    [axs[0], axs[1]])

    # --> End SIZES --> CLOSE FIGURE
    # ===============================================================
    fig.savefig("%s_%s_TP+FP_residuals.pdf" % (DATASET, SIZE))
    fig.savefig("%s_%s_TP+FP_residuals.png" % (DATASET, SIZE))


print("DONE")
