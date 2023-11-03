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

METHOD = "median"  # "mean"

RANDOM_NUMBERS = ["17", "36", "50", "142", "234", "777", "987"]
DATASETS = [("INSTANCE", "INSTANCE"), ("INSTANCE", "ETHZ")]
SIZES = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY", "SMALL", "MEDIUM", "LARGE"]
THRVAL = ["02", "05"]

BASEFOLDER_START = "./Rnd_"
BASEFOLDER_END = ""

# =================================================
# =================================================
# =================================================


def __extract_pickle__(inpath):
    with open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def create_residuals_plot_compare(resP_dkpn, resS_dkpn, resP_pn, resS_pn, size_name, axs,
                                  binwidth=0.025):

    bin_edges = arange(-11.05, 11.05 + binwidth, binwidth)

    print("BINWIDTH: %.3f" % binwidth)
    tpvl = [0.1, 0.2]  # tpvlp, tpvls
    ylim = [15000, 5000]  # tpvlp, tpvls
    _pidx = 0
    for (ax, data, title) in zip(
                    axs, ((resP_dkpn, resP_pn), (resS_dkpn, resS_pn)), (size_name+'_P', size_name+'_S')):

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


if __name__ == "__main__":

    for (_ds1, _ds2) in DATASETS:  # InDomain CrossDomain

        for _thr in THRVAL:

            fig, axs = plt.subplots(len(SIZES), 2, figsize=(12, 3*len(SIZES)))
            axs = axs.flatten()
            # store_file=("%s_%s_%s_%s_scores.pdf" % , METHOD.lower())))
            print("%s Training - %s Testing - THR: %.1f" % (_ds1, _ds2, float(_thr)*0.1))
            plt.suptitle("%s Training - %s Testing - THR: %.1f" % (_ds1, _ds2, float(_thr)*0.1),
                         fontweight='bold', fontsize=18)  # fontfamily='sans-serif', fontname="Lato")

            xx = 0
            for _sz in SIZES:

                print("... %s" % _sz.upper())

                list_DKPN_P_TP, list_PN_P_TP = [], []
                list_DKPN_S_TP, list_PN_S_TP = [], []
                #
                list_DKPN_P_FP, list_PN_P_FP = [], []
                list_DKPN_S_FP, list_PN_S_FP = [], []

                for _rndm in RANDOM_NUMBERS:
                    _work_path = Path(BASEFOLDER_START+_rndm+BASEFOLDER_END) / ("Results_"+_ds1+"_"+_ds2+"_"+_sz+"_"+_thr)

                    # ========================= P

                    # TruePositive
                    list_DKPN_P_TP.extend(
                        __extract_pickle__(
                            str(_work_path / "DKPN_TP_P_residuals.pickle")
                            )
                        )

                    list_PN_P_TP.extend(
                        __extract_pickle__(
                            str(_work_path / "PN_TP_P_residuals.pickle")
                            )
                        )

                    try:
                        # FalsePositive
                        list_DKPN_P_FP.extend(
                            __extract_pickle__(
                                str(_work_path / "DKPN_FP_P_residuals.pickle")
                                )
                            )

                        list_PN_P_FP.extend(
                            __extract_pickle__(
                                str(_work_path / "PN_FP_P_residuals.pickle")
                                )
                            )
                    except FileNotFoundError:
                        list_DKPN_P_FP.extend([])
                        list_PN_P_FP.extend([])

                    # =========================  S

                    # TruePositive
                    list_DKPN_S_TP.extend(
                        __extract_pickle__(
                            str(_work_path / "DKPN_TP_S_residuals.pickle")
                            )
                        )

                    list_PN_S_TP.extend(
                        __extract_pickle__(
                            str(_work_path / "PN_TP_S_residuals.pickle")
                            )
                        )

                    try:
                        # FalsePositive
                        list_DKPN_S_FP.extend(
                            __extract_pickle__(
                                str(_work_path / "DKPN_FP_S_residuals.pickle")
                                )
                            )

                        list_PN_S_FP.extend(
                            __extract_pickle__(
                                str(_work_path / "PN_FP_S_residuals.pickle")
                                )
                            )
                    except FileNotFoundError:
                        list_DKPN_S_FP.extend([])
                        list_PN_S_FP.extend([])

                ALL_RES_DKPN_P = list_DKPN_P_TP + list_DKPN_P_FP
                ALL_RES_DKPN_S = list_DKPN_S_TP + list_DKPN_S_FP
                ALL_RES_PN_P = list_PN_P_TP + list_PN_P_FP
                ALL_RES_PN_S = list_PN_S_TP + list_PN_S_FP

                # --> End RANDOM -->  populate figure
                _ = create_residuals_plot_compare(
                                ALL_RES_DKPN_P, ALL_RES_DKPN_S,
                                ALL_RES_PN_P, ALL_RES_PN_S,
                                _sz.upper(),
                                [axs[xx], axs[xx+1]])
                xx += 2

            # --> End SIZES --> CLOSE FIGURE
            # ===============================================================

            plt.tight_layout()  # Adjust the spacing between subplots
            fig.savefig("%s_%s_%s_TP+FP_residuals.pdf" % (_ds1, _ds2, _thr))

        # --> End THR

    # --> End DATASETs

print("DONE")
