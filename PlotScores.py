#!/usr/bin/env python

import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set(style="darkgrid")

TRAINNAME = sys.argv[1]
TESTNAME = sys.argv[2]

# =================================================
# =================================================
# =================================================

STORE_DIR = Path("./RESULTS")
if not STORE_DIR.is_dir():
    STORE_DIR.mkdir(parents=True, exist_ok=True)

# =================================================
# =================================================
# =================================================

DKPN_SCORES = {}
pickles_DKPN = Path(".").glob("Results_%s_%s_*/results_DKPN.pickle" % (
                               TRAINNAME, TESTNAME))

for pp in pickles_DKPN:
    pp = str(pp)
    SIZE = pp.split("/")[0].split("_")[-1]
    # Load the pickle file
    with open(pp, 'rb') as file:
        loaded_data = pickle.load(file)
    DKPN_SCORES[SIZE] = loaded_data

PN_SCORES = {}
pickles_PN = Path(".").glob("Results_INSTANCE_INSTANCE_*/results_PN.pickle")

for pp in pickles_PN:
    pp = str(pp)
    SIZE = pp.split("/")[0].split("_")[-1]
    # Load the pickle file
    with open(pp, 'rb') as file:
        loaded_data = pickle.load(file)
    PN_SCORES[SIZE] = loaded_data

# =================================================
# =================================================
# =================================================

color_list = ["darkgoldenrod", "magenta", "blue"]
sns.set(style='darkgrid')  # Set the style of the plot


def extract_dict_values(indict, what_list, order=["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY"]):
    value_lists = []
    for key in what_list:
        value_list = [indict[item][key] for item in order]
        value_lists.append(value_list)

    return value_lists

# =================================================
# =================================================
# =================================================


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ===============================================================
# ========================================  AX-1 --> scores P
ax = axs[0, 0]

# --- Get VALS
key_list = ["P_FN", "P_FP", "P_TP"]
x_labels = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY"]

VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=x_labels)
VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=x_labels)

for i, values in enumerate(VALUES_PN):
    sns.lineplot(x=x_labels, y=values, marker='o', linestyle="dashed",
                 label="PN_"+key_list[i], ax=ax, color=color_list[i])

for i, values in enumerate(VALUES_DKPN):
    sns.lineplot(x=x_labels, y=values, marker='s', label="DKPN_"+key_list[i],
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
x_labels = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY"]

VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=x_labels)
VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=x_labels)

for i, values in enumerate(VALUES_PN):
    sns.lineplot(x=x_labels, y=values, marker='o', linestyle="dashed",
                 label="PN_"+key_list[i], ax=ax, color=color_list[i])

for i, values in enumerate(VALUES_DKPN):
    sns.lineplot(x=x_labels, y=values, marker='s', label="DKPN_"+key_list[i],
                 ax=ax, color=color_list[i])

# --- Decorator
ax.set_ylim([0, 1])
ax.set_xlabel('train size', fontstyle='italic', fontsize=14, fontname="Lato")
ax.set_ylabel('value', fontstyle='italic', fontsize=14, fontname="Lato")
ax.set_title('P-SCORES', fontstyle='italic', fontsize=16, fontname="Lato")   # Set the title of the plot
ax.legend()  # Display the legend


# ===============================================================
# ========================================  AX-2 --> count S
ax = axs[0, 1]

# --- Get VALS
key_list = ["S_FN", "S_FP", "S_TP"]
x_labels = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY"]

VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=x_labels)
VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=x_labels)

for i, values in enumerate(VALUES_PN):
    sns.lineplot(x=x_labels, y=values, marker='o', linestyle="dashed",
                 label="PN_"+key_list[i], ax=ax, color=color_list[i])

for i, values in enumerate(VALUES_DKPN):
    sns.lineplot(x=x_labels, y=values, marker='s', label="DKPN_"+key_list[i],
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
x_labels = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY"]

VALUES_DKPN = extract_dict_values(DKPN_SCORES, key_list, order=x_labels)
VALUES_PN = extract_dict_values(PN_SCORES, key_list, order=x_labels)

for i, values in enumerate(VALUES_PN):
    sns.lineplot(x=x_labels, y=values, marker='o', linestyle="dashed",
                 label="PN_"+key_list[i], ax=ax, color=color_list[i])

for i, values in enumerate(VALUES_DKPN):
    sns.lineplot(x=x_labels, y=values, marker='s', label="DKPN_"+key_list[i],
                 ax=ax, color=color_list[i])

# --- Decorator
ax.set_ylim([0, 1])
ax.set_xlabel('train size', fontstyle='italic', fontsize=14, fontname="Lato")
ax.set_ylabel('value', fontstyle='italic', fontsize=14, fontname="Lato")
ax.set_title('S-SCORES', fontstyle='italic', fontsize=16, fontname="Lato")   # Set the title of the plot
ax.legend()  # Display the legend


# ===============================================================
plt.suptitle('ETHZ Training (B:32 - LR:1e-4 - EP: 100) - ETHZ Testing', fontweight='bold', fontsize=18, fontname="Lato")  #fontfamily='sans-serif')
plt.tight_layout()  # Adjust the spacing between subplots
fig.savefig(str(
        STORE_DIR / ("ScoresResults_%s_%s.pdf" % (TRAINNAME, TESTNAME))
    )
)
