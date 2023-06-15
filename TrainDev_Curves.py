#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set(style="darkgrid")

STORE_DIR = Path("./RESULTS")
if not STORE_DIR.is_dir():
    STORE_DIR.mkdir(parents=True, exist_ok=True)

# =================================================
# =================================================
# =================================================

for _picker in ("DKPN", "PN"):

    print("### Working with %s ..." % _picker)

    traintestlosses = {}
    for _path in Path(".").glob("%s_TrainDataset_*/TRAIN_TEST_loss.csv" % _picker):
        _path = str(_path)
        SIZE = _path.split("/")[0].split("_")[4]
        traintestlosses[SIZE] = pd.read_csv(_path)
    #
    TRAINNAME = _path.split("/")[0].split("_")[2]
    RND = int(_path.split("/")[0].split("_")[6])
    BATCH = int(_path.split("/")[0].split("_")[12])
    LR = float(_path.split("/")[0].split("_")[10]); LR = "{:.1e}".format(LR)
    # EPOCHS = int(_path.split("/")[0].split("_")[8])

    order = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY"]
    traintestlosses = OrderedDict((key, traintestlosses[key]) for key in order)

    nsubs = len(traintestlosses.keys())

    _train = [df[" TRAIN_LOSS"] for sz, df in traintestlosses.items()]
    _test = [df[" TEST_LOSS"] for sz, df in traintestlosses.items()]
    _train = pd.concat([s for s in _train])
    _test = pd.concat([s for s in _test])
    ylimmax = max(max(_train), max(_test))

    _epochs = [df["EPOCH"] for sz, df in traintestlosses.items()]
    _epochs = pd.concat([s for s in _epochs])
    xlimmax = max(_epochs)

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs_flat = np.ravel(axs)

    for ii, (sz, df) in enumerate(traintestlosses.items()):

        # ===  MAIN
        ax = axs_flat[ii]
        sns.lineplot(x=df["EPOCH"], y=df[" TRAIN_LOSS"], linestyle="dashed",
                     label="Train", color="red", ax=ax)
        sns.lineplot(x=df["EPOCH"], y=df[" TEST_LOSS"],
                     label="Dev", color="teal", ax=ax)
        #
        ax.set_ylim([0, ylimmax])
        ax.set_xlim([-4, xlimmax])
        ax.set_xlabel('epochs', fontstyle='italic', fontsize=14, fontname="Lato")
        ax.set_ylabel('value', fontstyle='italic', fontsize=14, fontname="Lato")

        # === decorator
        ax.set_title(sz.upper(), fontstyle='italic', fontsize=16, fontname="Lato")   # Set the title of the plot
        ax.legend()  # Display the legend

        # === SUBPANEL
        if ii >= 2:
            _maxepoch = max(df["EPOCH"])
            ZOOM_EPOCHS_INTERVAL = [_maxepoch-((_maxepoch//3)*2), _maxepoch]

            ax_subpanel = inset_axes(ax, width="50%", height="25%", loc="center right")
            sns.lineplot(x=df["EPOCH"], y=df[" TRAIN_LOSS"], linestyle="dashed", color="red", ax=ax_subpanel)
            sns.lineplot(x=df["EPOCH"], y=df[" TEST_LOSS"], color="teal", ax=ax_subpanel)
            ax_subpanel.set_xlim(ZOOM_EPOCHS_INTERVAL)
            ax_subpanel.set_ylim([0, 0.1])
            ax_subpanel.set_xlabel('')
            ax_subpanel.set_ylabel('')

    # ===============================================================
    supfigtitle = "%s Train/Dev loss curves (RND: %d -B:%d - LR:%s) - %s" % (
                _picker, RND, BATCH, LR, TRAINNAME)
    plt.suptitle(supfigtitle, fontweight='bold', fontsize=18, fontname="Lato")
    # plt.tight_layout()
    fig.savefig(str(STORE_DIR / ("TrainTest_CURVES_%s.pdf" % _picker)))

print("DONE")
