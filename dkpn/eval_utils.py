import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


def __reset_stats_dict__():
    stats_dict = {
        "TOTAL": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0
    }
    return stats_dict


def extract_picks(ts, smooth=True, thr=0.2, min_distance=50):
    """ Insert a prediction label TS per time """
    if smooth:
        smoothing_filter = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]  # remove rapid oscillations between samples
        ts = np.convolve(ts, smoothing_filter, mode='same')
    #
    peaks, _ = find_peaks(ts, thr, distance=min_distance)
    widths = peak_widths(ts, peaks, rel_height=0.5, prominence_data=None, wlen=None)
    widths = widths[0]
    #
    return (peaks, widths)


def compare_picks(peaks_model, peaks_ref, stats_dict, thr=25):
    stats_dict["TOTAL"] += len(peaks_ref)
    matches = []  # ... keeping track of matches

    residuals_tp, residuals_fp = [], []

    # First, we find TP and the corresponding matches
    matched_model_peaks = set()  # Keep track of model peaks that have been matched
    for pf in peaks_ref:
        match_found = False
        for pm in peaks_model:
            if pm in matched_model_peaks:
                continue  # Skip model peaks that have already been matched
            if abs(pm - pf) <= thr:
                residuals_tp.append(pm - pf)
                stats_dict["TP"] += 1
                matches.append((pm, pf))  # Save the match
                matched_model_peaks.add(pm)
                match_found = True
                break  # Once we found a match, no need to check other pm for this pf
        if not match_found:
            stats_dict["FN"] += 1

    # Now, we find FP. If a pm is not in the matches, it's a FP
    for pm in peaks_model:
        if pm not in [m[0] for m in matches]:
            stats_dict["FP"] += 1
            #
            for pf in peaks_ref:
                residuals_fp.append(pm - pf)

    return (stats_dict, residuals_tp, residuals_fp)


def calculate_scores(stats_dict):
    """ Go with the score """
    f1, precision, recall = 0, 0, 0
    #
    precision = stats_dict["TP"] / (stats_dict["TP"] + stats_dict["FP"] + 1e-6)
    recall = stats_dict["TP"] / (stats_dict["TP"] + stats_dict["FN"] + 1e-6)
    f1 = 2 * ((precision*recall)/((precision+recall)+1e-6))
    #
    return (f1, precision, recall)


def create_AL_plots(wave3c,
                    label3c,
                    dkpn_cfs,
                    label_PN,
                    label_DKPN,
                    picks_true_P,
                    picks_true_S,
                    picks_PN_P,
                    picks_PN_S,
                    picks_DKPN_P,
                    picks_DKPN_S,
                    save_path="image_probs.pdf"):
    #
    fig = plt.figure(figsize=(20, 18))
    axs = fig.subplots(9, 1, sharex=True)
    
    # 0-1-2
    axs[0].plot(wave3c[0], label="Z")
    axs[1].plot(wave3c[1], label="N")
    axs[2].plot(wave3c[2], label="E")
    # 3
    axs[3].plot(label3c[0], label="P_true")
    axs[3].plot(label3c[1], label="S_true")
    # 4
    axs[4].plot(label_PN[0], label="PN_P")
    axs[4].plot(label_PN[1], label="PN_S")
    # 5
    axs[5].plot(label_DKPN[0], label="DKPN_P")
    axs[5].plot(label_DKPN[1], label="DKPN_S")
    # 6
    axs[6].plot(dkpn_cfs[0], label="DKPN_CF_Z")
    axs[6].plot(dkpn_cfs[1], label="DKPN_CF_N")
    axs[6].plot(dkpn_cfs[2], label="DKPN_CF_E")
    # 7-8
    axs[7].plot(dkpn_cfs[3], label="DKPN_CF_incl.")
    axs[8].plot(dkpn_cfs[4], label="DKPN_CF_mod.")

    for pp in picks_true_P:
        for ax in axs:
            ax.axvline(x=pp, color='red', linestyle='-')

    for ss in picks_true_S:
        for ax in axs:
            ax.axvline(x=ss, color='red', linestyle='-')
    #
    for pp in picks_PN_P:
        for ax in axs:
            ax.axvline(x=pp, color='green', linestyle='-')

    for ss in picks_PN_S:
        for ax in axs:
            ax.axvline(x=ss, color='green', linestyle='-')
    #
    for pp in picks_DKPN_P:
        for ax in axs:
            ax.axvline(x=pp, color='black', linestyle='-')

    for ss in picks_DKPN_S:
        for ax in axs:
            ax.axvline(x=ss, color='black', linestyle='-')

    # FIX AX LIMIT + LEGEND
    for xx, ax in enumerate(axs):
        ax.legend()
        if xx in (3, 4, 5):
            ax.set_ylim(0, 1)
        
    # STORE+SAVE
    plt.tight_layout()
    _ = fig.savefig(str(save_path))
    return fig


def create_residuals_plot(resP, resS, binwidth=0.025, save_path="image_residuals.pdf"):
    fig = plt.figure(figsize=(10, 5))
    axs = fig.subplots(1, 2, sharex=True)
    
    for (ax, data, color, titles) in zip(
                    axs, (resP, resS), ("teal", "orange"), ('True Positive P-residuals',
                                                            'True Positive S-residuals')):
        bin_edges = np.arange(min(data), max(data) + binwidth, binwidth)
        ax.hist(data, bins=bin_edges, color=color)
        # Set labels and title
        ax.set_xlabel('residuals (s)')
        ax.set_ylabel('count')
        ax.set_title(titles)

    # STORE+SAVE
    plt.tight_layout()
    _ = fig.savefig(str(save_path))
    return fig


def create_residuals_plot_compare(resP_dkpn, resS_dkpn, resP_pn, resS_pn, 
                                  binwidth=0.025, save_path="image_residuals.pdf"):

    fig = plt.figure(figsize=(12, 3))
    axs = fig.subplots(1, 2, sharex=True)
    fig.suptitle('TP + FP')
    bin_edges = np.arange(-11.05, 11.05 + binwidth, binwidth)

    for (ax, data, title) in zip(
                    axs, ((resP_dkpn, resP_pn), (resS_dkpn, resS_pn)), ('P', 'S')):

        ax.hist(data[0], bins=bin_edges, color="orange", edgecolor=None,
                label="DKPN: mean=%.2f\n           std=%.2f" % (np.mean(data[0]), np.std(data[0])))
        ax.hist(data[1], bins=bin_edges, facecolor=(.0, .0, .0, .0), edgecolor='blue',
                label="PN: mean=%.2f\n      std=%.2f" % (np.mean(data[1]), np.std(data[1])))

        # Set labels and title
        ax.set_xlabel('residuals (s)')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylabel('count')
        ax.set_title(title)
        ax.legend()

    # STORE+SAVE
    plt.tight_layout()
    _ = fig.savefig(str(save_path))
    return fig
