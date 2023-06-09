# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================


import math
import numpy as np
from functools import lru_cache
from obspy.core import Trace, Stats
from obspy.signal.invsim import cosine_taper
from seisbench_utils.phasepapy_alomax.phasepicker import fbpicker


class FilterPickerCF():
    """
    Initialises an FBPicker and proves access to statistics for each band.

    :param t_long: the time in seconds of moving window to calculate CFn of each bandpass filtered data
    :type t_long: float
    :param freqmin: the center frequency of first octave filtering band
    :type freqmin: float
    :param corner: corner order of bandpass filtering
    :type corner: int
    :param perc_taper: percent cosine taper of each band after bandpass filtering
    :type perc_taper: float
    :param clip: clip level for characteristic function. If < 0, no clipping
    :type perc_taper: float
    :param mode: two options: standard deviation (std) or root mean square (rms)
    :type mode: string
    :param do_polarization: If true calculates polarization arrays based on FilterPicker channel-band CF peaks.
    :type do_polarization: bool, optional
    :param win_len: Sliding window length in samples. If < 2, computes the single point, instantaneous particle-motion
    :type win_len: int
    :param use_amax_only: If true calculates polarization arrays only for band with global max FilterPicker channel-band CF.
    :type use_amax_only: bool, optional
    :param normalize: If true normalizes FilterPicker and modulis waveforms, but not inclination.
    :type normalize: bool, optional
    :param log: If true normalizes FilterPicker and modulis waveforms, but not inclination.
    :type log: bool, optional
    :param reverse_components: If true reverses component order before processing, e.g. ENZ -> ZNE. Does not change metadata, use only when components miss-labeled.
    :type reverse_components: bool, optional
    :param kwargs: All kwargs are passed directly to FilterPickerCF

    """

    _picker = None

    def __init__(self, t_long = 5, freqmin = 1, corner = 1, perc_taper = 0.1, clip = -1.0, mode = 'rms',
                 do_polarization = False, polarization_win_len = None, use_amax_only=True, normalize=True, log=False,
                 reverse_components=False, **kwargs):
        # Create a picker object with pick parameters
        self._picker = fbpicker.FBPicker(t_long = t_long, freqmin = freqmin, corner = corner, perc_taper = perc_taper, \
            mode = mode)
        # the following FBPicker arguments are not needed for FBSummary
        #, t_ma = 20, nsigma = 8, t_up = 0.4, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
        self._clip = clip
        self._do_polarization = do_polarization
        self._polarization_win_len = polarization_win_len
        self._use_amax_only = use_amax_only
        self._normalize = normalize
        self._log = log
        self._reverse_components = reverse_components


    def __call__(self, state_dict):
        """
        Provides an augmentation for seisbench.generate.generator.GenericGenerator.add_augmentations().
        """

        #print(state_dict)

        waveforms, metadata = state_dict["X"]
        #print(metadata)
        #print(waveforms)
        #print(waveforms.shape)

        fp_cf_waveforms = []
        fp_band_data = []
        fp_max_arrays_band = []
        fp_max_arrays_val = []
        waveforms_work = waveforms
        if self._reverse_components:
            waveforms_work = []
            waveforms_work.append(waveforms[2].copy())
            waveforms_work.append(waveforms[1].copy())
            waveforms_work.append(waveforms[0].copy())
        for channel_index, waveform in enumerate(waveforms_work):
            # Create an obspy trace object
            stats = Stats()
            stats.sampling_rate = metadata['trace_sampling_rate_hz']
            trace = Trace(data=waveform, header=stats)
            summary = self.get_summary(trace)
            fp_cf_waveforms.append(summary.summary)
            if self._do_polarization:
                fp_band_data.append(summary.BF)
                #print("summary.FC.shape()", summary.FC.shape)
                #print("summary.FC", summary.FC)
                fp_max_array_band = np.argmax(summary.FC, axis=0)
                #print("fp_max_array_band.shape()", fp_max_array_band.shape)
                #print("fp_max_array_band", fp_max_array_band)
                fp_max_arrays_band.append(fp_max_array_band)
                fp_max_array_val = summary.summary
                #print("fp_max_array_val.shape()", fp_max_array_val.shape)
                #print("fp_max_array_val", fp_max_array_val)
                fp_max_arrays_val.append(fp_max_array_val)

        fp_cf_waveforms = np.asarray(fp_cf_waveforms)
        # clip
        if self._clip > 0.0:
            fp_cf_waveforms = np.clip(fp_cf_waveforms, a_min = None, a_max = self._clip)
        if self._log:
            fp_cf_waveforms = np.log10(fp_cf_waveforms + 1.0)
        if self._normalize:
            # normalize
            eps=1e-10
            fp_cf_waveforms = fp_cf_waveforms / (
                np.amax(np.abs(fp_cf_waveforms), axis=(0,1), keepdims=True) + eps)
        #print(fp_cf_waveforms.shape)
        state_dict["Xfp"] = (fp_cf_waveforms, {'t_long': 5}) # TODO: complete FP metadata

        if self._do_polarization:
            fp_band_data = np.asarray(fp_band_data)
            fp_max_arrays_band = np.asarray(fp_max_arrays_band)     # [nchan, npts]
            #print("fp_max_arrays_band.shape()", fp_max_arrays_band.shape)
            #print("fp_max_arrays_band", fp_max_arrays_band)
            #print("argmax", np.argmax(fp_max_arrays_band))
            fp_max_arrays_band_amax = np.amax(fp_max_arrays_band, axis=0)       # [npts]
            #print("fp_max_arrays_band_amax.shape()", fp_max_arrays_band_amax.shape)
            #print("fp_max_arrays_band_amax", fp_max_arrays_band_amax)
            band_amax = -1
            if self._use_amax_only:
                fp_max_arrays_val = np.asarray(fp_max_arrays_val)       # [nchan, npts]
                #print("fp_max_arrays_val.shape()", fp_max_arrays_val.shape)
                #print("fp_max_arrays_val", fp_max_arrays_val)
                fp_max_arrays_val_argmax_ndx = np.argmax(fp_max_arrays_val)
                #print("argmax", fp_max_arrays_val_argmax_ndx)
                index_amax = np.unravel_index(fp_max_arrays_val_argmax_ndx, fp_max_arrays_val.shape)
                #print("index_amax", index_amax)
                band_amax = fp_max_arrays_band[index_amax]
                #print("band_amax", band_amax)
            result_dict = self.PolarizationFP(fp_band_data, fp_max_arrays_band_amax, self._use_amax_only, band_amax, self._normalize, self._log)
            state_dict["Xpol_inc"] = (np.expand_dims(result_dict["incidence"], 0), {'method': "incidence_modulus"})
            state_dict["Xpol_modulus"] = (np.expand_dims(result_dict["modulus"], 0), {'method': "incidence_modulus"})

    def get_summary(self, trace):
        """
        Return an FBSummary summary array for specified Trace.

        :rtype: XXX
        :return: XXX

        The XXX contains the  for the current Trace object.
        """

        trace.detrend('linear')  # Perform a linear detrend on the data
        summary = fbpicker.FBSummary(self._picker, trace)
        return summary

    def PolarizationFP(self, fp_band_data, fp_max_arrays_band_amax,
                       use_amax_only=False, band_amax=-1, normalize=True,
                       log=False):

        nband = band_amax

        res = []

        if use_amax_only:
            res = self.incidence_modulus(fp_band_data[0][nband], fp_band_data[1][nband], fp_band_data[2][nband], normalize, log)
        else:
            # single-point particle-motion
            trace_len = len(fp_band_data[0][0])
            if False: # original point by point assembly of particle-motion traces
                data = np.zeros(shape=(3, trace_len))
                offset = 0
                while offset < trace_len:
                    #print("offset", offset)
                    nband = fp_max_arrays_band_amax[offset]
                    data[0][offset] = fp_band_data[0][nband][offset]
                    data[1][offset] = fp_band_data[1][nband][offset]
                    data[2][offset] = fp_band_data[2][nband][offset]
                    offset += 1
            else:  # TEST # vector assembly of particle-motion traces
                data = fp_band_data[:, fp_max_arrays_band_amax, np.arange(trace_len, dtype=int)]

            res = self.incidence_modulus(data[0], data[1], data[2], normalize, log)

        result_dict = {
                    "incidence": res[0],
                    "modulus": res[1],
                    }
        return result_dict


    def incidence_modulus(self, vertical, north, east, normalize, log):
        """
        Computes the single point, instantaneous particle-motion azimuth, incidence and magnitude of a 3-comp time-series

        :param vertical: ZNE trace data
        :type float
        :returns:  incidence (-1.0 (down) -> 0 (horiz) -> 1.0 (up)) and modulus
        """

        hxy = np.hypot(north, east)
        modulus = np.hypot(hxy, vertical)
        # if no horizontal signal, set incidence to 0
        if np.max(hxy) > np.max(vertical) / 1000.0:
            incidence = np.arctan2(vertical, hxy)   # -pi (down) -> 0 (horiz) -> pi (up)
            incidence = incidence / np.pi   # -1.0 (down) -> 0 (horiz) -> 1.0 (up)
        else:
            incidence = np.zeros_like(vertical)
        if log:
            modulus = np.log10(modulus + 1.0)
        if normalize:
            modulus = modulus / (np.max(modulus + 1e-6))    # normalized

        return [incidence, modulus]

