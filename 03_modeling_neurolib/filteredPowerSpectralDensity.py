#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Filter power density to get omega -- Version 1
Last edit:  2023/03/03
Authors:    Patow, Gustavo 
            Leone, Riccardo
Notes:      - Taken from WholeBrain (changed custom function for filtering to nilearn.clean)
            - Release notes:
                * Initial release
To do:      - 
Comments:   

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""
# #--------------------------------------------------------------------------
# COMPUTE POWER SPECTRA FOR
# NARROWLY FILTERED DATA WITH LOW BANDPASS (0.04 to 0.07 Hz)
# not # WIDELY FILTERED DATA (0.04 Hz to justBelowNyquistFrequency)
#     # [justBelowNyquistFrequency depends on TR,
#     # for a TR of 2s this is 0.249 Hz]
# #--------------------------------------------------------------------------
#%%
import numpy as np
import BOLDFilters as BOLDFilters


def conv(u,v):  # python equivalent to matlab conv 'same' method
    # from https://stackoverflow.com/questions/38194270/matlab-convolution-same-to-numpy-convolve
    npad = len(v) - 1
    full = np.convolve(u, v, 'full')
    first = npad - npad//2
    return full[first:first+len(u)]


def gaussfilt(t,z,sigma):
    # Apply a Gaussian filter to a time series
    #    Inputs: t = independent variable, z = data at points t, and
    #        sigma = standard deviation of Gaussian filter to be applied.
    #    Outputs: zfilt = filtered data.
    #
    #    based on the code by James Conder. Aug 22, 2013
    #    (partial) translation by Gustavo Patow
    n = z.size  # number of data
    a = 1/(np.sqrt(2*np.pi)*sigma)   # height of Gaussian
    sigma2 = sigma*sigma

    # check for uniform spacing
    # if so, use convolution. if not use numerical integration
    # uniform = false;
    dt = np.diff(t)
    dt = dt[0]
    # Only the uniform option is considered
    filter = dt * a * np.exp(-0.5*((t - np.mean(t)) ** 2)/sigma2)
    i = filter < dt * a * 1.e-6
    filter = np.delete(filter, i)  # filter[i] = []
    zfilt = conv(z, filter)
    onesToFilt = np.ones(np.size(z))     # remove edge effect from conv
    onesFilt = conv(onesToFilt, filter)
    zfilt = zfilt/onesFilt

    return zfilt


def filtPowSpetra(signal, TR):
    nNodes, Tmax = signal.shape  # Here we are assuming we receive only ONE subject...
    # Since we work with bandpass-filtered signal from when we load it, we don't do it again
    # This is just to keep Gustavo's code structure for future reference
    ts_filt_narrow = signal    #ts_filt_narrow = BOLDFilters.BandPassFilter(signal, removeStrongArtefacts=False)
    pw_filt_narrow = np.abs(np.fft.fft(ts_filt_narrow, axis=1))
    PowSpect_filt_narrow = pw_filt_narrow[:, 0:int(np.floor(Tmax/2))].T**2 / (Tmax/TR)
    return PowSpect_filt_narrow


def filtPowSpetraMultipleSubjects(signal, TR):
    if signal.ndim == 2:
        nSubjects = 1
        nNodes, Tmax = signal.shape  # Here we are assuming we receive only ONE subject...
        Power_Areas_filt_narrow_unsmoothed = filtPowSpetra(signal, TR)
    else:
        # In case we receive more than one subject, we do a mean...
        nSubjects, nNodes, Tmax = signal.shape
        PowSpect_filt_narrow = np.zeros((nSubjects, nNodes, int(np.floor(Tmax/2))))
        for s in range(nSubjects):
            print(f'filtPowSpectraMultipleSubjects: subject {s+1} (of {nSubjects})')
            PowSpect_filt_narrow[s] = filtPowSpetra(signal[s,:,:], TR).T
        Power_Areas_filt_narrow_unsmoothed = np.mean(PowSpect_filt_narrow, axis=0).T
    Power_Areas_filt_narrow_smoothed = np.zeros_like(Power_Areas_filt_narrow_unsmoothed)
    Ts = Tmax * TR
    freqs = np.arange(0,Tmax/2-1)/Ts
    # The smoothing is too much for our dataset and we lose all the regional variability so we work with unsmoothed data
    # for seed in np.arange(nNodes):
    #     Power_Areas_filt_narrow_smoothed[:,seed] = gaussfilt(freqs, Power_Areas_filt_narrow_unsmoothed[:,seed], 0.01)
    idxFreqOfMaxPwr = np.argmax(Power_Areas_filt_narrow_unsmoothed, axis=0)
    f_diff = freqs[idxFreqOfMaxPwr]
    return f_diff
