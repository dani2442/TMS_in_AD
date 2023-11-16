#%%
import numpy as np
import BOLDFilters as BOLDFilters
from petTOAD_setup import *
import neurolib.utils.functions as func
import matplotlib.pyplot as plt



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
    ts_filt_narrow = BOLDFilters.BandPassFilter(signal, removeStrongArtefacts=False)
    pw_filt_narrow = np.abs(np.fft.fft(ts_filt_narrow, axis=1))
    PowSpect_filt_narrow = pw_filt_narrow[:, 0:int(np.floor(Tmax/2))].T**2 / (Tmax/TR)
    return PowSpect_filt_narrow


def filtPowSpetraMultipleSubjects(signal, TR, smooth=True):
    if signal.ndim == 2:
        print("I am doing one subject")
        nSubjects = 1
        nNodes, Tmax = signal.shape  # Here we are assuming we receive only ONE subject...
        Power_Areas_filt_narrow_unsmoothed = filtPowSpetra(signal, TR)
    else:
        print("I am doing more than one subject")
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
    for seed in np.arange(nNodes):
        Power_Areas_filt_narrow_smoothed[:,seed] = gaussfilt(freqs, Power_Areas_filt_narrow_unsmoothed[:,seed], 0.01)
    if smooth:
        idxFreqOfMaxPwr = np.argmax(Power_Areas_filt_narrow_smoothed, axis=0)
    else:
        idxFreqOfMaxPwr = np.argmax(Power_Areas_filt_narrow_unsmoothed, axis=0)
    f_diff = freqs[idxFreqOfMaxPwr]
    return f_diff


# Get the timeseries for the chosen group
group_hc_no_wmh, timeseries_hc_no_wmh = get_group_ts_for_freqs(HC_no_WMH, all_fMRI_clean)
group_hc_wmh, timeseries_hc_wmh = get_group_ts_for_freqs(HC_WMH, all_fMRI_clean)
group_mci_no_wmh, timeseries_mci_no_wmh = get_group_ts_for_freqs(MCI_no_WMH, all_fMRI_clean)
group_mci_wmh, timeseries_mci_wmh = get_group_ts_for_freqs(MCI_WMH, all_fMRI_clean)
timeseries_all = np.concatenate([timeseries_hc_no_wmh, timeseries_hc_wmh, timeseries_mci_no_wmh, timeseries_mci_wmh])



#%%
f_diff_hc_no_wmh_smooth = filtPowSpetraMultipleSubjects(timeseries_hc_no_wmh, TR)
f_diff_hc_wmh_smooth = filtPowSpetraMultipleSubjects(timeseries_hc_wmh, TR)
f_diff_mci_no_wmh_smooth = filtPowSpetraMultipleSubjects(timeseries_mci_no_wmh, TR)
f_diff_mci_wmh_smooth = filtPowSpetraMultipleSubjects(timeseries_mci_wmh, TR)
f_diff_all_smooth = filtPowSpetraMultipleSubjects(timeseries_all, TR)

f_diff_hc_no_wmh_unsmooth = filtPowSpetraMultipleSubjects(timeseries_hc_no_wmh, TR, smooth = False)
f_diff_hc_wmh_unsmooth = filtPowSpetraMultipleSubjects(timeseries_hc_wmh, TR, smooth = False)
f_diff_mci_no_wmh_unsmooth = filtPowSpetraMultipleSubjects(timeseries_mci_no_wmh, TR, smooth = False)
f_diff_mci_wmh_unsmooth = filtPowSpetraMultipleSubjects(timeseries_mci_wmh, TR, smooth = False)
f_diff_all_unsmooth = filtPowSpetraMultipleSubjects(timeseries_all, TR, smooth = False)

#%%
from scipy.signal import welch as welch
def ws_group_welch(ts_group):
    fs = 1/TR
    fs_group = []
    for i in range(ts_group.shape[0]):
        fs_subj = []
        for ts_roi in ts_group[i]:
            (f, S)= welch(ts_roi, fs, nperseg=193)
            fs_subj.append(f[np.argmax(S)])
        fs_group.append(fs_subj)
    return np.array(fs_group).mean(axis=0)


print('Performing analyses with Welch method')
ws_hc_no_wmh = ws_group_welch(timeseries_hc_no_wmh)
ws_hc_wmh = ws_group_welch(timeseries_hc_wmh)
ws_mci_no_wmh = ws_group_welch(timeseries_mci_no_wmh)
ws_mci_wmh = ws_group_welch(timeseries_mci_wmh)

#%%
cols = ['HC', 'MCI']
rows = ['Smoothed', 'Unsmoothed', 'Welch']
  
fig, axs = plt.subplots(3, 2, figsize = (10,10))
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')

axs[0,0].hist(f_diff_hc_no_wmh_smooth, label = 'No WMH', alpha = 0.4);
axs[0,0].hist(f_diff_hc_wmh_smooth, label = 'WMH', alpha = 0.4);
axs[0,1].hist(f_diff_mci_no_wmh_smooth, alpha = 0.4);
axs[0,1].hist(f_diff_mci_wmh_smooth, alpha = 0.4);
axs[0,0].annotate(f'{np.unique(f_diff_hc_no_wmh_smooth)}', (0.01, 20))
axs[0,0].set_xlim(0.034, 0.037)
axs[0,1].set_xlim(0.034, 0.037)

axs[1,0].hist(f_diff_hc_no_wmh_unsmooth, alpha = 0.4, bins = 4);
axs[1,0].hist(f_diff_hc_wmh_unsmooth, alpha = 0.4, bins = 4);
axs[1,1].hist(f_diff_mci_no_wmh_unsmooth, alpha = 0.4, bins = 4);
axs[1,1].hist(f_diff_mci_wmh_unsmooth, alpha = 0.4, bins = 4);

axs[2,0].hist(ws_hc_no_wmh, alpha = 0.4, bins = 4);
axs[2,0].hist(ws_hc_wmh, alpha = 0.4, bins = 4);

axs[2,1].hist(ws_mci_no_wmh, alpha = 0.4, bins = 4);
axs[2,1].hist(ws_mci_wmh, alpha = 0.4, bins = 4);
fig.legend()
fig.tight_layout()


# %%
import seaborn as sns
fig, axs = plt.subplots(3, 2, figsize = (10,10), sharex='col')
for ax, col in zip(axs[0], cols):
    ax.set_title(col)
for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')

sns.kdeplot(ax = axs[0,0], 
            data=f_diff_hc_no_wmh_smooth, 
            fill = True,
            label = 'No WMH') 
sns.kdeplot(ax = axs[0,0], 
            data=f_diff_hc_wmh_smooth, 
            label = 'WMH',
            fill = True,
            alpha = 0.4);
sns.kdeplot(ax = axs[0,1], 
            data=f_diff_mci_no_wmh_smooth, 
            fill = True,
            alpha = 0.4);
sns.kdeplot(ax = axs[0,1], 
            data=f_diff_mci_wmh_smooth, 
            fill = True,
            alpha = 0.4);
sns.kdeplot(ax = axs[1,0], 
            data=f_diff_hc_no_wmh_unsmooth, 
            fill = True,
            bw_adjust = 2,
            alpha = 0.4);
sns.kdeplot(ax = axs[1,0], 
            data=f_diff_hc_wmh_unsmooth, 
            fill = True,
            bw_adjust = 2,
            alpha = 0.4);
sns.kdeplot(ax = axs[1,1], 
            data=f_diff_mci_no_wmh_unsmooth, 
            bw_adjust = 2,
            fill = True,
            alpha = 0.4);
sns.kdeplot(ax = axs[1,1], 
            data=f_diff_mci_wmh_unsmooth, 
            bw_adjust = 2,
            fill = True,
            alpha = 0.4);

sns.kdeplot(ax = axs[2,0], 
            data=ws_hc_no_wmh, 
            fill = True,
            bw_adjust = 2,
            alpha = 0.4);
sns.kdeplot(ax = axs[2,0], 
            data=ws_hc_wmh, 
            fill = True,
            bw_adjust = 2,
            alpha = 0.4);
sns.kdeplot(ax = axs[2,1], 
            data=ws_mci_no_wmh, 
            bw_adjust = 2,
            fill = True,
            alpha = 0.4);
sns.kdeplot(ax = axs[2,1], 
            data=ws_mci_wmh, 
            bw_adjust = 2,
            fill = True,
            alpha = 0.4);

axs[0,0].set_xlim(0.025, 0.045)
axs[1,0].set_xlim(0.025, 0.045)
axs[2,0].set_xlim(0.025, 0.045)

axs[0,1].set_xlim(0.025, 0.045)
axs[1,1].set_xlim(0.025, 0.045)
axs[2,1].set_xlim(0.025, 0.045)

fig.legend()
fig.tight_layout()
# %%
