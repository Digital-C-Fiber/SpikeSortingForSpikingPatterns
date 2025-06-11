from snakemake.script import snakemake
import pandas as pd
import numpy as np
import typing 
from scipy.signal import resample, savgol_filter

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()
from helper import *

# read dataframes
raw_data = pd.read_pickle(snakemake.input.raw_data)
stimulations = pd.read_pickle(snakemake.input.stimulations)
spikes = pd.read_pickle(snakemake.input.spikes)

# 30 datapoints to capture 3 ms at a sampling frequency 10,000 Hz 
lower = 15
upper = 15


fig, ax = plt.subplots()

# create df with index of raw signal and align by negative peak in first derivative space
ap_window_iloc = spikes[["spike_ts"]]\
    .progress_apply(ts_to_idx_col_fd_min, args=(raw_data,),lower=lower, upper=upper,  axis=0)\
    .progress_apply(bounds, axis=1, result_type='expand',lower=-lower, upper=upper )\
        .rename(columns={0:'start_iloc', 1:'end_iloc'})


# create df with derivatives 
ap_derivatives = ap_window_iloc[['start_iloc','end_iloc']]\
        .progress_apply(calculate_fd_sd, args=(raw_data,), axis=1, result_type='expand')\
        .rename(columns={0:'fd', 1:'sd', 2:'fd_crossings', 3: "raw"})

# merge df with indices and with spike times and track
ap_track_window = ap_window_iloc.merge(spikes, on='spike_idx')


# all to pickle as output files
ap_window_iloc.to_pickle(snakemake.output.ap_window_iloc)
ap_derivatives.to_pickle(snakemake.output.ap_derivatives)
ap_track_window.to_pickle(snakemake.output.ap_track_window)
