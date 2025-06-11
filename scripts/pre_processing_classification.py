from snakemake.script import snakemake
import pandas as pd
from helper import * 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()

raw_data = pd.read_pickle(snakemake.input.raw_data)
spikes_detected = pd.read_pickle(snakemake.input.spikes_detected)
spikes = pd.read_pickle(snakemake.input.spikes)
track_names = sorted(spikes['track'].unique())
track_of_interest_label = snakemake.params.track_of_interest_label

# create dataframes for feature set extraction
def create_dataframes(spikes):
    ap_window_iloc = spikes[["spike_ts"]]\
        .progress_apply(ts_to_idx_col_fd_min, args=(raw_data,), lower=15, upper=15, axis=0)\
        .progress_apply(bounds, axis=1, result_type='expand',lower=-15, upper=15 )\
        .rename(columns={0:'start_iloc', 1:'end_iloc'})
    ap_track_window = ap_window_iloc.merge(spikes, on='spike_idx')
    ap_raw_background = ap_window_iloc[['start_iloc','end_iloc']]\
        .progress_apply(extract_raw_values, args=(raw_data,), axis=1, result_type='expand')\
        .rename(columns={0: "raw"})
    ap_derivatives = ap_window_iloc[['start_iloc','end_iloc']]\
        .progress_apply(calculate_fd_sd, args=(raw_data,), axis=1, result_type='expand')\
        .rename(columns={0:'fd', 1:'sd', 2:'fd_crossings', 3: "raw"})
    return ap_window_iloc, ap_raw_background, ap_track_window, ap_derivatives

# create df from detected spikes
ap_window_iloc_detected, ap_raw_detected, \
    ap_track_window_detected, ap_derivatives_detected = create_dataframes(spikes_detected)

# iterate over each track label
background_dfs = []
for track in track_names:
    spikes_track = spikes[spikes["track"] == track].copy()
    spikes_track_background = check_if_background(spikes_track, 'spike_ts')
    #if track == track_of_interest_label:
    #    spikes_background_single = spikes_track_background[spikes_track_background['is_background'] == True]\
    #                    .drop(['is_background'], axis=1)
    spikes_track_background = spikes_track_background[spikes_track_background["is_background"]]
    background_dfs.append(spikes_track_background.drop("is_background", axis=1))
spikes_background_all = pd.concat(background_dfs, sort=False).sort_index()

# create df from background spikes
ap_window_iloc_background_all, ap_raw_background_all,\
ap_track_window_background_all, ap_derivatives_background_all = create_dataframes(spikes_background_all)


# save all dataframes 
ap_window_iloc_detected.to_pickle(snakemake.output.ap_window_iloc_d)
ap_derivatives_detected.to_pickle(snakemake.output.ap_derivatives_d)
ap_track_window_detected.to_pickle(snakemake.output.ap_track_window_d)
ap_raw_detected.to_pickle(snakemake.output.ap_raw_d)
spikes_detected.to_pickle(snakemake.output.spikes_d)

ap_window_iloc_background_all.to_pickle(snakemake.output.ap_window_iloc_a)
ap_derivatives_background_all.to_pickle(snakemake.output.ap_derivatives_a)
ap_track_window_background_all.to_pickle(snakemake.output.ap_track_window_a)
ap_raw_background_all.to_pickle(snakemake.output.ap_raw_a)
spikes_background_all.to_pickle(snakemake.output.spikes_a)

