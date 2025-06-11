from snakemake.script import snakemake
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import quantities as pq
import typing 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from helper import * 

tqdm.pandas()

# read in dataframes s
spikes = pd.read_pickle(snakemake.input.spikes)
raw_data = pd.read_pickle(snakemake.input.raw_data)
track_of_interest_label = snakemake.params.track_of_interest_label
dataset_name = snakemake.params.name
stimulations = pd.read_pickle(snakemake.input.stimulations)
lower = 15
upper = 15
track_names = sorted(spikes['track'].unique())

# check if time ranges contain stimuli
def time_range_check(stimulations, time_ranges):
    time_range_check = {}
    for time_range_index in range(len(time_ranges)):
        for stim in stimulations:
            if (stim > time_ranges[time_range_index][0]) & (stim < time_ranges[time_range_index][1]):
                time_range_check[time_ranges[time_range_index]] = 1
    return time_range_check

# remove stimulation artifacts from detected spikes 
def remove_stimulation_artifacts(indices,times_list, stimulation):
    indices_copy = list(indices)
    for index in indices:
        timing = times_list[index]
        for stim in stimulation:
            # duration of stimulus not longer than 0.01 seconds 
            if (timing > stim) and (timing < (stim + 0.01)):
                indices_copy.remove(index)
    return indices_copy

# filter true spikes in time ranges for evaluation
def filter_spikes_in_ranges(spikes, time_ranges):
    spike_in_range = False
    for start, end in time_ranges:
        spike_in_range |= (spikes['spike_ts'] >= start) & (spikes['spike_ts'] <= end)
    return spikes[spike_in_range]

# function to evalute detection
def evaluate_detection(true_spikes, detected_spikes, tolerance= 0.002):
    true_spikes['matched'] = False
    detected_spikes['matched'] = False
    true_positive = 0

    # Match detected spikes to true spikes
    for i, true_spike in true_spikes.iterrows():
        in_range = detected_spikes[
            (detected_spikes['spike_ts'] >= (true_spike['spike_ts'] - tolerance)) &
            (detected_spikes['spike_ts'] <= (true_spike['spike_ts'] + tolerance)) &
            (~detected_spikes['matched'])]  # Unmatched spikes only
        if not in_range.empty:
            # If a match is found, mark both as matched
            true_spikes.at[i, 'matched'] = True
            if len(in_range) > 1:
                differences = in_range.apply(lambda row: np.abs(row.spike_ts - true_spike.spike_ts).min(), axis=1)
                index_with_smallest_difference = differences.idxmin()
                true_spikes.at[i, 'matched'] = True
                detected_spikes.at[index_with_smallest_difference, 'matched'] = True
            else:
                detected_spikes.at[in_range.index[0], 'matched'] = True
            true_positive += 1

    false_negative = len(true_spikes) - true_positive
    false_positive = len(detected_spikes) - true_positive

    results ={'TP': true_positive,
    'FN': false_negative,
    'FP': false_positive,
    'Precision': true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0,
    'Recall': true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0,
    'F1': (2 * true_positive / (2 * true_positive + false_positive + false_negative))
          if true_positive + false_positive + false_negative > 0 else 0}
    return(results)

# extract track of interest, specified before in config file 
spikes_of_interest = spikes[spikes["track"] == track_of_interest_label ].reset_index(drop=False).pipe(rename_index, 'spike_idx')
spike_of_interest_times = spikes_of_interest.drop(["track"], axis=1)

# label spikes and stimuli by their onset if they belong to the background or are extra spikes/stimuli
spikes_of_interest = check_if_background(spikes_of_interest, 'spike_ts')
stimulations_of_interest = stimulations.reset_index()
stimulations_of_interest = check_if_background(stimulations_of_interest, 'stimulation_ts').set_index('stimulation_idx')
stimulations_background = stimulations_of_interest[stimulations_of_interest['is_background'] == True]\
                        .drop(['is_background'], axis=1).reset_index(drop=True).pipe(rename_index, 'stimulation_idx')
spikes_of_interest_background = spikes_of_interest[spikes_of_interest['is_background'] == True]\
                        .drop(['is_background'], axis=1)

# extract and align spikes from raw signal
sio_background_window_iloc = spikes_of_interest_background[["spike_ts"]]\
    .progress_apply(ts_to_idx_col_fd_min, args=(raw_data,), lower=lower, upper=upper, axis=0)\
    .progress_apply(bounds, axis=1, result_type='expand',lower=-lower, upper=upper )\
    .rename(columns={0:'start_iloc', 1:'end_iloc'})

sio_background_raw = sio_background_window_iloc[['start_iloc','end_iloc']]\
    .progress_apply(extract_raw_values, args=(raw_data,), axis=1, result_type='expand')

maxima = sio_background_raw.loc[:, 15:29].max(axis=1).tolist()
print(sorted(maxima))
# read in threshold from config file based on tracked spikes 
threshold =  snakemake.params.threshold

# add latency in ms 
spikes_of_interest_background["latency"] = -1
spikes_of_interest_background["stimulation_ts"] = -1
for index, row in stimulations_background.iterrows():
    stim_onset = row['stimulation_ts']
    matching_rows = spikes_of_interest_background[(spikes_of_interest_background['spike_ts'] > stim_onset) \
                                       & (spikes_of_interest_background['spike_ts'] <= stim_onset + 4)]
    if len(matching_rows) > 0:
        latency = abs(stim_onset-matching_rows.spike_ts) * 1000
        spikes_of_interest_background.loc[matching_rows.index[0], 'latency'] = round(float(latency.iloc[0]), 4)
        spikes_of_interest_background.loc[matching_rows.index[0], 'stimulation_ts'] = stim_onset
print(spikes_of_interest_background)

# if the latency jump/difference is greater than the threshold value, the speed of fiber correction slows down
# and activity-dependent slowing was observed, this threshold is fiber dependent 
# limit search space
latency_difference = 0.5
indices_latency_jumps = spikes_of_interest_background["latency"].diff()[spikes_of_interest_background["latency"].diff() > latency_difference].index

# determine the start of the signal segment to know where the search region begins
onsets_segments_jumps = [spikes_of_interest_background.loc[index]['stimulation_ts'] for index in indices_latency_jumps]

# collect time tanges of signal segments
time_ranges = []
stimulation_onsets = []

# for each segment with latency jump greater than threshold 
for onset in onsets_segments_jumps:
        index = stimulations_background.index[stimulations_background['stimulation_ts'] == onset].tolist()
        time_ranges.append((stimulations_background.loc[index[0]-1]['stimulation_ts'],
                           stimulations_background.loc[index[0]]['stimulation_ts']))
        stimulation_onsets.append(stimulations_background.loc[index[0]-1]['stimulation_ts'])
        stimulation_onsets.append(stimulations_background.loc[index[0]]['stimulation_ts'])


# for evaluation: limit time changes, where actual stimuli were applied 
#time_range_check_dict = time_range_check(stimulations, time_ranges)
#time_ranges_filtered = [t for t in time_ranges if time_range_check_dict.get(t,0) == 1]
#time_ranges = time_ranges_filtered


# apply find_peaks for spike detection
crossing_indices_scipy = {}
times_list_scipy = {}
amplitude_list_scipy = {}
for index_time_range in range(len(time_ranges)):
    time_range_start = time_ranges[index_time_range][0]
    time_range_end = time_ranges[index_time_range][1]
    signal_piece = raw_data[time_range_start:time_range_end] 
    #signal_piece = pandas_gradient(signal_piece)
    peaks, _ = find_peaks(signal_piece, height = threshold, distance=15, prominence=0)
    crossing_indices_scipy[index_time_range] = peaks
    times_list_scipy[index_time_range] = signal_piece.index
    amplitude_list_scipy[index_time_range] = signal_piece.values 

# iterate over each crossing and remove artifacts 
spike_times = []
for k in crossing_indices_scipy:
    crossing_indices_scipy[k] = remove_stimulation_artifacts(crossing_indices_scipy[k],times_list_scipy[k], stimulations.values)
    for index in crossing_indices_scipy[k]:
        spike_times.append(times_list_scipy[k][index])

# save detected spike times in dataframe
spikes_detected = pd.DataFrame({"spike_ts":spike_times}).reset_index(drop=True).pipe(rename_index, 'spike_idx')

# evaluate spike detection
true_spikes_filtered = filter_spikes_in_ranges(spikes[spikes["track"]== track_of_interest_label], time_ranges)
results = {"dataset":dataset_name, "detected_spikes_count": len(spikes_detected)}
results_spike_detection = evaluate_detection(true_spikes_filtered, spikes_detected)
final_dict =results | results_spike_detection
df_result = pd.DataFrame.from_dict([final_dict])

# savd dataframes
spike_of_interest_times.to_csv(snakemake.output.spikes_of_interest_file)
spikes_of_interest.to_pickle(snakemake.output.spikes_of_interest_df)
spikes_detected.to_csv(snakemake.output.spikes_detected_file)
spikes_detected.to_pickle(snakemake.output.spikes_detected_df)
df_result.to_csv(snakemake.output.result_detection)
#fig.savefig(snakemake.output.template_threshold, dpi=300)
