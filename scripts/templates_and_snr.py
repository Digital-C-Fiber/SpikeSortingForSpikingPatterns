from snakemake.script import snakemake
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import stats
import matplotlib
from helper import *
tqdm.pandas()

# read in input files
raw_data = pd.read_pickle(snakemake.input.raw_data)
ap_track_window = pd.read_pickle(snakemake.input.ap_track_window)
ap_derivatives = pd.read_pickle(snakemake.input.ap_derivatives)
ap_window_iloc = pd.read_pickle(snakemake.input.ap_window_iloc)
spikes = pd.read_pickle(snakemake.input.spikes)
stimulations = pd.read_pickle(snakemake.input.stimulations)
track_of_interest_label = snakemake.params.track_of_interest_label
name = snakemake.params.name

# color list to extract colors 
colors = ["tab:blue", "tab:green","tab:orange", "tab:red","tab:cyan", "tab:brown", "tab:pink", "tab:olive"]
track_names = sorted(ap_track_window['track'].unique())
# if error, new colors needs to be added
assert len(track_names) < len(colors)
track_colors = {track_names[i]:colors[i] for i in range(len(track_names))}


# plot tempaltes of tracks
def plot_template(row, ax1, track_colors):
    template = pd.Series(row["template"], index=[round(x, 4) for x in np.arange(0, len(row["template"]) * 0.0001, 0.0001)])
    ax1.plot([round(x, 4) for x in np.arange(0.0001, (len(template) * 0.0001 + 0.0001), 0.0001)], template.values,
         c=track_colors[row.track], alpha=1, linestyle="dashed",
         label="Template of " + row.track)
    ax1.legend(fontsize=25)
    print(name)
    ax1.set_title(name, fontsize=33)
    ax1.set_xlabel("Time (s)", fontsize=28)
    ax1.set_ylabel(u'Voltage (${\mu}V$)', fontsize=28)
    ax1.set_xticks([0.000, 0.001, 0.002, 0.003])
    ax1.xaxis.set_tick_params(labelsize=27)
    ax1.yaxis.set_tick_params(labelsize=27)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)


matplotlib.rcParams["axes.formatter.useoffset"] = False

# get piece of noise based on first stimulation stimulus
def get_noise_segment(raw_data, first_stimulus, first_spike, time_offset1=0.015, time_offest2=0.041):
    data_piece = raw_data[first_stimulus+time_offset1:first_stimulus+time_offset1+ time_offest2]
    return data_piece

# compute SNR with MAD estimation
def compute_snr_from_arrays(spike_waveform, noise_segment):
    spike_amplitude = np.max(spike_waveform)  
    noise_std = stats.median_abs_deviation(noise_segment) 
    snr = spike_amplitude / noise_std if noise_std > 0 else np.inf
    return snr

# create templates for each track
ap_templates = compute_templates(ap_track_window, raw_data)

# plot templates
fig = plt.figure(figsize=(10,8), dpi=300)
plt.rcParams["font.size"] = 10
ax1 = fig.add_subplot(111)
ap_templates.progress_apply(plot_template, args=(ax1,track_colors), axis = 1 , result_type="expand")
fig.tight_layout()

# compute snr from track-of-interest template 
template_spike = np.array(ap_templates[ap_templates["track"] == track_of_interest_label].template.values[0])
first_spike = spikes.iloc[0].spike_ts
first_stimulus = stimulations.iloc[0]
noise_piece = get_noise_segment(raw_data, first_stimulus, first_spike)
snr = round(compute_snr_from_arrays(template_spike, noise_piece.values),2)
mid_point_noise = int(len(noise_piece.values)/2)

# plot noise and spike template
fig_snr, ax_snrs = plt.subplots(1,1, dpi=300, figsize=(10,8))
ax_snr = ax_snrs
#ax_snr1 =ax_snrs[1]
ax_snr.plot(noise_piece.values, label="Noise", c="tab:blue")
ax_snr.plot(np.arange(mid_point_noise-15,mid_point_noise+15, 1),template_spike, label="Template", c="tab:orange")
#ax_snr1.plot(np.arange(mid_point_noise-15,mid_point_noise+15, 1), noise_piece.values[mid_point_noise-15: mid_point_noise+15], label="Noise", c="tab:blue")
#ax_snr1.plot(np.arange(mid_point_noise-15,mid_point_noise+15, 1),template_spike, label="Template", c="tab:orange")

#for ax_snr in ax_snrs:
ax_snr.set_xlabel("Datapoint", fontsize=28)
ax_snr.set_ylabel(u'Voltage (${\mu}V$)', fontsize=28)
ax_snr.xaxis.set_tick_params(labelsize=27)
ax_snr.yaxis.set_tick_params(labelsize=27)
ax_snr.legend(fontsize=25)
ax_snr.set_title(f"{name}, {snr}", fontsize= 33)
ax_snr.spines["top"].set_visible(False)
ax_snr.spines["right"].set_visible(False)

# save SNR values in dataframe 
df_snr = pd.DataFrame([{"Dataset": name, "snr": snr}])

# save dataframes 
df_snr.to_csv(snakemake.output.snr_file, index=False)
fig.savefig(snakemake.output.template_figure, dpi=300)
fig_snr.savefig(snakemake.output.snr_figure, dpi=300)
ap_templates.to_pickle(snakemake.output.ap_templates)
