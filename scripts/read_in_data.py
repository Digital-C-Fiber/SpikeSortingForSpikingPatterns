from snakemake.script import snakemake
from numba import njit
from numpy import float32, float64
import pydapsys
from pydapsys.file import File
from pydapsys.page import TextPage, WaveformPage
from pydapsys.toc.entry import StreamType, Stream, Folder
from pydapsys.neo_convert.ni_pulse_stim import NIPulseStimulatorToNeo
import pandas as pd
import matplotlib.pyplot as plt

import quantities as pq
import numpy as np
import neo 
from neo.io import NixIO
from scipy.io import loadmat
from scripts.create_nix import create_neo_block

RECORDING_FILE = snakemake.input[0]

# methods to get continuous recording and return as Pandas series
@njit
def _kernel_offset_assign(target: np.array, calc_add, calc_mul, pos_offset, n):
    for i in range(n):
        target[pos_offset + i] = calc_add + i * calc_mul

def get_continuous_recording(file: File, path_dapsys) -> pd.Series:
    # it could be needed to adjust path variable and write NI Pulse Stimulator
    # or NI Puls Stimulator
    path = path_dapsys
    total_datapoint_count = sum(len(wp.values) for wp in file.get_data(path, stype=StreamType.Waveform))
    values = np.empty(total_datapoint_count, dtype=float32)
    timestamps = np.empty(total_datapoint_count, dtype=float64)
    current_pos = 0
    for wp in file.get_data(path, stype=StreamType.Waveform):
        wp: WaveformPage
        n = len(wp.values)
        values[current_pos:current_pos + n] = wp.values
        if wp.is_irregular:
            timestamps[current_pos:current_pos + n] = wp.timestamps
        else:
            _kernel_offset_assign(timestamps, wp.timestamps[0], wp.interval, current_pos, n)
        current_pos += n
    print("finished loading continuous recording")
    return pd.Series(data=values, index=pd.Index(data=timestamps, copy=False, name="raw_ts"),
                     name="raw_amplitude", copy=False)

# helper to rename index of dataframe or series 
def rename_index(pd_obj, new_name: pq.second):
    return pd_obj.reindex(pd_obj.index.rename(new_name))

# read in spike trains
def spike_train_to_pd_series(train: neo.SpikeTrain):
    return pd.Series( train.name,index=train.as_array().flatten(),name='track')\
            .pipe(rename_index, 'spike_ts')

# read in comments
def comments_to_pd_series(event: neo.Event):       
    return pd.Series( event.labels,index=event,name='text')\
            .pipe(rename_index, 'comment_ts')

def flatten_ndarray(arr: np.ndarray):
    for row in arr:
        yield row[0]

#  read in file as neo segment
def get_neo_segment(filepath, block_idx, segment_idx):
    with NixIO(filename=filepath, mode='ro') as nxio:
        neo_structure = nxio.read()
    return neo_structure[block_idx].segments[segment_idx]

# transfrom neo data as pandas series 
def get_neo_as_pd_series(neo_data, name="raw_amplitude", index_name="raw_ts"):
    return pd.Series(neo_data.as_array().flatten(), index=np.asarray(neo_data.times) , name=name)\
          .pipe(rename_index, index_name)

# internal Dapsys path and time range of the recording
PATH_DAPSYS = snakemake.params.path_dapsys
time1 = snakemake.params.time1
time2 = snakemake.params.time2

# read in dapsys file and convert to Neo block
with open(RECORDING_FILE, 'rb') as file:
    file = File.from_binary(file)
    neo_block = NIPulseStimulatorToNeo(file, grouping_tolerance=1e-9).to_neo()

# create Series with raw_data
raw_data = get_continuous_recording(file, PATH_DAPSYS)
    # create df with stimulation times
stimulations = pd.Series(neo_block.segments[0].events[0].as_array().flatten(), name='stimulation_ts')\
                .pipe(rename_index, 'stimulation_idx')

# create df with spike times
spikes = pd.concat(spike_train_to_pd_series(train) for train in neo_block.segments[0].spiketrains)\
    .sort_index()\
    .to_frame()\
    .reset_index()\
    .pipe(rename_index, 'spike_idx')

spikes = spikes[(spikes['spike_ts'] >= time1) & (spikes['spike_ts'] <= time2)]\
            .reset_index(drop=True)\
            .pipe(rename_index, 'spike_idx')

track_names = sorted(spikes['track'].unique())


stimulations = stimulations[(stimulations >= time1) & (stimulations <= time2)]\
            .reset_index(drop=True)\
            .pipe(rename_index, 'stimulation_idx')

# use neo to create nix file
neo_block_new = create_neo_block(neo_block, raw_data)
with NixIO(filename=snakemake.output.nix_file, mode='ow') as nxio:
    nxio.write(neo_block_new)

# save dataframes 
raw_data.to_pickle(snakemake.output.raw_data)
stimulations.to_pickle(snakemake.output.stimulations)
spikes.to_pickle(snakemake.output.spikes)