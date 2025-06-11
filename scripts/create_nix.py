import neo
from neo.io import NixIO
import quantities as pq 
import pandas as pd
import numpy as np
from datetime import datetime as dt

# pad signal as neo expects a continuous recording 
def pad_signal(raw_data):
    sampling_interval = 0.0001  # 10 kHz
    start_time = raw_data.index[0]
    time_index = np.array(raw_data.index, dtype=np.float64)
    sample_numbers = np.round((time_index - start_time) / sampling_interval).astype(int)
    num_samples = sample_numbers[-1] + 1
    filled_values = np.zeros(num_samples, dtype=np.float32)
    filled_values[sample_numbers] = raw_data.values
    filled_time_index = np.round(start_time + np.arange(num_samples) * sampling_interval, 6)
    filled_ts = pd.Series(data=filled_values, index=filled_time_index, name='raw_data')
    return filled_ts

# create new neo block with old neo block 
def create_neo_block(neo_block, raw_data):
    neo_block_new = neo.Block("block", description="Data stored in integers to avoid floating point percision problems. This works, because Dapsys seems to export the data in integers to begin with")
    neo_segment_new = neo.Segment("segment")
    for train in neo_block.segments[0].spiketrains:
        neo_segment_new.spiketrains.append(train)
    neo_segment_new.events.append(neo_block.segments[0].events[0])
    raw_data_filled = pad_signal(raw_data)
    neo_segment_new.analogsignals.append(neo.AnalogSignal(raw_data_filled.values, units=pq.V, t_start=raw_data_filled.index[0] * pq.s, file_origin="data", sampling_rate=10000*pq.Hz))
    neo_block_new.segments.append(neo_segment_new)
    return neo_block_new


