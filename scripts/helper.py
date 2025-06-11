import pandas as pd
import numpy as np
from scipy.signal import resample
import typing 
import quantities as pq


# compute gradient
def pandas_gradient(series):
    #return series.diff()
    return pd.Series(np.gradient(series.values), index= series.index)

def compute_templates(ap_track_window, raw_data):
    ap_templates = pd.DataFrame()
    for track in ap_track_window['track'].unique():
        ap_track_window_sorted = ap_track_window[(ap_track_window['track'] == track)]
        print("track", track, len(ap_track_window_sorted))
        ap_raw = ap_track_window_sorted[['start_iloc','end_iloc']]\
            .progress_apply(extract_raw_values, args=(raw_data,), axis=1, result_type='expand')
        template = ap_raw.mean()
        data = pd.DataFrame({"track":track, "template" :[template.to_list()]})
        ap_templates = pd.concat([ap_templates, data])
    return ap_templates.reset_index(drop=True)

# extract raw values
def extract_raw_values(row, data:pd.Series, idx_window_start_iloc=0, idx_window_end_iloc=1):
    start = row[idx_window_start_iloc]
    end = row[idx_window_end_iloc]
    raw = data.iloc[start:end]
    return raw.to_list()

# align by negative peak of first derivative  
def ts_to_idx_col_fd_min(column, data:pd.Series, lower, upper):
    # index array holds the index of the spike and it should be the minimum 
    index_array = data.index.get_indexer(column, method='nearest')
    #print(index_array)
    for i in range(len(index_array)):
        bounds = [index_array[i]-lower, index_array[i]+upper]  
        bounds_arange = np.arange(index_array[i]-lower, index_array[i]+upper)  
        data_piece = data.iloc[bounds[0] :bounds[1]]
        upsampling_factor = 2
        data_piece_upsampled = pd.Series(resample(data_piece, upsampling_factor * len(data_piece)))
        fd_data_piece = pandas_gradient(data_piece_upsampled).iloc[1:]
        fd_min = fd_data_piece[5:25].argmin() + 5 + 1
        mid_index_temp = int(round(fd_min/upsampling_factor,0))
        mid_index = bounds_arange[mid_index_temp]
        # check if index array value aligns with min_index, if not replace it by min index to
        # ensure alignment of all spikes
        if not (index_array[i] ==  (mid_index)):
            index_array[i] = mid_index 
    return index_array


def ts_to_idx_col_max(column, data:pd.Series, lower, upper):
    # index array holds the index of the spike and it should be the minimum 
    index_array = data.index.get_indexer(column, method='nearest')
    #print(index_array)
    for i in range(len(index_array)):
        bounds = [index_array[i]-lower, index_array[i]+upper]  
        bounds_arange = np.arange(index_array[i]-lower, index_array[i]+upper)  
        #if not snakemake.params.use_bristol_processing:
        mid_index = bounds_arange[data.iloc[bounds[0] :bounds[1]].argmax()]-2 
        #else:
         #   mid_index = bounds_arange[data.iloc[bounds[0] :bounds[1]].argmin()]-2 
             # two points before max as alignment 
        # check if index array value aligns with min_index, if not replace it by min index to
        # ensure alignment of all spikes
        if not (index_array[i] ==  (mid_index)):
            index_array[i] = mid_index 
    return index_array

# helper function to get bounds of data slice
def bounds(row, idx_val:typing.Union[str,int]=0, lower:int=0,upper:int=0):
    v=row[idx_val]
    return [v+lower, v+upper]

# compute first and second derivative of signal
def calculate_fd_sd(row, data:pd.Series, idx_window_start_iloc=0, idx_window_end_iloc=1):
    start = row[idx_window_start_iloc]
    end = row[idx_window_end_iloc]
    raw = data.iloc[start:end]
    raw_upsampled = pd.Series(resample(raw, 60))
    fd = pandas_gradient(raw_upsampled)
    fd_zero = zerocrossings(fd)
    sd = pandas_gradient(fd)
    return [fd.iloc[1:].to_list(), sd.iloc[2:].to_list(), fd_zero.iloc[1:].to_list(), raw.values]

# compute zerocrossing
def zerocrossings(series: pd.Series) -> pd.Series:
    signs = np.sign(series.values)
    return pd.Series((signs[i] != signs[i - 1] for i in range(1,len(series))),index=series.index[1:])

# helper to rename index of dataframe or series 
def rename_index(pd_obj, new_name: pq.second):
    return pd_obj.reindex(pd_obj.index.rename(new_name))

# check if it is a background spike/stimulus based on previouse timing 
def check_if_background(df_timestamps, col, time_threshold=3.8):
    # get prev. and next value to compare spike onsets
    df_timestamps['prev'] = df_timestamps[col].shift(1)   
    df_timestamps['next'] = df_timestamps[col].shift(-1)  
    df_timestamps['is_background'] = True
    df_timestamps.loc[1:len(df_timestamps) - 2, 'is_background'] = ((abs(df_timestamps[col] - df_timestamps['prev']) > time_threshold) | 
                                    (abs(df_timestamps[col] - df_timestamps['next']) > time_threshold))
    df_timestamps.drop(['prev', 'next'], axis=1, inplace=True)
    return df_timestamps

