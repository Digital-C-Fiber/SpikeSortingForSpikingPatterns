from snakemake.script import snakemake
import pandas as pd
import numpy as np
from math import e, log, sqrt
from scipy.stats import moment
from statistics import stdev
from scipy.signal import resample
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()


raw_data = pd.read_pickle(snakemake.input.raw_data)
spikes = pd.read_pickle(snakemake.input.spikes)
ap_track_window = pd.read_pickle(snakemake.input.ap_track_window)
ap_derivatives = pd.read_pickle(snakemake.input.ap_derivatives)
ap_window_iloc= pd.read_pickle(snakemake.input.ap_window_iloc)
dataset_name = snakemake.params.name

# compute components 
def __lasttrue(series:pd.Series) -> float:
    last_true_at = None
    for idx,v in series.items():
        if v:
            last_true_at = idx
    if last_true_at == None:
        last_true_at = 0
    return last_true_at


# updated function to find principal points for SS-SPDF method, can fail due to spike shape
# and the first derivative
def __calc_components(name, fd, crossings, raw_index, raw):
    p2 = fd.iloc[20:40].idxmin()
    p1,p3,p4,p5,p6 = None,None,None,None,None
    try:
        p1 = __lasttrue(crossings.iloc[:fd.index.get_loc(p2)])
        try:
            p3 = crossings.iloc[fd.index.get_loc(p2)+1:].idxmax()
            try:
                p5 = crossings.iloc[fd.index.get_loc(p3)+1:].idxmax()
                try:
                    p4 = fd.iloc[fd.index.get_loc(p3)+1:fd.index.get_loc(p5)].idxmax()
                except Exception as e:
                    print("error in p4", e)
                    pass
                try:
                    p6 = fd.iloc[fd.index.get_loc(p5)+1:fd.index.get_loc(p5)+10].idxmin()
                except:
                    print("error in p6")
                    pass
            except:
                print("error in p5")
                pass
        except:
            print("error in p3")
            pass
    except Exception as e:
        #print(e)
        print("error in p1")
        pass
    return [p1,p2,p3,p4,p5,p6]


# method to start the principal point computation
def calculate_components(row, data:pd.Series, idx_start_iloc='start_iloc', idx_end_iloc='end_iloc', idx_fd='fd', idx_fd_crossings='fd_crossings'):
    ap_start = row[idx_start_iloc]
    ap_end = row[idx_end_iloc]
    raw = data.iloc[ap_start:ap_end]
    fd = pd.Series(row[idx_fd])
    crossings = pd.Series(row[idx_fd_crossings])#, index=raw.index)
    return __calc_components(row.name,fd,crossings, raw.index, raw) 

# helper to compute mean change between two values
def __mean_change_between(series:pd.Series, x:float, y:float) -> float:
    return (series.at[x] - series.at[y]) / (x-y)

# helper to compute root mean square of series 
def __calculate_rms(series:pd.Series) -> float:
    summed_series = series.map(lambda x: x * x).sum()
    div = series.index[-1] - series.index[0]
    return sqrt(summed_series / div)

# helper to compute slope ratio of AP
def __slope_ratio(series:pd.Series, a:float,b:float,c:float) -> float:
    part_a = (series.at[b] - series.at[a]) / (b-a)
    part_b = (series.at[c] - series.at[b]) / (c-b)
    return part_a/part_b

# helper to compute inter quantile range
def __iqr(series:pd.Series) -> float:
    Q3 = np.quantile(series, 0.75)
    Q1 = np.quantile(series, 0.25)
    return Q3-Q1

# helper function to compute moment
def __sampling_moment_dev(vals: pd.Series, n: int) -> float:
    return moment(vals, n) / pow(stdev(vals), n)

# feature definitions
def __feature_calc(fd:pd.Series, sd:pd.Series, p1:float,p2:float,p3:float,p4:float,p5:float,p6:float):
    f = [None]*24

    p1_loc,p5_loc= fd.index.get_indexer([p1,p5], method='nearest')
    # shape based features
    f[0] = p5-p1
    f[1] = fd.at[p4]-fd.at[p2]
    f[2] = fd.at[p6]-fd.at[p2]
    # f[3] skip F4 as we have redefined it and will calculate it at another point
    f[4] = log(abs(__mean_change_between(fd, p4,p2)),e)
    f[5] = __mean_change_between(fd,p6,p4)
    try:
        f[6] = log(abs(__mean_change_between(fd,p6,p2)),e)
    except:
        print("F6 error")
        pass
    f[7] = __calculate_rms(fd.iloc[:p1_loc+1])
    f[8] = __slope_ratio(fd,p1,p2,p3)
    f[9] = __slope_ratio(fd,p3,p4,p5)
    f[10] = fd.at[p2]/fd.at[p4]

    # phase based features
    for i,p in enumerate((p1,p3,p4,p5,p6)):
        f[11+i] = fd.at[p]
    for i,p in enumerate((p1,p3,p5)):
        f[16+i] = sd.at[p]

    # distribution based features
    ## not sure it should not be p6
    for i,ser in enumerate((fd,sd)):
        f[19+i] = __iqr(ser.loc[p1:p5])
    f[21] = __sampling_moment_dev(fd.loc[p1:p5],4)
    f[22] = __sampling_moment_dev(fd.loc[p1:p5],3)
    f[23] = __sampling_moment_dev(sd.loc[p1:p5],3)
    return f

# method to start feature computation
def calculate_features(row, data, idx_start_iloc='start_iloc', idx_end_iloc='end_iloc', idx_p1='P1', idx_p2='P2', idx_p3='P3', idx_p4='P4', idx_p5='P5', idx_p6='P6', idx_fd="fd", idx_sd='sd'):
    ap_start = row[idx_start_iloc]
    ap_end = row[idx_end_iloc]
    p1 = row[idx_p1]
    p2 = row[idx_p2]
    p3 = row[idx_p3]
    p4 = row[idx_p4]
    p5 = row[idx_p5]
    p6 = row[idx_p6]
    raw = data.iloc[ap_start:ap_end]
    fd = pd.Series(row[idx_fd])#, index=raw.index)
    sd = pd.Series(row[idx_sd])#, index=raw.index)
    return __feature_calc(fd,sd,p1,p2,p3,p4,p5,p6)

feature_descriptors = {f'F{i}':desc for i,desc in (
    (1,'waveform duration'),
    (2,'FD peak-to-valley amplitude'),
    (3,'FD valley-to-valley amplitude'),
    (4,''),
    (5,'Natural logarithm of the FDs positive deflection'),
    (6,'FD negative deflection'),
    (7,'Natural logarithm of the FDs slope among valleys'),
    (8,'RMS of the FDs pre-AP amplitudes'),
    (9,'FD negative slope ratio'),
    (10,'FD positive slope ratio'),
    (11,'FD peak-to-valley ratio'),
    (12,'FD amplitude at P1'),
    (13,'FD amplitude at P3'),
    (14,'FD amplitude at P4'),
    (15,'FD amplitude at P5'),
    (16,'FD amplitude at P6'),
    (17,'SD amplitude at P1'),
    (18,'SD amplitude at P3'),
    (19,'SD amplitude at P5'),
    (20,'FD IQR'),
    (21,'SD IQR'),
    (22,'FD Kurtosis coefficient'),
    (23,'FD Fisher asymmetry'),
    (24,'SD Fisher asymmetry')
)}


# plot feature values 
def iter_axes(axs):
    if not hasattr(axs, '__iter__'):
        yield axs
    else:
        for ax in axs:
            yield from iter_axes(ax)
    
# create df with components
components = ap_derivatives\
                    .join(ap_window_iloc)\
                    .progress_apply(calculate_components, args=(raw_data,), axis=1, result_type="expand")\
                    .rename(columns={i:f"P{i+1}" for i in range(6)})


# drop na
components.dropna(inplace=True)

# create df with SS-SPDF features
features_spdf = components\
                    .join(ap_derivatives)\
                    .join(ap_window_iloc)\
                    .progress_apply(calculate_features, args=(raw_data,), axis=1, result_type="expand")\
                    .rename(columns={i:f"F{i+1}" for i in range(24)})\
                    .drop(columns=["F4"])
features_spdf = features_spdf[~features_spdf.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
features_spdf_fv3 = features_spdf[['F14', 'F18', 'F19']]

# create df with raw features 
features_raw = pd.DataFrame(ap_derivatives["raw"].to_list(),  index=ap_derivatives.index,
                            columns=np.arange(0,len(ap_derivatives["raw"].iloc[0])))

# save length per feature
dataframe_list = ['spikes', 'features_spdf', 'features_raw']
length_list  = [len(spikes), len(features_spdf), len(features_raw)]
try:
    track_names = sorted(ap_track_window['track'].unique())
    for track in track_names:
        df_temp_track = spikes[spikes["track"] == track]
        dataframe_list.append(track)
        length_list.append(len(df_temp_track))
except KeyError:
    print("no tracks in ap_track")
df_lengths = {
    'DataFrame': dataframe_list,
    'Length': length_list
}
lengths_df = pd.DataFrame(df_lengths)

# save all dataframes 
lengths_df.to_csv(snakemake.output.length_df, index=False)
feature_name = snakemake.wildcards.feature
if feature_name == "spdf_fv3":
    features_spdf_fv3.to_pickle(snakemake.output.features)
elif feature_name == "spdf":
    features_spdf.to_pickle(snakemake.output.features)
elif feature_name == "w_raw":
    features_raw.to_pickle(snakemake.output.features)