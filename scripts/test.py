import pandas as pd

feature = pd.read_pickle("workflow/features/background_all/spdf_fv3/A1_features.pkl")
feature2 = pd.read_pickle("workflow/dataframes/background_all/A1_ap_derivatives.pkl")
#feature2 = pd.read_pickle("workflow/dataframes/detected/A1_ap_track_window.pkl")
#print(feature.index)
#print(feature2.index)
#print(feature.join(feature2))

df_all = pd.read_pickle("workflow/features/background_all/w_raw/A1_features.pkl")
df = pd.read_pickle("workflow/dataframes/background_all/A1_ap_track_window.pkl")
#print("raw", df)
track_of_interest_label = "Track1"
import quantities
def rename_index(pd_obj, new_name: quantities.second):
    return pd_obj.reindex(pd_obj.index.rename(new_name))

df_spikes = df[df["track"] == track_of_interest_label ].reset_index(drop=False).pipe(rename_index, 'spike_idx')
df_spikes = df_spikes.rename(columns={"spike_idx":"old_spike_idx"})
#print(df_spikes['old_spike_idx'])
#print(df)
#print(df.loc[df_spikes['old_spike_idx']])

# df = pd.read_pickle(r"workflow/features/background_all/spdf/A4_features.pkl")
# print(df)
# import numpy as np
# is_infinite = df.isin([np.inf, -np.inf])
# print(df.isna().any())
# res = is_infinite.any().any()
# print(res)

df_all = pd.read_pickle("workflow/dataframes/A6_spikes.pkl")
df_background = pd.read_pickle("workflow/dataframes/background_all/A6_spikes.pkl")
track_names = sorted(df_all['track'].unique())

#for track in track_names:
df_temp_track_background = df_background[df_background["track"] == "Track1"]
df_temp_track = df_all[df_all["track"] == "Track1"]

print(pd.read_pickle("workflow/dataframes/background_all/AC_ap_track_window.pkl"))

