# --- third-party ---
import os
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tabulate

pd.options.display.max_columns = 15
# --- own ---

from datatools import (
    MeasurementDataReader,
    Tool,
    Config,
    MeasurementSeries,
    Measurement,
    DataTypes,
    Action,
    to_ts_data,
)
from datatools import ACC, GYR, MAG, MIC, POS, VEL
from fhgutils import contextual_recarray_dtype


def combine_sensors(reference_data, others, firstTimeStamp=np.Inf):
    """combine dataframes from different sensors.
    Resample all entries to the frequency of reference_data.
    Fills N/As as nearest occurrence timewise
    """
    # firstTimeStamp = np.Inf
    for sensor in [reference_data] + others:
        firstTimeStamp = min(firstTimeStamp, sensor["time [s]"].min())

    res = reference_data.copy()
    for df in others:
        res = pd.merge_ordered(
            res,
            df,
            left_on="time [s]",
            right_on="time [s]",
            #direction="nearest",
        )
        res = res.loc[:, ~res.columns.duplicated()]
        res["label"] = np.where(np.isnan(res["label_x"]),res["label_y"],res["label_x"])
        del res["label_x"]
        del res["label_y"]
    res = res.loc[res["label"] != -1]
    res = res.loc[res["label"] !=  8]
    res = res.rename({res.columns[0]: "time"}, axis="columns")
    res["time"] = res["time"] - firstTimeStamp
    res = res.fillna(0)
    return res, firstTimeStamp


def extract_same_label(data, window_size):
    """
    :param data: 2d np matrix with label as last column
    :param window_size: Number of timestamps per window
    :return: 3d np array with shape (#windows, #stamps, #features+label)
    Only returns windows, where each stamp has the same class
    """
    res = np.empty((0, window_size, data.shape[1]))
    i = 0
    while i + window_size < data.shape[0]:
        candidate = data[i: i + window_size]
        labels = candidate[:, -1]
        if len(np.unique(labels)) == 1:
            i += window_size
            res = np.concatenate((res, candidate.reshape((1, candidate.shape[0], candidate.shape[1]))))
        else:
            i += np.asarray(labels != labels[0]).nonzero()[0][0]
    return res


def vote_label_time(window, majorityPortion=0.5):
    """
    Parameters
    ----------
    window : a window of data with shape (#stamps,#features)
    majorityPortion : how much is required to define majority

    Returns
    -------
    the label that occured most in that window
    """
    labels = np.unique(window[:, -1])
    timePerLabel = np.empty((0, 2))
    diffs = np.vstack((np.diff(window[:, 0]), window[:-1, -1])).T
    for l in labels:
        t = sum(diffs[diffs[:, -1] == l, 0])
        timePerLabel = np.vstack((timePerLabel, np.array([t, l]).reshape(1, -1)))
    if np.max(timePerLabel[:, 0]) > majorityPortion * sum(timePerLabel[:, 0]):
        return timePerLabel[timePerLabel[:, 0] == np.max(timePerLabel[:, 0]), -1].astype(int)[0]
    else:
        return -2  # inconsistent


def extract_by_time(data, length, overlap):
    """
    Parameters
    ----------
    data : 2d np matrix with label as last column
    length : length of time for each window in seconds
    overlap : portion of overlap between the windows

    Returns
    -------
    Two lists: (due to inconsistent amount of timestamps in each window)
    1st : list made of windows as np. matrices
    2nd : list with majority labels in each window
    """
    res = []
    res_labels = []
    maxStamp = max(data[:, 0])
    currentStamp = min(data[:, 0])
    # lastStamp = 0
    while currentStamp + length <= maxStamp:
        candidate = data[np.logical_and((data[:, 0] < currentStamp + length), (data[:, 0] >= currentStamp))]
        if candidate.size > 0:
            if vote_label_time(candidate) != -2:
                res = res + [candidate[:, :-1]]
                res_labels = res_labels + [vote_label_time(candidate)]
        currentStamp = data[data[:, 0] >= currentStamp + (1 - overlap) * length][0, 0]
        # lastStamp = currentStamp + (1-overlap)*length
        # currentStamp = currentStamp + (1-overlap)*length
    return res, res_labels


mytool = "electric_screwdriver"
# mytool = "pneumatic_screwdriver"
# mytool = "pneumatic_rivet_gun"
data_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tool-tracking-data")
# data_filepath = "./tool-tracking-data"
mdr = MeasurementDataReader(source=data_filepath)
data_dict = (mdr.query(query_type=Measurement).filter_by(Tool == mytool, DataTypes == [ACC, GYR, MIC, MAG]).get())

relabel_dict = {  # Labels are shared across all tools, 'holes' in both pneumatics
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    8: 0,  # undefined AKA nothing is happening
    14: 7, 25: 8, 28: 9, 29: 10, 30: 11, 38: 12}

# relabel_dict = {  # Each label has its own labels, no 'holes'
#     8: 0, 7: 1,  # undefined and shaking exist for all tools
#     2: 2,3: 3,4: 4,5: 5, #  shared across both screwdrivers
#     25: 2, 28: 3, 29: 4, 30: 5  # rivet gun
#     6: 6,  # manual_motor_rotation for electric screwdriver
#     38: 6,  # impact exists for both pneumatics
#     14: 7}  # tightening_double from both screwdrivers

relabel_actions = np.vectorize(lambda x: relabel_dict[x])

firstTimeStamp = np.Inf
window_size = 20  #
nr_features = 12  # hardcoded, idk. 2 + ACC:3, GYR:3, MIC:1, MAG:3

data_windowed = np.empty((0, window_size, nr_features))
data = np.empty((0, nr_features))
for measurement_campaign in ["01", "02", "03", "04"]:  # All recordings
# for measurement_campaign in ["02"]:  # Only 1st recording
    print(measurement_campaign)
    acc = pd.DataFrame(data_dict.get(measurement_campaign).acc)
    acc = acc.loc[acc["label"] != 8]
    gyr = pd.DataFrame(data_dict.get(measurement_campaign).gyr)
    gyr = gyr.loc[gyr["label"] != 8]
    mic = pd.DataFrame(data_dict.get(measurement_campaign).mic)
    mic = mic.loc[mic["label"] != 8]
    # mic = mic.iloc[::10,]
    mag = pd.DataFrame(data_dict.get(measurement_campaign).mag)
    mag = mag.loc[mag["label"] != 8]

    # data_df, firstTimeStamp = combine_sensors(acc, [gyr], firstTimeStamp)  # Only ACC and GYR sensors
    data_df, firstTimeStamp = combine_sensors(acc, [gyr,mag,mic], firstTimeStamp)  # All sensors, downsampled
    # data_df, firstTimeStamp = combine_sensors(mic, [acc,gyr,mag], firstTimeStamp)  # All sensors, upsampled
    data_df['label'] = relabel_actions(data_df['label'])
    data = np.vstack((data, data_df.values))
    data_windowed = np.concatenate((data_windowed, extract_same_label(data_df.values, window_size)))


X_windowed = data_windowed[:, :, :-1]
y_windowed = data_windowed[:, 0, -1].astype(int)
y_windowed_1hot = np.eye(max(y_windowed) + 1)[y_windowed]
pickle.dump(X_windowed, open('../X_winsize_20.pickle', 'wb'))
pickle.dump(y_windowed, open('../y_winsize_20.pickle', 'wb'))
pickle.dump(y_windowed_1hot, open('../y1hot_winsize_20.pickle', 'wb'))


data_60 = extract_same_label(data,60)
X_windowed = data_60[:, :, :-1]
y_windowed = data_60[:, 0, -1].astype(int)
y_windowed_1hot = np.eye(max(y_windowed) + 1)[y_windowed]
pickle.dump(X_windowed, open('../X_winsize_60.pickle', 'wb'))
pickle.dump(y_windowed, open('../y_winsize_60.pickle', 'wb'))
pickle.dump(y_windowed_1hot, open('../y1hot_winsize_60.pickle', 'wb'))

data_100 = extract_same_label(data,100)
X_windowed = data_100[:, :, :-1]
y_windowed = data_100[:, 0, -1].astype(int)
y_windowed_1hot = np.eye(max(y_windowed) + 1)[y_windowed]
pickle.dump(X_windowed, open('../X_winsize_100.pickle', 'wb'))
pickle.dump(y_windowed, open('../y_winsize_100.pickle', 'wb'))
pickle.dump(y_windowed_1hot, open('../y1hot_winsize_100.pickle', 'wb'))

# run for loop before that
window_length = 20  # seconds
overlap = 0.5  # portion
# bottom are lists because windows have differing amount of stamps
X_windowed_time, y_windowed_time = extract_by_time(data, window_length, overlap)
y_windowed_time_1hot = np.eye(max(y_windowed_time) + 1)[y_windowed_time]
