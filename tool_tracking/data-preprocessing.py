# --- third-party ---
import os

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
        res = pd.merge_asof(
            res,
            df,
            left_on="time [s]",
            right_on="time [s]",
            direction="nearest",
        )
    res = res.loc[:, ~res.columns.duplicated()]
    res["label"] = res["label_x"]
    res = res.rename({res.columns[0]: "time"}, axis="columns")
    del res["label_x"]
    del res["label_y"]
    res = res.loc[res["label"] != -1]

    res = res.loc[
        res["label"] != 8
    ]  # Comment out if you want "undefined" as a class

    res["time"] = res["time"] - firstTimeStamp
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
        candidate = data[i : i + window_size]
        labels = candidate[:, -1]
        if len(np.unique(labels)) == 1:
            i += window_size
            res = np.concatenate(
                (
                    res,
                    candidate.reshape(
                        (1, candidate.shape[0], candidate.shape[1])
                    ),
                )
            )
        else:
            i += np.asarray(labels != labels[0]).nonzero()[0][0]
    return res

mytool = "electric_screwdriver"
# mytool = "pneumatic_screwdriver"
# mytool = "pneumatic_rivet_gun"
data_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tool-tracking-data")
# data_filepath = "./tool-tracking-data"
mdr = MeasurementDataReader(source=data_filepath)
data_dict = (mdr.query(query_type=Measurement).filter_by(Tool == mytool, DataTypes == [ACC, GYR, MIC, MAG]).get())
firstTimeStamp = np.Inf


window_size = 20  # 60 is too big, @data_folder/info.md
nr_features = 12  # hardcoded, idk. 2 + ACC:3, GYR:3, MIC:1, MAG:3

data_windowed = np.empty((0,window_size,nr_features))
for measurement_campaign in ["01","02","03","04"]:  # All recordings
# for measurement_campaign in ["01"]:  # Only 1st recording
    acc = pd.DataFrame(data_dict.get(measurement_campaign).acc)
    gyr = pd.DataFrame(data_dict.get(measurement_campaign).gyr)
    mic = pd.DataFrame(data_dict.get(measurement_campaign).mic)
    mag = pd.DataFrame(data_dict.get(measurement_campaign).mag)
    # data_df, firstTimeStamp = combine_sensors(acc, [gyr], firstTimeStamp)  # Only ACC and GYR sensors
    data_df, firstTimeStamp = combine_sensors(acc, [gyr,mic,mag], firstTimeStamp)  # All sensors, downsampled
    # data_df, firstTimeStamp = combine_sensors(mic, [acc,gyr,mag], firstTimeStamp)  # All sensors, upsampled
    data = data_df.values
    data_windowed = np.concatenate((data_windowed, extract_same_label(data, window_size)))

X_windowed = data_windowed[:, :, :-1]
y_windowed = data_windowed[:, 0, -1].astype(int)
y_windowed_1hot = np.eye(max(y_windowed)+1)[y_windowed]
