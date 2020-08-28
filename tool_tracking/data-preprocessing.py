# --- third-party ---
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
mytool = "electric_screwdriver"
# mytool = "pneumatic_screwdriver"
# mytool = "pneumatic_rivet_gun"
firstTimeStamp = np.Inf

mdr = MeasurementDataReader(source="/home/a/Desktop/SoSe_2020/tool-tracking/tool-tracking-data")
data_dict = (
    mdr.query(query_type=Measurement)
    .filter_by(Tool == mytool, DataTypes == [ACC, GYR, MIC, MAG])
    .get()
)

measurement_campaign = "01"
acc = pd.DataFrame(data_dict.get(measurement_campaign).acc)
gyr = pd.DataFrame(data_dict.get(measurement_campaign).gyr)
mic = pd.DataFrame(data_dict.get(measurement_campaign).mic)
mag = pd.DataFrame(data_dict.get(measurement_campaign).mag)


def combine_sensors(reference_data, others, firstTimeStamp=np.Inf):
    """combine dataframes of data frome different sensors.
    Resample all entries to the frequency of reference_data.
    """
    # firstTimeStamp = np.Inf
    for sensor in [reference_data] + others:
        firstTimeStamp = min(firstTimeStamp, sensor["time [s]"].min())
    #     firstTimeStamp = np.min((firstTimeStamp, timestamps.min()))

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


data_df, firstTimeStamp = combine_sensors(acc, [gyr])


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


data = data_df.values
window_size = 60
data_windowed = extract_same_label(data, window_size)

X_windowed = data_windowed[:, :, :-1]
y_windowed = data_windowed[:, :, -1]
