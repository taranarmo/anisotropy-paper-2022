import numpy as np
import pandas as pd
import scipy.io
import os
import h5py
import datetime
import json
import sqlite3
import adcp
import matplotlib.pyplot as plt

DATA_DIRECTORY = "data"
SIGNATURE_DATA_FILE = "signature.h5"
START_DATE= "2019-05-14"
END_DATE = "2019-05-16"


def matlab_ts2python_time(matlab_timestamp):
    """
    Converts MATLAB timestamp to python datetime object
    """
    result = (
            datetime.datetime.fromordinal(int(matlab_timestamp)) +
            datetime.timedelta(days=matlab_timestamp % 1) -
            datetime.timedelta(days=366)
    )
    return result


def get_radiation_data(start_datetime=None, end_datetime=None):
    """
    Reads solar radiation data
    """
    original_data = scipy.io.loadmat(os.path.join(DATA_DIRECTORY, "rad05.mat"), squeeze_me=True)
    dates = pd.to_datetime(list(map(matlab_ts2python_time, original_data["date05"])))
    data = original_data["rad05"]
    return pd.Series(data, index=dates.round("T")).loc[start_datetime: end_datetime]


def get_temperature_data(data_directory):
    def find_rbr_file(rbr_id, extenstion='.rsk', data_directory=data_directory):
        names = filter(lambda x: x.startswith(rbr_id) and x.endswith(extenstion), os.listdir(data_directory))
        filename = next(names)
        try:
            filename = next(names)
        except StopIteration:
            return filename
        raise Exception("more than 1 file with such id in directory")

    with open(os.path.join(data_directory, "setup.json")) as f:
        metadata = json.load(f)
    data = []
    positions = []
    distances = []
    for device in metadata["devices"]:
        filename = find_rbr_file(device["id"])
        with sqlite3.connect(os.path.join(data_directory, filename)) as conn:
            data.append(pd.read_sql("select tstamp, channel01 from data;", conn).set_index("tstamp"))
        positions.append(device["position"])
        distances.append(device["distance_to_previous"])
    permutation_index = np.argsort(positions)
    distances = np.cumsum(np.array(distances)[permutation_index])
    data = pd.concat({key:value for key, value in zip(distances, np.array(data, dtype="object")[permutation_index])}, axis=1)
    data.index = pd.to_datetime(data.index, unit="ms")
    data.index.name = ""
    return data.droplevel(level=1, axis=1)


def get_buoyancy_flux(radiation_data):
    """
    Calculates buoyancy flux based on the solar radiation data, currently without thermistor chain data
    """
    def B(z):
        sw_alpha = 1.65e-5  # sw_alpha = sw_alpha(0,1,0)
        beta = sw_alpha * 9.81 / 4.18e6
        return beta

    sw_alpha = 1.65e-5  # sw_alpha = sw_alpha(0,1,0)
    beta = sw_alpha * 9.81 / 4.18e6
    depth05 = 2.2       # m
    gamma = 0.3         # 1/m
    rad0 = radiation_data / np.exp(-gamma*depth05);
    delta = 1           # m, approximate, needs T-chain data
    hmix = 9            # m, approximate, needs T-chain data
    buoyancy_flux = beta*(radiation_data*np.exp(-gamma*delta)+radiation_data*np.exp(-gamma*hmix) -2/hmix*radiation_data*(np.exp(-gamma*delta)-np.exp(-gamma*hmix)))
    return buoyancy_flux


def main():
    radiation_data = get_radiation_data(START_DATE, END_DATE)
    buoyancy_flux = get_buoyancy_flux(radiation_data)
    beams = [f"beam{i}" for i in (1,2)]
    fig, ax = plt.subplots(figsize=(7, 5))
    buoyancy_flux.rolling("100T").mean().plot(ax=ax)
    for beam in beams:
        currents_data = adcp.read_adcp_data(SIGNATURE_DATA_FILE, beam)
        dissipation_rate = currents_data.resample("T").detrend("10T").get_epsilon(window="10T", reference_point=0.25)
        dissipation_rate = dissipation_rate.loc[START_DATE:END_DATE]
        dissipation_rate.rolling("100T").mean().plot(ax=ax)
    ax.legend(ax.lines, ["B", *[f"ϵ, {beam}" for beam in beams]])
    plt.savefig(f"beams_and_epsilon.png")
    plt.close()

if __name__ == "__main__":
    main()
