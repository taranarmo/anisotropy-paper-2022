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
from functools import partial
from scipy.interpolate import interp1d
from math import exp

DATA_DIRECTORY = "data"
SIGNATURE_DATA_FILE = "signature.h5"
START_DATE = "2019-05-14"
END_DATE = "2019-05-16"
EXPERIMENT_TIMEFRAME = slice(START_DATE, END_DATE)

GRAVITATIONAL_ACCELERATION = 9.8067

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
        raise Exception("more than 1 file with the same id in directory")

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
    distances = np.cumsum(np.array(distances)[permutation_index]) / 1e2
    data = pd.concat({key:value for key, value in zip(distances, np.array(data, dtype="object")[permutation_index])}, axis=1)
    data.index = pd.to_datetime(data.index, unit="ms")
    data.index.name = ""
    data = data.droplevel(level=1, axis=1)
    return data.resample('T').mean()


def get_cml_boundaries(temperature_data, mixing_threshold=1e-2):
    pattern = np.gradient(temperature_data) < mixing_threshold
    boundaries = {
            "upper": temperature_data[pattern].index.min(),
            "lower": temperature_data[pattern].index.max(),
    }
    return boundaries


def static_cml_boundaries(upper=0, lower=5):
    return {'upper': upper, 'lower': lower}


def get_buoyancy_flux(
        radiation_data,
        temperature_data,
        integration_step=0.1,
        ice_transparency=0.32,
        gamma=0.3,
        mixing_threshold=1e-2,
        ):
    """
    Calculates buoyancy flux based on the solar radiation data, currently without thermistor chain data
    """
    # sw_alpha = 1.65e-5  # sw_alpha = sw_alpha(0,1,0)
    # beta = sw_alpha * 9.81 / 4.18e6
    # depth05 = 2.2       # m
    GAMMA = gamma         # 1/m
    radiation_data = radiation_data.resample('T').mean().dropna()
    radiation_data = radiation_data * ice_transparency / 4.18e6 # 1 W/m2 ≈ 4.18 μmole * m2/s
    temperature_data = temperature_data.resample('T').mean().dropna()
    temperature_data, radiation_data = temperature_data.align(radiation_data, join='inner', axis=0)

    def I(z, radiation_data=radiation_data, gamma=gamma):
        return radiation_data * np.exp(-gamma*z)

    def beta(temperature_data, depths_range, z, Tr = 277):
        T = np.interp(z, depths_range, temperature_data)
        return 1.65e-5 * GRAVITATIONAL_ACCELERATION * (T - Tr)

    integral_buoyancy_flux = {}
    for timestamp, temperature in temperature_data.iterrows():
        boundaries = get_cml_boundaries(
                temperature_data=temperature_data.loc[timestamp, :],
                mixing_threshold=mixing_threshold
                )
        integral_buoyancy_flux[timestamp] = sum([
            beta(temperature, temperature.index, boundaries["upper"]) *
                I(boundaries["upper"], radiation_data=radiation_data.loc[timestamp], gamma=gamma),
            beta(temperature, temperature.index, boundaries["lower"]) *
                I(boundaries["lower"], radiation_data=radiation_data.loc[timestamp], gamma=gamma),
            -sum(
                (
                    beta(temperature, temperature.index, z) * I(z, radiation_data=radiation_data.loc[timestamp], gamma=gamma)
                    for z in np.arange(boundaries["upper"], boundaries["lower"], integration_step)
                )
            ) * integration_step * 2 / (boundaries["lower"] - boundaries["upper"])
        ])
    return pd.Series(integral_buoyancy_flux)


def main():
    radiation_data = get_radiation_data(START_DATE, END_DATE)
    temperature_data = get_temperature_data('data/T_chain')
    temperature_data = temperature_data.loc[EXPERIMENT_TIMEFRAME]
    buoyancy_flux = get_buoyancy_flux(
            temperature_data=temperature_data,
            radiation_data=radiation_data,
            gamma=.2,
    )
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
    plt.contourf(temperature_data.index.values, temperature_data.columns.values, temperature_data.T)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig('temperatures.png')
    plt.close()

if __name__ == "__main__":
    main()
