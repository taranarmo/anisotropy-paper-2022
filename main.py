import numpy as np
import pandas as pd
import scipy.io
import os
import h5py
import datetime
import matplotlib.pyplot as plt
from signature import SignatureData, AdcpData

DATA_DIRECTORY = "data"


def matlab_ts2python_time(matlab_timestamp):
    """
    Converts MATLAB timestamp to python datetime object
    """
    result = (
            datetime.datetime.fromordinal(int(matlab_timestamp)) +
            datetime.timedelta(days=matlab_timestamp % 1) -
            datetime.timedelta(days = 366)
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


def get_buoyancy_flux(radiation_data):
    """
    Calculates buoyancy flux based on the solar radiation data, currently without thermistor chain data
    """
    depth05 = 2.2       # m
    gamma = 0.3         # 1/m
    rad0 = radiation_data / np.exp(-gamma*depth05);
    delta = 1           # m, approximate, needs T-chain data
    hmix = 9            # m, approximate, needs T-chain data 
    sw_alpha = 1.65e-5 # sw_alpha = sw_alpha(0,1,0)
    beta = sw_alpha * 9.81 / 4.18e6
    buoyancy_flux = beta*(radiation_data*np.exp(-gamma*delta)+radiation_data*np.exp(-gamma*hmix) -2/hmix*radiation_data*(np.exp(-gamma*delta)-np.exp(-gamma*hmix)))
    return buoyancy_flux


radiation_data = get_radiation_data('13-May-2019 18:00:00', '17-May-2019 00:00:00')
buoyancy_flux = get_buoyancy_flux(radiation_data)
