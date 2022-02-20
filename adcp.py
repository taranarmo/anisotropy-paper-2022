import numpy as np
import pandas as pd
import scipy.io
import os
import h5py


class AdcpData:
    def __init__(self, index, cells, currents, phi=None, theta=None, isdetrended=False):
        self.index = index
        self.cells = cells
        self.currents = currents
        self.phi = phi
        self.theta = theta
        self.isdetrended = isdetrended
    
    def detrend(self, window='100T', inplace=False):
        if self.isdetrended:
            raise Exception("Data is already tetrended")
        df = pd.DataFrame(data=self.currents, index=pd.to_datetime(self.index, unit='s'), columns=self.cells)
        currents = (df - df.rolling(window).mean()).values
        if inplace:
            self.currents = currents
            self.isdetrended = True
        else:
            return AdcpData(
                    index=self.index,
                    cells=self.cells,
                    currents=currents,
                    isdetrended=True
                    )


class SignatureData:
    def __init__(self, **beams):
        self.beams = beams


def read_adcp_data(filename, beam):
    with h5py.File(filename) as data_file:
        group = data_file[beam]
        return AdcpData(
                index=group["timestamps"][:],
                cells=group["cells"][:],
                currents=group["currents"][:],
                ) 
