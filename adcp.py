import numpy as np
import pandas as pd
import scipy.io
import os
import h5py


class AdcpData:
    def __init__(
            self,
            index,
            cells,
            currents,
            phi=None,
            theta=None,
            isdetrended=False
            ):
        self.index = index
        self.cells = cells
        self.currents = currents
        self.phi = phi
        self.theta = theta
        self.isdetrended = isdetrended

    def __repr__(self):
        return pd.DataFrame(
            index=self.index,
            columns=self.cells,
            data=self.currents,
            ).__repr__()

    def detrend(
            self,
            window='100T',
            inplace=False
            ):
        if self.isdetrended:
            raise Exception("Data is already tetrended")
        df = pd.DataFrame(data=self.currents, index=self.index, columns=self.cells)
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

    def resample(self, window):
        data = pd.DataFrame(
                index=self.index,
                columns=self.cells,
                data=self.currents,
                )
        data = data.resample(window).mean()
        return AdcpData(
                index=data.index.values,
                cells=data.columns.values,
                currents=data.values,
                isdetrended=self.isdetrended,
                )

    def get_structure_function(
            self,
            reference_point=1,
            window="100T",
            ):
        reference_point_id = np.argmin(np.abs(self.cells - reference_point))
        data = pd.DataFrame(
                index=self.index,
                columns=np.subtract(self.cells, self.cells[reference_point_id, None]),
                data=np.subtract(self.currents, self.currents[:, reference_point_id, None])**2,
                )
        return data.rolling(window).mean()

    def get_epsilon(
            self,
            reference_point,
            window,
            ):
        structure_function = self.get_structure_function(reference_point=reference_point, window=window)
        r = np.abs(structure_function.columns.values)**(2/3)
        epsilon_func = lambda y: (np.dot(r, y) / np.dot(r, r) / 2.09)**1.5
        return structure_function.apply(epsilon_func, axis=1)


class SignatureData:
    def __init__(self, **beams):
        self.beams = beams


def read_adcp_data(filename, beam):
    with h5py.File(filename) as data_file:
        group = data_file[beam]
        time_shift = group["timestamps"][0] % 1
        index = group["timestamps"][:] - time_shift
        index = pd.to_datetime(index, unit='s').values
        return AdcpData(
                index=index,
                cells=group["cells"][:],
                currents=group["currents"][:],
                )
