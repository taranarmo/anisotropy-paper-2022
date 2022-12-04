import numpy as np
import pandas as pd
from math import sin, cos, radians
import scipy.io
import os
import h5py


class AdcpData:
    def __init__(
            self,
            index,
            cells,
            currents,
            phi=None, # Orientation
            theta=None, # Vertical angle
            isdetrended=False,
            coordinates='beam',
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
            raise Exception("Data is already detrended")
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
    def __init__(
            self,
            beams,
            heading=None,
            pitch=None,
            roll=None,
            theta=radians(25),
            angle_units='radians'
            ):
        self.beams = beams
        if angle_units=='radians':
            self.heading = heading
            self.pitch = pitch
            self.roll = roll
            self.theta = theta
        else:
            self.heading = radians(heading)
            self.pitch = radians(pitch)
            self.roll = radians(roll)
            self.theta = radians(theta)

    def get_xyz(self):
        vx = (self.beams['beam1'].currents - self.beams['beam3'].currents)/2 * sin(self.theta)
        vy = (self.beams['beam2'].currents - self.beams['beam4'].currents)/2 * sin(self.theta)
        vz = sum([
            self.beams['beam1'].currents,
            self.beams['beam2'].currents,
            self.beams['beam3'].currents,
            self.beams['beam4'].currents,
            ]) / 4 * cos(self.theta)
        return [
            AdcpData(
                index=self.beams['beam1'].index,
                cells=self.beams['beam1'].cells,
                currents=currents,
                ) for currents in (vx, vy, vz)]

    def get_tke(self, detrend_window='100T', resample_window='1T'):
        enu = [x.resample(resample_window).detrend(detrend_window) for x in self.get_xyz()]
        tke = pd.DataFrame(
                data=sum([x.currents**2 for x in enu]),
                index=enu[0].index,
                columns=enu[0].cells,
                )
        return tke


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
