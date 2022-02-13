import numpy as np
import pandas as pd
import scipy.io
import os
import h5py


class AdcpData:
    def __init__(self, index, cells, currents, phi=None, theta=None):
        self.index = index
        self.cells = cells
        self.currents = currents
        self.phi = phi
        self.theta = theta


class SignatureData:
    def __init__(self, **beams):
        self.beams = beams
