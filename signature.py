import numpy as np
import pandas as pd
import scipy.io
import os
import h5py
from adcp import AdcpData, SignatureData

DATA_DIRECTORY = "data/Signature"
DATA_BASENAME = "kilpisjarvi"


def read_signature_data_from_mat_files(deployment_name, path='.'):
    files = list(
        filter(
            lambda x: x.endswith(".mat") and x.startswith(deployment_name),
            os.listdir(path=path)
        )
    )
    files.sort()
    index, cells = [], []
    i_index, i_cells, i_currents = [], [], []
    adcp_datasets = {}
    beams = None
    for file in files:
        print(file)
        data = scipy.io.loadmat(os.path.join(path, file))
        data = data["Data\x00\x00\x00\x00"][0][0]
        if not beams:
            beams = list(filter(lambda x: x.startswith("BurstHR_Vel"), data.dtype.names))
            for beam in beams:
                adcp_datasets[beam] = []
        index.append(data["BurstHR_TimeStamp"][:, 0])
        cells.append(data["BurstHR_Range"][0, :])
        for beam in beams:
            adcp_datasets[beam].append(data[beam])
        i_index.append(data["IBurstHR_TimeStamp"][:, 0])
        i_cells.append(data["IBurstHR_Range"][0, :])
        i_currents.append(data["IBurstHR_VelBeam5"])
    index = np.concatenate(index, axis=0)
    sort_pattern = np.argsort(index)
    index = np.take_along_axis(index, sort_pattern, axis=0)
    for beam in beams:
        adcp_datasets[beam] = np.concatenate(adcp_datasets[beam], axis=0)
        adcp_datasets[beam] = np.take_along_axis(adcp_datasets[beam], sort_pattern[:, None], axis=0)
    cells = cells[0]
    i_index = np.concatenate(i_index, axis=0)
    i_currents = np.concatenate(i_currents, axis=0)
    sort_pattern = np.argsort(i_index)
    i_index = np.take_along_axis(i_index, sort_pattern, axis=0)
    i_currents = np.take_along_axis(i_currents, sort_pattern[:, None], axis=0)
    i_cells = i_cells[0]
    data = {f"beam{i+1}":AdcpData(index, cells, adcp_datasets[beam]) for i, beam in enumerate(sorted(beams))}
    data[f"beam{len(beams)+1}"] = AdcpData(i_index, i_cells, i_currents)
    return SignatureData(data)


def mat2hdf5(data=None, filename="signature.h5"):
    if not data:
        data = read_signature_data_from_mat_files(DATA_BASENAME, DATA_DIRECTORY)
    with h5py.File(filename, "w") as f:
        beams = data.beams
        for beam in beams:
            group = f.create_group(beam)
            group.create_dataset("timestamps", data=beams[beam].index)
            group.create_dataset("cells", data=beams[beam].cells)
            group.create_dataset("currents", data=beams[beam].currents)


if __name__ == "__init__":
    mat2hdf5()
