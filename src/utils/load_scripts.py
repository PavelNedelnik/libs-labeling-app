import numpy as np
import json
from typing import Union
from pathlib import Path
from utils.app_modes import App_modes


def load_toy_dataset():
    dim = (70, 70)
    X, _, calibration = load_npy_dataset('data/toy_dataset', dim)

    return X, None, calibration, dim, App_modes.Default


def load_contest_dataset():
    dim = (500, 500)
    X, y, calibration = load_npy_dataset('data/contest', dim)
    return X, y, calibration, dim, App_modes.Benchmark


def load_npy_dataset(dataset_path: Union[str, Path], dim):
    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)

    # wavelengths
    calibration = np.load(open(dataset_path / 'calibration.npy', 'rb'))
    X = np.load(open(dataset_path / 'X.npy', 'rb'))
    
    # measured data
    X.resize(dim + (calibration.shape[0],))  # expected dimensions are (index of measurement, wavelength)
    X[::2, :] = X[::2, ::-1]  # expects 2d grid data with snake index
    
    # labels
    y = np.array(json.load(open(dataset_path / 'y.json', 'rb')))

    return X, y, calibration

