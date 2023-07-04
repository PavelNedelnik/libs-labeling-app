import numpy as np


def load_toy_dataset():
    dim = (70, 70)
    calibration = np.load(open('data/calibration.npy', 'rb'))  # wavelengths
    X = np.load(open('data/X.npy', 'rb'))  # measured data, dimensions are (index of measurement, wavelength)

    # make hyperspectral map
    X.resize(dim + (calibration.shape[0],))
    # input data has snake index
    X[::2, :] = X[::2, ::-1]

    return X, np.zeros(dim), calibration, dim