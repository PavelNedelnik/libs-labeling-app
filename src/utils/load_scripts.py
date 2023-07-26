import numpy as np
import json
import h5py
from typing import Union, Tuple, Optional
from pathlib import Path
from utils.app_modes import App_modes


def contains_files(path, file_name_list):
    if not path.is_dir():
        return False
    contents = map(lambda f: f.name, list(path.iterdir()))
    return all(name in contents for name in file_name_list)



def is_numpy_dataset(path):
    return contains_files(path, ['X.npy', 'wavelengths.npy', 'dim.json'])


def is_h5_dataset(path):
    try:
        return path.suffix == '.h5'
    except:
        return False


def is_libs_dataset(path):
    if not path.is_dir():
        return False
    contents = map(lambda f: f.suffix, list(path.iterdir()))
    return '.libsdata' in contents and '.libsmetadata' in contents


def load_data():
    path = Path('../data')
    for file_path in path.iterdir():
        if is_numpy_dataset():
            print('Recognized as <numpy dataset>. Loading...')
            X, y, wavelengths, dim = load_npy_dataset(file_path)
            print('Loading done! Launching the application...')
            return X, y, wavelengths, dim
        elif is_h5_dataset(file_path):
            print('Recognized as <h5 dataset>...')
            X, y, wavelengths, dim = load_h5_dataset(file_path)
            print('Loading done! Launching the application...')
            return X, y, wavelengths, dim
        elif is_libs_dataset(file_path):
            print('Recognized as <libs dataset>...')
            X, y, wavelengths, dim = load_libs_dataset(file_path)
            print('Loading done! Launching the application...')
            return X, y, wavelengths, dim
    raise RuntimeError('Dataset not recognized as any of the supported formats.')


def load_npy_dataset(dataset_path: Path):
    dim = json.load(open(dataset_path / 'dim.json', 'rb'))
    print('Dimension loaded...')
    X = np.load(open(dataset_path / 'X.npy', 'rb'))
    print('Spectra loaded...')
    
    X.resize(dim + (wavelengths.shape[0],))
    X[::2, :] = X[::2, ::-1]
    
    # labels
    try:
        y = np.array(json.load(open(dataset_path / 'y.json', 'rb')))
        print('True labels loaded...')
    except:
        y = None
        print('No true labels found! Skipping...')

    wavelengths = np.load(open(dataset_path / 'wavelengths.npy', 'rb'))
    print('Wavelengths loaded...')

    return X, y, wavelengths, dim


def load_libs_dataset(dataset_path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    for f in dataset_path.glob('**/*.libsdata'):
        try:
            meta = json.load(open(f.with_suffix('.libsmetadata'), 'r'))
            print('Recognized .libsdata and .libsmetadata pair...')
        except:
            print('[WARNING] Failed to load metadata for file {}. Skipping!'.format(f))
            continue

        dim = [int(meta['spectra'] + 1), int(meta['wavelengths'])]
        print('Dimension loaded...')
        X = np.fromfile(open(f, 'rb'), dtype=np.float32)
        X = np.reshape(X, (int(meta['spectra'] + 1), int(meta['wavelengths'])))
        X[::2, :] = X[::2, ::-1]
        print('Spectra loaded...')
        y = None  # TODO
        print('No true labels found! Skipping...')
        wavelengths, X = X[0], X[1:]
        print('Wavelengths loaded...')

        return X, y, wavelengths, dim
    raise RuntimeError("Failed to load. No valid .libsdata and .libsmetadata found")


def load_h5_dataset(dataset_path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    f = h5py.File(dataset_path, "r")

    f = f[list(f.keys())[0]]
    f = f[list(f.keys())[0]]
    f = f['libs']
    dim = max(f['metadata']['X']) + 1, max(f['metadata']['Y']) + 1
    print('Dimension loaded...')
    X = np.array(f['data'])
    print('Spectra loaded...')
    y = None  # TODO
    print('No true labels found! Skipping...')
    wavelengths = np.array(f['calibration'])
    print('Wavelengths loaded...')

    return X, y, wavelengths, dim
