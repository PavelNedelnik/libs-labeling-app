import numpy as np
import json
import h5py
from typing import Union, Tuple, Optional
from pathlib import Path
from utils.app_modes import App_modes


def contains_files(path, file_name_list):
    if not path.is_dir():
        return False
    contents = list(map(lambda f: f.name, list(path.iterdir())))
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
    contents = list(map(lambda f: f.suffix, list(path.iterdir())))
    return '.libsdata' in contents and '.libsmetadata' in contents


def load_data(path: Tuple[str, Path]):
    if not isinstance(path, Path):
        path = Path(path)
    for file_path in path.iterdir():
        if is_numpy_dataset(file_path):
            print('Recognized as <numpy dataset>. Loading...', flush=True)
            X, y, wavelengths, dim = load_npy_dataset(file_path)
            print('Loading done! Launching the application...', flush=True)
            return X, y, wavelengths, dim
        elif is_h5_dataset(file_path):
            print('Recognized as <h5 dataset>...', flush=True)
            X, y, wavelengths, dim = load_h5_dataset(file_path)
            print('Loading done! Launching the application...', flush=True)
            return X, y, wavelengths, dim
        elif is_libs_dataset(file_path):
            print('Recognized as <libs dataset>...', flush=True)
            X, y, wavelengths, dim = load_libs_dataset(file_path)
            print('Loading done! Launching the application...', flush=True)
            return X, y, wavelengths, dim
    raise RuntimeError('Dataset not recognized as any of the supported formats.')


def load_npy_dataset(dataset_path: Path):
    dim = json.load(open(dataset_path / 'dim.json', 'rb'))
    print('Dimension loaded...', flush=True)
    X = np.load(open(dataset_path / 'X.npy', 'rb'))
    print('Spectra loaded...', flush=True)
    
    wavelengths = np.load(open(dataset_path / 'wavelengths.npy', 'rb'))
    print('Wavelengths loaded...', flush=True)
    X.resize(dim + [wavelengths.shape[0]])
    X[::2, :] = X[::2, ::-1]
    print('Spectra loaded...', flush=True)
    
    # labels
    try:
        y = np.array(json.load(open(dataset_path / 'y.json', 'rb')))
        print('True labels loaded...', flush=True)
    except:
        y = None
        print('No true labels found! Skipping...', flush=True)

    return X, y, wavelengths, dim


def load_libs_dataset(dataset_path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    for f in dataset_path.glob('**/*.libsdata'):
        try:
            meta = json.load(open(f.with_suffix('.libsmetadata'), 'r', encoding='utf8'))
            print('Recognized .libsdata and .libsmetadata pair...', flush=True)
        except:
            print('[WARNING] Failed to load metadata for file {}. Skipping!'.format(f), flush=True)
            continue

        dim = [int(meta['yPosCount']), int(meta['xPosCount'])]
        print('Dimension loaded...', flush=True)
        X = np.fromfile(open(f, 'rb'), dtype=np.float32)
        print('Data loaded...', flush=True)
        X = np.reshape(X, (int(meta['spectra']) + 1, int(meta['wavelengths'])))
        wavelengths, X = X[0], X[1:]
        print('Wavelengths loaded...', flush=True)
        X = np.reshape(X, dim + [-1])
        X[::2, :] = X[::2, ::-1]
        print('Data reshaped...', flush=True)
        print('Spectra loaded...', flush=True)
        y = None  # TODO
        print('No true labels found! Skipping...', flush=True)

        return X, y, wavelengths, dim
    raise RuntimeError("Failed to load. No valid .libsdata and .libsmetadata recognized.")


def load_h5_dataset(dataset_path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    f = h5py.File(dataset_path, "r")

    f = f[list(f.keys())[0]]
    f = f[list(f.keys())[0]]
    f = f['libs']
    dim = [max(f['metadata']['X']) + 1, max(f['metadata']['Y']) + 1]
    print('Dimension loaded...', flush=True)
    X = f['data'][()]
    X = np.reshape(X, dim + [-1])
    X[::2, :] = X[::2, ::-1]
    print('Spectra loaded...', flush=True)
    y = None  # TODO
    print('No true labels found! Skipping...', flush=True)
    wavelengths = np.array(f['calibration'])
    print('Wavelengths loaded...', flush=True)

    return X, y, wavelengths, dim
