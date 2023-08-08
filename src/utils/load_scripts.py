import numpy as np
import json
import h5py
import tkinter
import tkinter.filedialog
from typing import Union, Tuple, Optional
from pathlib import Path
from utils.app_modes import App_modes


def contains_files(path, file_name_list):
    contents = list(map(lambda f: f.name, list(path.iterdir())))
    return all(name in contents for name in file_name_list)



def is_numpy_dataset(path):
    return contains_files(path, ['X.npy', 'wavelengths.npy', 'dim.json'])


def is_h5_dataset(path):
    contents = list(map(lambda f: f.suffix, list(path.iterdir())))
    return '.h5' in contents


def is_libs_dataset(path):
    contents = list(map(lambda f: f.suffix, list(path.iterdir())))
    return '.libsdata' in contents and '.libsmetadata' in contents


def prompt_file():
    top = tkinter.Tk()
    top.withdraw()
    file_name = tkinter.filedialog.askdirectory(parent=top)
    top.destroy()
    return file_name


def load_data():
    print('Please, select a folder containing the data to be analyzed...', end='', flush=True)
    path = Path(prompt_file())
    print(' Done!', flush=True)
    if is_numpy_dataset(path):
        print('Recognized as <numpy dataset>. Loading...', flush=True)
        X, y, wavelengths, dim = load_npy_dataset(path)
        print('Loading done! Launching the application...', flush=True)
        return X, y, wavelengths, dim
    elif is_h5_dataset(path):
        print('Recognized as <h5 dataset>...', flush=True)
        X, y, wavelengths, dim = load_h5_dataset(path)
        print('Loading done! Launching the application...', flush=True)
        return X, y, wavelengths, dim
    elif is_libs_dataset(path):
        print('Recognized as <libs dataset>...', flush=True)
        X, y, wavelengths, dim = load_libs_dataset(path)
        print('Loading done! Launching the application...', flush=True)
        return X, y, wavelengths, dim


def load_npy_dataset(dataset_path: Path):
    print('    Loading dimensions...', end='', flush=True)
    dim = json.load(open(dataset_path / 'dim.json', 'rb'))
    print(' Done!', flush=True)


    print('    Loading spectra...', end='', flush=True)
    X = np.load(open(dataset_path / 'X.npy', 'rb'))
    print(' Done!', flush=True)
    
    print('    Loading wavelengths...', end='', flush=True)
    wavelengths = np.load(open(dataset_path / 'wavelengths.npy', 'rb'))
    print(' Done!', flush=True)

    print('    Reshaping spectra...', end='', flush=True)
    X.resize(dim + [wavelengths.shape[0]])
    X[::2, :] = X[::2, ::-1]
    print(' Done!', flush=True)
    
    # labels
    print('    Loading true labels...', end='', flush=True)
    try:
        y = np.array(json.load(open(dataset_path / 'y.json', 'rb')))
        print(' Done!', flush=True)
    except:
        y = None
        print(' No true labels found! Skipping!', flush=True)

    return X, y, wavelengths, dim


def load_libs_dataset(dataset_path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    for f in dataset_path.glob('**/*.libsdata'):
        try:
            meta = json.load(open(f.with_suffix('.libsmetadata'), 'r', encoding='utf8'))
            print('    Recognized .libsdata and .libsmetadata pair...', flush=True)
        except:
            print('\n[WARNING] Failed to load metadata for file {}! Skipping!'.format(f), flush=True)
            continue

        try:
            print('    Loading dimensions...', end='', flush=True)
            try:
                dim = [int(meta['yPosCount']), int(meta['xPosCount'])]
            except KeyError:
                print('\n[WARNING] Dimensions could not be automatically deduced!')
                dim = input('Please, input the x dimension'), input('Please, input the y dimension')
            print(' Done!', flush=True)

            print('    Loading spectra...', end='', flush=True)
            X = np.fromfile(open(f, 'rb'), dtype=np.float32)
            print(' Done!', flush=True)

            print('    Loading wavelegnths...', end='', flush=True)
            X = np.reshape(X, (int(meta['spectra']) + 1, int(meta['wavelengths'])))
            wavelengths, X = X[0], X[1:]
            print(' Done!', flush=True)

            print('    Reshaping spectra...', end='', flush=True)
            X = np.reshape(X, dim + [-1])
            X[::2, :] = X[::2, ::-1]
            print(' Done!', flush=True)

            print('    Loading true labels...', end='', flush=True)
            y = None  # TODO
            print(' No true labels found! Skipping...', flush=True)

            return X, y, wavelengths, dim
        except:
            print('\n[WARNING] Failed to load! Skipping!'.format(f), flush=True)
            continue
    raise RuntimeError("Failed to load! No valid .libsdata and .libsmetadata found!")


def load_h5_dataset(dataset_path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    for file_path in dataset_path.glob('**/*.h5'):
        try:
            f = h5py.File(file_path, "r")
            f = f[list(f.keys())[0]]
            f = f[list(f.keys())[0]]
            f = f['libs']
            print('    Loading dimensions...', end='', flush=True)
            dim = [max(f['metadata']['X']) + 1, max(f['metadata']['Y']) + 1]
            print(' Done!', flush=True)

            print('    Loading spectra...', end='', flush=True)
            X = f['data'][()]
            print(' Done!', flush=True)

            print('    Loading wavelengths...', end='', flush=True)
            wavelengths = np.array(f['calibration'])
            print(' Done!', flush=True)

            print('    Reshaping spectra...', end='', flush=True)
            X = np.reshape(X, dim + [-1])
            X[::2, :] = X[::2, ::-1]
            print(' Done!', flush=True)

            print('    Loading true labels...', end='', flush=True)
            y = None  # TODO
            print(' Done!', flush=True)

            return X, y, wavelengths, dim
        except:
            print('\n[WARNING] Failed to load. Skipping!'.format(f), flush=True)
            continue
    raise RuntimeError('Failed to load! No valid file found!')
