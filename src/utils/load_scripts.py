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
    file_name = tkinter.filedialog.askopenfilename(parent=top)
    top.destroy()
    return file_name


def run_wizard():
    print('Please, select a file containing the data to be analyzed...', end='', flush=True)
    path = Path(prompt_file())
    print(' Done!', flush=True)
    if path.suffix == '.npy':
        print('Recognized as <numpy dataset>. Loading...', flush=True)
        X, wavelengths, dim = load_npy_dataset(path)
    elif path.suffix == '.h5':
        print('Recognized as <h5 dataset>...', flush=True)
        X, wavelengths, dim = load_h5_dataset(path)
    elif path.suffix == '.libsdata':
        print('Recognized as <libs dataset>...', flush=True)
        X, wavelengths, dim = load_libs_dataset(path)
    else:
        raise RuntimeError('Not recognized as any of the supported formats!')
    
    y = None
    prompt = 'Would you like to include ground truth labels? [y]es or [n]o: '
    while input(prompt).lower() == 'y':
        label_path = Path(prompt_file())
        print('    Loading true labels...', end='', flush=True)
        try:
            y = np.array(json.load(open(label_path, 'rb')))
            print(' Done!', flush=True)
            break
        except Exception as e:
            print(' loading failed with error: {}'.format(e), flush=True)
            prompt = 'Would you like to retry? [y]es or [n]o'
    
    print('Loading done! Launching the application...', flush=True)
    return X, y, wavelengths, dim


def load_npy_dataset(path: Path):
    try:
        dataset_path = path.parents[0]

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
    except Exception as e:
        raise RuntimeError('\nFailed to load data from directory {} with error message: {}.'.format(dataset_path, e))

    return X, wavelengths, dim


def load_libs_dataset(path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    try:
        meta_path = path.with_suffix('.libsmetadata')
        meta = json.load(open(meta_path, 'r', encoding='utf8'))
        print('    Recognized .libsdata and .libsmetadata pair...', flush=True)
    except Exception as e:
        raise RuntimeError('\nOpening metadata file {} ends with error: {}'.format(meta_path, e))

    try:
        print('    Loading dimensions...', end='', flush=True)
        try:
            try:
                xs, ys = list(zip(*[(int(m['x']), int(m['y'])) for m in meta['data']]))
                dim = [max(xs) - min(xs), max(ys) - min(ys)]
            except KeyError:
                xs, ys = list(zip(*[(int(m['X']), int(m['Y'])) for m in meta['data']]))
                dim = [max(xs) - min(xs), max(ys) - min(ys)]
        except Exception as e:
            print('\nFailed to load dimensions with error message: {}'.format(e))
            dim = [int(input('Please, manually input the x dimension: ')), int(input('Please, manually input the y dimension: '))]
        print(' Done!', flush=True)

        print('    Loading spectra...', end='', flush=True)
        X = np.fromfile(open(path, 'rb'), dtype=np.float32)
        print(' Done!', flush=True)

        print('    Loading wavelegnths...', end='', flush=True)
        X = np.reshape(X, (int(meta['spectra']) + 1, int(meta['wavelengths'])))
        wavelengths, X = X[0], X[1:]
        print(' Done!', flush=True)

        print('    Reshaping spectra...', end='', flush=True)
        X = np.reshape(X, dim + [-1])
        X[::2, :] = X[::2, ::-1]
        print(' Done!', flush=True)
    except Exception as e:
        raise RuntimeError('\nFailed to load file {} with error message: {}.'.format(path, e))
    
    return X, wavelengths, dim


def load_h5_dataset(path: Path) -> Tuple[np.array, Optional[np.array], np.array, list]:
    try:
        f = h5py.File(path, "r")
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
    except Exception as e:
        raise RuntimeError('\nFailed to load file {} with error message: {}.'.format(path, e))

    return X, wavelengths, dim