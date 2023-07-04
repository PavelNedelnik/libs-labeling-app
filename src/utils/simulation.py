import warnings
import numpy as np
from SimulatedLIBS import simulation
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d

PARAMS = dict(
    Te=1.0,
    Ne=10**18,
    resolution=1000,
    low_w=200,
    upper_w=1000,
    max_ion_charge=3,
    webscraping='dynamic',
)

def get_spectra(n, elements, background_fluctuation, save=False):
    spectra_path = f'simulated_data/{elements}_{n}.npy'
    calibration_path = f'simulated_data/{elements}_{n}_calibration.npy'
    try:
        resampled_spectra = np.load(open(spectra_path, 'rb'), allow_pickle=True) # TODO allow pickle ?
        master = np.load(open(calibration_path, 'rb'))
    except OSError:
        if not input("Data not found. Input empty string to dowload the data."):
            raw = list(map(
                lambda x: get_spectrum(
                    elements=elements,
                    percentages=[60 + x[1], 30 - x[1], 5, x[0], 5-x[0]],
                    **PARAMS,
                ),
                list(zip(np.linspace(0, 5, n), background_fluctuation))
            ))
            
            spectra, wavelengths = list(zip(*raw))
            master = max(wavelengths, key=len)

            resampled_spectra = np.stack([np.interp(master, w, s) for s, w in zip(spectra, wavelengths)], axis=0)
            if save:
                np.save(open(spectra_path, 'wb'), resampled_spectra)
                np.save(open(calibration_path, 'wb'), master)
    return resampled_spectra, master


def get_spectrum(return_wavelength=True, *args, **kwargs):
    spectrum = simulation.SimulatedLIBS(
        *args,
        **kwargs,
    ).get_raw_spectrum()
    if return_wavelength:
        return spectrum['intensity'].to_numpy(), spectrum['wavelength'].to_numpy()
    return spectrum['intensity'].to_numpy()


def generate_map(n, elements, seed_array, background_fluctuation, cache=False, kernel=None, smooth_kernel=np.ones(3) / 3, noise_var=None, noise_mean=None, boundary_size=None):
    """
    seed array: in  the shape of the final hyperspectral image. -1 -> unseeded, i -> i-th element
    """
    # spectra palette
    sps, calibration = get_spectra(n, elements, background_fluctuation, save=cache)

    # populating the image
    if kernel is None:
        kernel = np.asarray([
            [1, 1, 1],
            [1, 8, 1],
            [1, 1, 1],
        ], dtype=float)
        kernel /= kernel.sum()

    zero_img = np.zeros(seed_array.shape)
    zero_img[seed_array == 1] = 1.
    zero_img[seed_array != 1] = 0.

    for i in range(100):
        zero_img = convolve2d(zero_img, kernel, mode='same', boundary='fill', fillvalue=.1)

    one_img = np.zeros(seed_array.shape)
    one_img[seed_array == 0] = 1.
    one_img[seed_array != 0] = 0.

    for i in range(100):
        one_img = convolve2d(one_img, kernel, mode='same', boundary='fill', fillvalue=.1)

    result = zero_img / (zero_img + one_img)
    result -= result.min()
    result /= result.max()
    result *= n
    result = np.rint(result)

    # creating labels
    y = np.zeros(result.shape) - 2
    if boundary_size is None:
        boundary_size = n // 5
    bucket_size = (n - 2 * boundary_size) // 3
    y[result < bucket_size] = 0
    y[(result >= bucket_size + boundary_size) & (result < 2 * bucket_size + boundary_size)] = 1
    y[(result >= 2 * bucket_size + 2 * boundary_size)] = 2

    # replacing labels with spectra
    X = np.zeros(seed_array.shape + (calibration.shape[0],))
    for i in range(sps.shape[0]):
        X[result == i, :] = sps[i]

    # adding noise
    X = convolve1d(X, smooth_kernel, axis=-1, mode='nearest')
    X /= X.max()
    X = X + np.random.normal(noise_mean, noise_var, X.shape)
    return X, y, calibration