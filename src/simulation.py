from SimulatedLIBS import simulation
import warnings
import numpy as np

PARAMS = dict(
    Te=1.0,
    Ne=10**18,
    resolution=1000,
    low_w=200,
    upper_w=1000,
    max_ion_charge=3,
    webscraping='dynamic',
)

def get_spectra(n, elements, save=False):
    if elements != sorted(elements):
        warnings.warn("The elements were sorted to be in alphabetical order.", Warning)
    elements = sorted(elements) # WARNING! sorts the elements in alphabetical order to prevent duplicities
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
                    percentages=[x,100-x],
                    **PARAMS,
                ),
                np.linspace(0, 100, n), 
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