# Overview
This app is designed for manual labeling of hyperspectral maps obtained by Laser-Induced Breakdown Spectroscopy.

## Data Description
The application is designed to work with internal data from the CEITEC LIBS laboratory. As of writing this, the used formats are undergoing rapid development without any standards or documentation. You will likely have to write a custom script to load the data (see [src/load_scripts.py](src/load_scripts.py)). A small and simple dataset is shared with the application, see Project Structure.

## Interface Description
The main focus point of the application is the Image Panel at the left part of the screen, it displays the hyperspectral image as a heatmap with intensities summed over all measured wavelengths and provides the user with various tools that allow them to quickly assign labels to large areas of the image by drawing over them. The Spectrum Panel, displayed on the right part of the screen, consists of two line plots: one allows the user to inspect any spectrum by hovering over the corresponding point in the heatmap, while the other displays the mean spectrum and allows them to change which wavelengths are used to calculate the intensity map. The labeling process is further supported by a variety of machine-learning models connected to the application via the Control Panel at the top of the screen. The assigned labels can be downloaded (or uploaded to continue a previous session) via the Application Panel at the bottom.

# Usage
After cloning the repository and installing the modules listed in requirements.txt the app can be run with the [src/app.py](src/app.py) script. The script will run a simple wizard to gain access to the spectra to be analyzed and launch an interactive Plotly Dash application on [localhost](http://127.0.0.1:8050). The setup could be completed with the following commands:

```
$ git clone https://github.com/PavelNedelnik/libs-labeling-app.git  # clone the repository
$ pip install -r requirements.txt
$ ./src/app.py
```

# Project Structure

```
.
├── datasets                  # place to store your data
│   └── toy_dataset           # example dataset so the app can be run out-of-the-box
├── simulated_data            # data used by and created by simulation.py
│   └── TODO
├── src                       # source files
│   ├── components            # logical parts of the app layout
│   ├── segmentation          # the supporting machine-learning models
│   │   ├── ...
│   │   └── models.py             # 
│   ├── utils                 # supporting code
│   │   ├── application_utils.py  #
│   │   ├── colors.py  #
│   │   ├── load_scripts.py  #
│   │   ├── rasterization.py  #
│   │   ├── simulation.py  #
│   │   ├── style.py              # currently unused
│   │   └── visualization.py  #
│   ├── app.py                #
│   └── simulated_data.ipynb  # TODO
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt          # required packages
```

# Adding Machine Learning Models
TODO

