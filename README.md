# Overview
This app is designed for manual labeling of hyperspectral maps obtained by Laser-Induced Breakdown Spectroscopy.

## Data Description
The application is designed to work with internal data from the CEITEC LIBS laboratory. As of writing this, the used formats are undergoing rapid development without any standards or documentation. You will likely have to write a custom script to load the data (see [src/load_scripts.py](src/load_scripts.py)). A small and simple dataset is shared with the application, see Project Structure.

## Interface Description
The main focus point of the application is the Image Panel, it displays the hyperspectral image as a heatmap with intensities summed over all measured wavelengths and provides the user with various tools that allow them to quickly assign labels to large areas of the image by drawing over them. A line plot allows the user to inspect any spectrum by hovering over the corresponding point in the heatmap. To change which wavelengths are used to calculate the user can select a region on a line plot of the mean spectrum. The labeling process is further supported by a variety of machine-learning models incorporated in the application which can be trained to predict the yet unlabelled spectra using the already assigned labels.

# Usage
After cloning the repository and installing the modules listed in requirements.txt the app can be run with the [src/app.py](src/app.py) script. The script will run a simple wizard to gain access to the spectra to be analyzed and launch an interactive Plotly Dash application on [localhost](http://127.0.0.1:8050).

# Project Structure

# Sharing the Application
A single file executable version of the application can be created using [PyInstaller](https://pyinstaller.org/en/stable/) with the following commands (when starting from the root folder of the project).

```
cd src
pyinstaller --onefile app.py
```

# Adding Machine Learning Models

