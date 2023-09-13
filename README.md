# Overview
This app is designed for labeling hyperspectral maps obtained by Laser-Induced Breakdown Spectroscopy. The main entry point to the application is the [src/app.py](src/app.py) script which, when run, will run a simple wizard to gain access to the spectra to be analyzed and launch an interactive Plotly Dash application on [localhost](http://127.0.0.1:8050).

# Installation

# Usage

# Project Structure

# Sharing the Application
A single file executable version of the application can be created using [PyInstaller](https://pyinstaller.org/en/stable/) with the following commands (when starting from the root folder of the project).

```
cd src
pyinstaller --onefile app.py
```

## Adding Machine Learning Models

