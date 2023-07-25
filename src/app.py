import numpy as np
import pandas as pd
import json
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import utils.style as style
from dash import Dash, html, Input, Output, State, dcc
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from components.hyperspectral_image import hyperspectral_image
from components.image_controls import make_control_panel
from components.spectrum_panel import spectrum_panel
from components.app_controls import app_controls
from components.meta import make_meta
from segmentation.models import models
from utils.visualization import plot_spectra, add_colorbar, add_legend, draw_hyperspectral_image
from utils.rasterization import rasterize_and_draw
from utils.application import mouse_path_to_indices, coordinates_from_hover_data
from utils.load_scripts import load_toy_dataset, load_contest_dataset
from utils.app_modes import App_modes
from PIL import Image, ImageDraw
from base64 import b64decode
from matplotlib import cm
import io

mode = 0
num_classes = 7

if mode == 0:
    X, y_true, wavelengths, dim, app_mode = load_toy_dataset()
elif mode == 1:
    X, y_true, wavelengths, dim, app_mode = load_contest_dataset()
else:
    raise NotImplementedError


# precompute mean (mostly) for selected spectrum plot
mean_spectrum = X.mean(axis=(0, 1))

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'LIBS Segmentation'

app.layout = html.Div([
    dbc.Container([
        dbc.Row(
            dbc.Col(
                make_control_panel(app_mode)
            )
        ),
        html.Br(),
        dbc.Row([
            dbc.Col([
                hyperspectral_image
            ], width=8),
            dbc.Col([
                spectrum_panel
            ], width=4)
        ], justify="evenly",),
        html.Br(),
        dbc.Row(
            dbc.Col(
                app_controls
            )
        ),
        html.Br(),
        dbc.Row([
            dbc.Col([make_meta(dim)])
        ])
    ], fluid=True)
])


@app.callback(
    Output('retrain_btn', 'outline'),
    Input('retrain_btn', 'n_clicks'),
    Input('manual_labels', 'data'),
    Input('model_identifier', 'value'),
    prevent_initial_call=True
)
def highlight_retrain_btn(retrain, manual_labels, model_id):
    if ctx.triggered_id == 'retrain_btn':
        return True
    return manual_labels is not None


@app.callback(
    Output('download', 'data'),
    Input('save_labels', 'n_clicks'),
    Input('save_output', 'n_clicks'),
    State('manual_labels', 'data'),
    State('model_output', 'data'),
    prevent_initial_call=True
)
def download_files(l_click, s_click, manual_labels, model_out):
    if ctx.triggered_id == 'save_labels':
        return dcc.send_data_frame(pd.DataFrame(manual_labels).to_csv, "manual_labels.csv")
    # save_output
    return dcc.send_data_frame(pd.DataFrame(model_out).to_csv, "full_segmentation.csv")


@app.callback(
    Output('manual_labels', 'data', allow_duplicate=True),
    Input('load_labels', 'contents'),
    prevent_initial_call=True,
)
def upload_labels(upload):
    content_type, content_string = upload.split(",")
    decoded = b64decode(content_string)
    return pd.read_csv(io.BytesIO(decoded), sep=',', header=0, index_col=0).values


if app_mode == App_modes.Benchmark:
    from itertools import permutations
    @app.callback(
        Output('accuracy', 'children'),
        Input('model_output', 'data'),
        prevent_initial_call=True
    )
    def display_accuracy(y):
        y = np.array(y)
        new_y = np.zeros(y.shape)
        scores = []
        for label0, label1, label2 in permutations((0, 1, 2)):
            new_y[y == 0] = label0
            new_y[y == 1] = label1
            new_y[y == 2] = label2
            scores.append(np.sum((new_y == y_true) & (y_true != -2)) / np.sum((y_true != -2)))
        return f'Accuracy: {max(scores)}'


@app.callback(
    Output('manual_labels', 'data'),
    State('manual_labels', 'data'),
    Input('apply_changes_btn', 'n_clicks'),
    Input('reset_manual_labels_btn', 'n_clicks'),
    Input('x_map', 'relayoutData')
)
def update_manual_labels(memory, apply, reset, relayout, width=2):
    if ctx.triggered_id == 'reset_manual_labels_btn' or memory is None:
        return np.zeros(dim) - 1
    if ctx.triggered_id == 'apply_changes_btn' and 'shapes' in relayout:
        memory = np.array(memory)
        for shape in relayout['shapes']:
            memory = rasterize_and_draw(shape, memory)
        return memory
    raise PreventUpdate


# TODO disable show output button
@app.callback(
    Output('model_output', 'data'),
    Input('retrain_btn', 'n_clicks'),
    State('manual_labels', 'data'),
    State('model_identifier', 'value'),
    prevent_initial_call=True
)
def update_model_output(retrain, labels, model_id):
    y_in = np.array(labels).flatten()
    X_in = X.reshape((-1, wavelengths.shape[0]))
    return models[int(model_id)].fit(X_in, y_in).predict(X_in).reshape(dim)


@app.callback(
    Output('spectral_intensities', 'data'),
    Input('global_spectrum', 'relayoutData')
)
def update_spectral_intensities(wave_range):
    if wave_range is None or "xaxis.autorange" in wave_range or 'autosize' in wave_range:
        return X.sum(axis=2)
    else:
        return X[:, :, (wavelengths >= float(wave_range["xaxis.range[0]"])) & (wavelengths <= float(wave_range["xaxis.range[1]"]))].sum(axis=2)


@app.callback(
    Output('uirevision', 'children'),
    State('uirevision', 'children'),
    Input('reset_manual_labels_btn', 'n_clicks'),
    Input('apply_changes_btn', 'n_clicks'),
)
def update_revision(memory, *args):
    if memory is None:
        return 0
    return memory + 1 % 2


def make_spectral_image(spectral_intensities):
    spectral_intensities = np.array(spectral_intensities)
    zmin = spectral_intensities.min()
    zmax = spectral_intensities.max()
    spectral_image = cm.Reds((spectral_intensities - zmin) / zmax, alpha=1.) * 255
    return spectral_image, zmin, zmax


def make_model_output_image(model_output, num_classes):
    model_output = np.array(model_output)
    return cm.Set1(model_output / (num_classes + 1), alpha=.6) * 255


def make_true_output_image():
    img = cm.Set1(y_true / (num_classes + 1), alpha=0.6) * 255
    img[y_true == -2, :] = cm.Set1(np.array([1]), alpha=0.6) * 255
    return img


def add_manual_labels(img, manual_labels, num_classes, add_input):
    if add_input % 2 == 0:
        manual_labels = np.array(manual_labels)
        manual_labels_image = cm.Set1(manual_labels / (num_classes + 1), alpha=1.) * 255
        mask = np.repeat(manual_labels[:,:, np.newaxis], 4, axis=2)
        return np.where(mask >= 0, manual_labels_image, img)
    return img


def check_if_update(ctx, buffer_id, add_input):
    if ctx.triggered_id not in [buffer_id, 'image_output_mode_btn', 'show_input_btn', 'manual_labels']:
        raise PreventUpdate
    if ctx.triggered_id == 'manual_labels' and add_input % 2 == 1:
        raise PreventUpdate


@app.callback(
    Output('show_input_btn', 'outline'),
    Input('show_input_btn', 'n_clicks'),

)
def update_test(n_clicks):
    return n_clicks % 2 == 0


@app.callback(
    Output('x_map', 'figure'),
    State('x_map', 'figure'),
    Input('manual_labels', 'data'),
    Input('model_output', 'data'),
    Input('spectral_intensities', 'data'),
    Input('image_output_mode_btn', 'value'),
    Input('show_input_btn', 'n_clicks'),
    State('uirevision', 'children')
)
def update_X_map(state, manual_labels, model_output, spectral_intensities, mode, add_input, reset_ui):
    if mode is None or mode == 'show_spectra':
        check_if_update(ctx, 'spectral_intensities', add_input)
        img, zmin, zmax = make_spectral_image(spectral_intensities)
        img = add_manual_labels(img, manual_labels, num_classes, add_input)
    
    elif mode == 'show_output':
        check_if_update(ctx, 'model_output', add_input)
        if model_output is None:
            raise RuntimeError('Model not trained')  # TODO disable the button
        img, zmin, zmax = make_model_output_image(model_output, num_classes), 0, 0
        img = add_manual_labels(img, manual_labels, num_classes, add_input)

    elif mode == 'show_true_labels':
        check_if_update(ctx, 'no_buffer', add_input)
        img, zmin, zmax = make_true_output_image(), 0, 0
        img = add_manual_labels(img, manual_labels, num_classes, add_input)

    return draw_hyperspectral_image(img, zmin, zmax, reset_ui, state, num_classes)


@app.callback(
    Output('selected_spectrum', 'figure'),
    Input('x_map', 'hoverData'),
)
def update_selected_spectrum(hover):
    if hover is not None:
        x, y = coordinates_from_hover_data(hover)
    else:
        x, y = 0, 0
    fig = plot_spectra([mean_spectrum, X[x, y, :]], wavelengths=wavelengths, labels=['mean', 'hover'])
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0,),
        legend=dict(
            bgcolor = 'rgba(0, 0, 0, 0)',
            x=0,
            y=1,
        )
    )
    return fig


@app.callback(
    Output('global_spectrum', 'figure'),
    Input('model_output', 'data'),
)
def update_global_spectrum(y):
    spectra = [mean_spectrum]
    labels = ['global mean']
    colors = [px.colors.qualitative.Set1[-1]]
    if y is not None:
        for cls in np.unique(y):
            spectra.append(X[y==cls].mean(axis=0))
            labels.append(f'class {cls} mean')
            colors.append(px.colors.qualitative.Set1[cls])
    fig = plot_spectra(spectra=spectra, wavelengths=wavelengths, labels=labels, colormap=px.colors.qualitative.Set1)
    fig.update_layout(
        template='plotly_white',
        yaxis=dict(fixedrange=True,),
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0,),
        legend=dict(
            bgcolor = 'rgba(0, 0, 0, 0)',
            x=0,
            y=1,
        )
    )
    return fig


# TODO remove debug
if __name__ == "__main__":
    app.run_server(debug=True)