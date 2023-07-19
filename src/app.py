import numpy as np
import json
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import utils.style as style
from dash import Dash, html, Input, Output, State
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from components.hyperspectral_image import hyperspectral_image
from components.image_controls import make_control_panel
from components.spectrum_panel import spectrum_panel
from components.app_controls import app_controls
from components.meta import make_meta
from segmentation.models import models
from utils.visualization import plot_spectra, add_colorbar, add_legend
from utils.application import mouse_path_to_indices, coordinates_from_hover_data
from utils.load_scripts import load_toy_dataset, load_contest_dataset
from utils.app_modes import App_modes
from PIL import Image, ImageDraw
from base64 import b64decode
from matplotlib import cm
import io

mode = 1  # 0 for normal use, 1 for benchmark with known y, 2 for benchmark on simulated data
num_classes = 3  # might be overriden by dataset choice

if mode == 0:
    X, y_true, calibration, dim, app_mode = load_toy_dataset()
elif mode == 1:
    X, y_true, calibration, dim, app_mode = load_contest_dataset()
    num_classes = len(np.unique(y_true))
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
    prevent_initial_call=True,
)
def highlight_retrain_btn(*args, **kwargs):
    if ctx.triggered_id == 'retrain_btn':
        return True
    return False


@app.callback(
    Output('download', 'data'),
    Input('save_labels', 'n_clicks'),
    Input('save_output', 'n_clicks'),
    Input('manual_labels', 'data'),
    Input('model_output', 'data'),
    prevent_initial_call=True,
)
def download_files(l_click, s_click, manual_labels, model_out):
    if ctx.triggered_id == 'save_labels':
        return {'content': json.dumps(manual_labels), 'filename':'manual_labels.json'}
    elif ctx.triggered_id == 'save_output':
        return {'content': json.dumps(model_out), 'filename':'segmentation_mask.json'}
    raise PreventUpdate


@app.callback(
    Output('manual_labels', 'data', allow_duplicate=True),
    Input('load_labels', 'contents'),
    prevent_initial_call=True,
)
def upload_labels(upload):
    content_type, content_string = upload.split(",")
    decoded = b64decode(content_string)
    return json.loads(io.BytesIO(decoded).getvalue())


if app_mode == App_modes.Benchmark:
    from itertools import permutations
    @app.callback(
        Output('accuracy', 'children'),
        Input('model_output', 'data'),
        prevent_initial_call=True,
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
    Input('manual_labels', 'data'),
    Input('apply_changes_btn', 'n_clicks'),
    Input('reset_manual_labels_btn', 'n_clicks'),
    Input('x_map', 'relayoutData'),
    # line width
)
def update_manual_labels(memory, apply, reset, relayout, width=2):
    if ctx.triggered_id == 'reset_manual_labels_btn' or memory is None:
        return np.zeros(dim) - 1
    if ctx.triggered_id == 'apply_changes_btn':
        for shape in relayout['shapes']:
            try:
                img = Image.fromarray(np.array(memory))
                draw = ImageDraw.Draw(img)
                node_coords = mouse_path_to_indices(shape['path'])
                draw.line(node_coords, fill=mode, width=int(width) if width else 2, joint='curve')
                return np.asarray(img)
            except ValueError:
                pass
            except KeyError:
                pass
    raise PreventUpdate


# TODO disable show output button
@app.callback(
    Output('model_output', 'data'),
    Input('retrain_btn', 'n_clicks'),
    Input('manual_labels', 'data'),
    Input('model_identifier', 'value'),
)
def update_model_output(retrain, labels, model_id):
    if ctx.triggered_id == 'retrain_btn':
        y_in = np.array(labels).flatten()
        X_in = X.reshape((-1, calibration.shape[0]))
        return models[int(model_id)].fit(X_in, y_in).predict(X_in).reshape(dim)
    raise PreventUpdate


@app.callback(
    Output('spectral_intensities', 'data'),
    Input('global_spectrum', 'relayoutData'),
)
def update_spectral_intensities(wave_range):
    if wave_range is None or "xaxis.autorange" in wave_range or 'autosize' in wave_range:
        return X.sum(axis=2)
    else:
        return X[:, :, (calibration >= float(wave_range["xaxis.range[0]"])) & (calibration <= float(wave_range["xaxis.range[1]"]))].sum(axis=2)


@app.callback(
    Output('image', 'data'),
    Input('manual_labels', 'data'),
    Input('model_output', 'data'),
    Input('spectral_intensities', 'data'),
    Input('image_output_mode_btn', 'value'),
)
def update_image(manual_labels, model_output, spectral_intensities, mode):
    # TODO string value?
    if mode is None or mode == 'show_spectra':
        spectral_intensities = np.array(spectral_intensities)
        manual_labels = np.array(manual_labels)
        mask = np.repeat(manual_labels[:,:, np.newaxis], 4, axis=2)

        manual_labels_image = cm.Set1(manual_labels / (num_classes + 1), alpha=1.) * 255
        zmin = spectral_intensities.min()
        zmax = spectral_intensities.max()
        spectral_image = cm.Reds((spectral_intensities - zmin) / zmax, alpha=1.) * 255
        return (np.where(mask >= 0, manual_labels_image, spectral_image), zmin, zmax)
    
    elif mode == 'show_output':
        manual_labels = np.array(manual_labels)
        model_output = np.array(model_output)
        mask = np.repeat(manual_labels[:,:, np.newaxis], 4, axis=2)

        manual_labels_image = cm.Set1(manual_labels / (num_classes + 1), alpha=1.) * 255
        model_output_image = cm.Set1(model_output / (num_classes + 1), alpha=.6) * 255
        return (np.where(mask >= 0, manual_labels_image, model_output_image), 0, 0)
    elif mode == 'show_true_labels':
        img = cm.Set1(y_true / (num_classes + 1), alpha=1) * 255
        img[y_true == -2, :] = (128, 128, 128, 255)
        return (img, 0, 0)
    raise NotImplementedError


@app.callback(
    Output('uirevision', 'children'),
    State('uirevision', 'children'),
    Input('reset_manual_labels_btn', 'n_clicks'),
    Input('apply_changes_btn', 'n_clicks'),
    Input('clear_changes_btn', 'n_clicks'),
)
def update_revision(memory, *args):
    try:
        return str(int(memory) + 1 % 2)
    except ValueError:
        return '0'
    except TypeError:
        return '0'


@app.callback(
    Output('x_map', 'figure'),
    Input('image', 'data'),
    Input('uirevision', 'children'),
)
def update_X_map(image, reset_ui):
    img, zmin, zmax = image
    img = np.array(img)

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    add_colorbar(fig, zmin, zmax)

    fig.update_layout(
        legend_orientation='h',
        template='plotly_white',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        uirevision=reset_ui,
        newshape=dict(line=dict(color=px.colors.qualitative.Set1[0])),
        updatemenus = list([
            dict(type = "buttons",
                direction = "down",
                active = 0,
                showactive = True,
                x = 0,
                y = 1,
                buttons = [
                    dict(
                        label = f'Class {i}',
                        method = "relayout", 
                        args = [{'newshape.line.color': px.colors.qualitative.Set1[i]}]
                    ) for i in range(num_classes)
                ]
            ),
            dict(type = "dropdown",
                active = 0,
                direction="right",
                x = 0.1,
                y = 0,
                showactive = True,
                buttons = [
                    dict(
                        label = str(width),
                        method = "relayout", 
                        args = [{'newshape.line.width': width}]
                    ) for width in range(2, 10)
                ]
            )
        ]),
    )

    return fig


@app.callback(
    Output('selected_spectrum', 'figure'),
    Input('x_map', 'hoverData'),
)
def update_selected_spectrum(hover):
    if hover is not None:
        x, y = coordinates_from_hover_data(hover)
    else:
        x, y = 0, 0
    fig = plot_spectra([mean_spectrum, X[x, y, :]], calibration=calibration, labels=['mean', 'hover'])
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0,),
    )
    return fig


@app.callback(
    Output('global_spectrum', 'figure'),
    Input('model_output', 'data'),
)
def update_global_spectrum(y):
    fig = plot_spectra([mean_spectrum], calibration=calibration, colormap=style.RANGE_SLIDER_COLORS)
    fig.update_layout(
        template='plotly_white',
        yaxis=dict(fixedrange=True,),
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0,),
    )
    return fig


# TODO remove debug
if __name__ == "__main__":
    app.run_server(debug=True)