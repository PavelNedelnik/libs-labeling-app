import numpy as np
import json
import dash_bootstrap_components as dbc
import utils.style as style
from dash import Dash, html, Input, Output
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from components.hyperimage_panel import make_hyperimage_panel
from components.app_controls import app_controls
from components.ml_model_controls import make_ml_model_controls
from components.spectrum_panel import spectrum_panel
from segmentation.models import models
from utils.visualization import plot_spectra, plot_values_map, plot_output_map, plot_labels_map
from utils.application import mouse_path_to_indices, coordinates_from_hover_data
from utils.load_scripts import load_toy_dataset, load_contest_dataset
from components.meta import make_meta
from utils.app_modes import App_modes
from PIL import Image, ImageDraw
from base64 import b64decode
from matplotlib import cm
import io

"""
TODO
add other modes
"""
mode = 1  # 0 for normal use, 1 for benchmark with known y, 2 for benchmark on simulated data
num_classes = 3  # might be overriden by dataset choice

if mode == 0:
    X, y_true, calibration, dim, app_mode = load_toy_dataset()
elif mode == 1:
    X, y_true, calibration, dim, app_mode = load_contest_dataset()
    num_classes = len(np.unique(y_true))
else:
    raise NotImplemented


# precompute mean (mostly) for selected spectrum plot
mean_spectrum = X.mean(axis=(0, 1))

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'LIBS Segmentation'

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Row(
                    dbc.Col(
                        make_hyperimage_panel(num_classes)
                    )
                ),
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        app_controls
                    )
                ])
            ], width=7),
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        make_ml_model_controls(app_mode)
                    )
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        spectrum_panel
                    )
                ]),
            ], width=4)
        ], justify="evenly",),
        html.Br(),
        dbc.Row([
            dbc.Col([make_meta(dim)])
        ])
    ], fluid=True)
])


app.clientside_callback(
    """
    function(href) {
        var w = window.innerWidth;
        var h = window.innerHeight;
        return JSON.stringify({'height': h, 'width': w});
    }
    """,
    Output('screen_resolution', 'children'),
    Input('url', 'href')
)


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
        img = Image.fromarray(np.array(memory))
        draw = ImageDraw.Draw(img)
        node_coords = mouse_path_to_indices(relayout['shapes'][-1]['path'])
        draw.line(node_coords, fill=mode, width=int(width) if width else 2, joint='curve')
        manual_labels = np.asarray(img)
        return cm.Set1(manual_labels / (num_classes), alpha=1.) * 255
    raise PreventUpdate


# TODO disable retrain button
@app.callback(
    Output('model_output', 'data'),
    Input('retrain_btn', 'n_clicks'),
    Input('manual_labels', 'data'),
    Input('model_identifier', 'value'),
)
def update_model_output(retrain, labels, model_id):
    # TODO crashes for untrained model
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
    Input('manual_labels', 'value'),
    Input('model_output', 'value'),
    Input('spectral_intensities', 'value'),
    Input('image_output_mode_btn', 'relayoutData'),
)
def update_image(manual_labels, model_output, spectral_intensities, mode):
    # TODO string value?
    if mode == 'show_spectra':
        spectral_intensities = np.array(spectral_intensities)
        manual_labels = np.array(manual_labels)
        mask = np.repeat(manual_labels[:,:, np.newaxis], 4, axis=2)

        manual_labels_image = cm.Set1(manual_labels / (num_classes), alpha=1.) * 255
        spectral_image = cm.Reds((spectral_intensities - spectral_intensities.min()) / spectral_intensities.max(), alpha=1.) * 255
        return np.where(mask >= 0, manual_labels_image, spectral_image)
    
    elif mode == 'show_output':
        manual_labels = np.array(manual_labels)
        model_output = np.array(model_output)
        mask = np.repeat(manual_labels[:,:, np.newaxis], 4, axis=2)

        manual_labels_image = cm.Set1(manual_labels / (num_classes), alpha=1.) * 255
        model_output_image = cm.Set1(model_output / (num_classes), alpha=.6) * 255
        return np.where(mask >= 0, manual_labels_image, model_output_image)
    elif mode == 'show_true_labels':
        pass
    raise NotImplemented




# TODO
# Input('active_class_selector', 'value'),


@app.callback(
    Output('x_map', 'figure'),
    Input('image', 'data'),
    Input('show_output_btn', 'n_clicks' if app_mode == App_modes.Default else 'value'),
    Input('screen_resolution', 'children'),
)
def update_X_map(spectral_intensities, manual_labels, model_output, mode, output_button, y):
    """
    if ctx.triggered_id in ['mode_button', 'model_output']:
        raise PreventUpdate
    """
    # unpack input values
    spectra_image = np.array(spectra_image)
    manual_labels = np.array(manual_labels)
    screen_resolution = json.loads(screen_resolution)
    y = np.array(y)

    # broadcast manual labels to multi-channel image
    mask = np.repeat(manual_labels[:,:, np.newaxis], 4, axis=2)

    if app_mode == App_modes.Default:
        if (output_button is None or output_button % 2 == 0):
            fig = plot_values_map(spectra_image, manual_labels, mask, num_classes)
        else:
            fig = plot_output_map(y, mask, manual_labels, num_classes)
    elif app_mode == App_modes.Benchmark:
        if output_button == 2:
            fig = plot_values_map(spectra_image, manual_labels, mask, num_classes)
        elif output_button == 0:
            fig = plot_output_map(y, mask, manual_labels, num_classes)
        else:
            fig = plot_labels_map(y, mask, num_classes)
    else:
        raise NotImplemented

    fig.update_traces(
        colorbar_orientation='h',
        selector=dict(type='heatmap'),
    )
    fig.update_layout(
        legend_orientation='h',
        template='plotly_white',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        #newshape=dict(opacity=0),  # TODO shapes are currently just hidden but not deleted
        #xaxis=dict(visible=False, range=fig['layout']['xaxis']['range'] if fig else None),
        #yaxis=dict(visible=False, range=fig['layout']['yaxis']['range'] if fig else None),
        width=int(min(screen_resolution['height'] * .9, screen_resolution['width'] * .7)),
        height=int(min(screen_resolution['height'] * .9, screen_resolution['width'] * .7)),
        #uirevision='None',
        #shapes=[],  # TODO this does not remove the shapes!
    )
    #fig.update_shapes(editable=False)

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