import numpy as np
import json
import dash_bootstrap_components as dbc
import plotly.express as px
import utils.style as style
from dash import Dash, html, Input, Output
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from components.hyperimage_panel import make_hyperimage_panel
from components.app_controls import app_controls
from components.ml_model_controls import make_ml_model_controls
from components.spectrum_panel import spectrum_panel
from segmentation.models import models
from utils.visualization import plot_map, plot_spectra
from utils.application import mouse_path_to_indices, coordinates_from_hover_data
from utils.load_scripts import load_toy_dataset
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
num_classes = 2
X, y_true, calibration, dim = load_toy_dataset()
app_mode = App_modes.Default


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


# TODO delete
@app.callback(
    Output('test', 'children'),
    Input('load_labels', 'contents'),
)
def idk(inp):
    return ''


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
    Output('manual_labels', 'data'),
    Input('manual_labels', 'data'),
    Input('mode_button', 'value'),
    Input('width', 'value'),
    Input('x_map', 'relayoutData'),
    prevent_initial_call=True,
)
def update_manual_labels(memory, mode, width, relayout):
    if mode == -4:
        return np.zeros(dim) - 1
    if ctx.triggered_id != 'x_map' or 'shapes' not in relayout or mode < -2:
        raise PreventUpdate
    img = Image.fromarray(np.array(memory))
    draw = ImageDraw.Draw(img)
    node_coords = mouse_path_to_indices(relayout['shapes'][-1]['path'])
    # TODO bug leaves little holes
    draw.line(node_coords, fill=mode, width=int(width) if width else 2, joint='curve')
    return np.asarray(img)


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


# TODO better solve for Benchmark
if app_mode == App_modes.Default:
    @app.callback(
        Output('show_output_btn', 'disabled'),
        Input('retrain_btn', 'n_clicks'),
        prevent_initial_call=True,
    )
    def disable_show_segmentation(click):
        if click is not None:
            return False
        return True
    

@app.callback(
    Output('model_output', 'data'),
    Input('retrain_btn', 'n_clicks'),
    Input('manual_labels', 'data'),
    Input('model_identifier', 'value'),
)
def calculate_model_output(_, labels, model_identifier):
    if ctx.triggered_id != 'retrain_btn':
        raise PreventUpdate
    
    model_identifier = int(model_identifier) if model_identifier else 0

    y_in = np.array(labels).flatten()

    X_in = X.reshape((-1, calibration.shape[0]))

    return models[int(model_identifier)].fit(X_in, y_in).predict(X_in).reshape(dim)


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
    

if app_mode == App_modes.Benchmark:
    """
    if (app_mode > 0 and show_segment_btn == 2) or (not app_mode > 0 and (show_segment_btn is None or show_segment_btn % 2 == 0)):
        # show image
        if wave_range is None or "xaxis.autorange" in wave_range or 'autosize' in wave_range:
            values = X.sum(axis=2)
        else:
            values = X[:, :, (calibration >= float(wave_range["xaxis.range[0]"])) & (calibration <= float(wave_range["xaxis.range[1]"]))].sum(axis=2)
        img = np.where(mask >= 0, cm.Set1(manual_labels / (num_classes), alpha=1.) * 255, cm.Reds((values - values.min()) / values.max(), alpha=1.) * 255)
    elif (app_mode > 0 and show_segment_btn == 0) or not app_mode > 0:
        # show segmentation
        img = np.where(mask >= 0, cm.Set1(manual_labels / (num_classes), alpha=1.) * 255, cm.Set1(y / (num_classes), alpha=.8) * 255)
    else:
        img = cm.Set1(y_true / (num_classes), alpha=1) * 255
        img[y_true == -2, :] = (128, 128, 128, 255)
    """
    raise NotImplemented

@app.callback(
    Output('x_map', 'figure'),
    Input('global_spectrum', 'relayoutData'),
    Input('manual_labels', 'data'),
    Input('screen_resolution', 'children'),
    Input('mode_button', 'value'),
    Input('show_output_btn', 'n_clicks'),  # 'value' if app_mode > 0 else 
    Input('model_output', 'data'),
)
def update_X_map(wave_range, manual_labels, screen_resolution, mode, show_segment_btn, y):
    # unpack input values
    manual_labels = np.array(manual_labels)
    screen_resolution = json.loads(screen_resolution)
    y = np.array(y)

    # broadcast manual labels to multi-channel image
    mask = np.repeat(manual_labels[:,:, np.newaxis], 4, axis=2)

    # choose one of two main modes
    if show_segment_btn is None or show_segment_btn % 2 == 0:
        # show image
        if wave_range is None or "xaxis.autorange" in wave_range or 'autosize' in wave_range:
            values = X.sum(axis=2)
        else:
            values = X[:, :, (calibration >= float(wave_range["xaxis.range[0]"])) & (calibration <= float(wave_range["xaxis.range[1]"]))].sum(axis=2)
        img = np.where(mask >= 0, cm.Set1(manual_labels / (num_classes), alpha=1.) * 255, cm.Reds((values - values.min()) / values.max(), alpha=1.) * 255)
    else:
        # show segmentation
        img = np.where(mask >= 0, cm.Set1(manual_labels / (num_classes), alpha=1.) * 255, cm.Set1(y / (num_classes), alpha=.8) * 255)

    img = np.where(mask == -2, 128, img)

    # generate plot
    fig = px.imshow(img=img, labels={})
    fig.update_traces(
        hovertemplate='<',
        hoverinfo='skip',
    )
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode='zoom' if mode < -2 else 'drawopenpath',
        newshape=dict(opacity=0),  # TODO shapes are currently just hidden but not deleted
        xaxis=dict(visible=False, range=fig['layout']['xaxis']['range'] if fig else None),
        yaxis=dict(visible=False, range=fig['layout']['yaxis']['range'] if fig else None),
        width=int(min(screen_resolution['height'] * .9, screen_resolution['width'] * .7)),
        height=int(min(screen_resolution['height'] * .9, screen_resolution['width'] * .7)),
        uirevision='None',
        shapes=[],  # TODO this does not remove the shapes!
    )
    fig.update_shapes(editable=False)

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
    fig = plot_spectra([X.mean(axis=(0, 1)), X[x, y, :]], calibration=calibration, labels=['mean', 'hover'])
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0,),
    )
    return fig


@app.callback(
    Output('global_spectrum', 'figure'),
    Input('accuracy', 'children'),  # TODO
)
def update_global_spectrum(_):
    fig = plot_spectra([X.mean(axis=(0, 1))], calibration=calibration, colormap=style.RANGE_SLIDER_COLORS)
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