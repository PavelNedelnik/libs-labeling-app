import numpy as np
from dash import html, dcc

# TODO make sure nothing displays
def make_meta(dim):
    meta = html.Div(
        [
            # TODO dcc.Store storage type? currently loses data on reload
            dcc.Store(id='manual_labels', data=np.zeros(dim) - 1),
            dcc.Store(id='model_output', data=None),
            dcc.Store(id='spectra_image', data=None),
            html.Div(id='test'),  # TODO delete
            dcc.Location(id='url'),
            html.Div(id='screen_resolution', style={'display': 'none'}),
            dcc.Download(id='download'),

            html.Div(id='accuracy'),
        ],
    )

    return meta