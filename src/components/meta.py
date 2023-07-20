import numpy as np
from dash import html, dcc

# TODO make sure nothing displays
def make_meta(dim):
    meta = html.Div(
        [
            # TODO dcc.Store storage type? currently loses data on reload
            dcc.Store(id='manual_labels'),
            dcc.Loading(children=dcc.Store(id='model_output'), fullscreen=True),
            dcc.Store(id='spectral_intensities'),
            dcc.Store(id='image'),
            dcc.Location(id='url'),
            html.Div(id='uirevision', style={'display': 'none'}),
            dcc.Download(id='download'),

            html.Div(id='accuracy'),
        ],
    )

    return meta