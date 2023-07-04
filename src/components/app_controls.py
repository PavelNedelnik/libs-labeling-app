import dash_bootstrap_components as dbc
from dash import dcc

app_controls = dbc.Card([
    dbc.CardHeader('Application panel'),
        dbc.Row([
            dbc.Col([dbc.Button('Download Manual Labels', id='save_labels')]),
            dbc.Col(dcc.Upload(dbc.Button('Upload Manual Labels'),id='load_labels')),
            dbc.Col(dbc.Button('Download Segmentation', id='save_output')),
        ]),
])