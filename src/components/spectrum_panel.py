import dash_bootstrap_components as dbc
from dash import dcc, html

btns = [
    'zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale'
]

selected_spectrum = dbc.Card([
    dbc.CardHeader('Currently selected spectrum'),
    dbc.CardBody([
        dcc.Graph(
            id='selected_spectrum',
            style={'height':'30vh', 'width':'28vw'},
            config={
                'modeBarButtonsToRemove': btns,
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': 'selected_spectrum',
                }
            }
        ),
    ])
])

global_spectrum = dbc.Card([
    dbc.CardHeader('Mean spectrum (resize to change how the total intensity is calculated)'),
    dbc.CardBody([
        dcc.Graph(
            id='global_spectrum',
            style={'height':'30vh', 'width':'28vw'},
            config={
                'modeBarButtonsToRemove': btns,
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': 'global_spectrum',
                }
            }
        ),
    ])
])

spectrum_panel = dbc.Card(
    dbc.CardBody([
        dbc.Row(
            dbc.Col(
                selected_spectrum
            )
        ),
        html.Br(),
        dbc.Row(
            dbc.Col(
                global_spectrum
            )
        ),
    ])
)