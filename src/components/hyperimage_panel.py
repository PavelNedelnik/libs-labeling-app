import dash_bootstrap_components as dbc
from dash import dcc, html

def make_hyperimage_panel(num_classes):
    hyperimage_controls = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody(dbc.RadioItems(
                    id="mode_button",
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=[
                        {"label": "Reset", "value": -4},
                        {"label": "Zoom", "value": -3},
                        {"label": "Clear", "value": -1},
                        {"label": "Ignore", "value": -2}, ] + [
                        {'label': f'Class {i}', 'value': i} for i in range(num_classes)
                    ],
                    value=0
                )),
            ]),
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody(dcc.Input(
                    id='width',
                    type='number',
                    placeholder='Brush width (2)'
                )),
            ]),
        ]),
    ])

    hyperimage_panel = dbc.Card([
        dbc.CardHeader('Image panel'),
        dbc.CardBody([
            hyperimage_controls,
            html.Br(),
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(
                        id='x_map',
                        config={
                            'displayModeBar': False
                        },
                    ))),
                )
            ])
        ])
    ])

    return hyperimage_panel