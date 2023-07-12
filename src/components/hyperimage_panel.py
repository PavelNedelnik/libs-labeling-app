import dash_bootstrap_components as dbc
from dash import dcc, html

btns = [
    'drawline',
    'drawopenpath',
    'drawclosedpath',
    'drawcircle',
    'drawrect',
    'eraseshape',
]

def make_hyperimage_panel(num_classes):
    hyperimage_controls = dbc.Row([
        dbc.Col([
            dbc.Card(
                html.Button('Reset', id='reset_manual_labels_btn', n_clicks=0)
            ),
        ]),

        dbc.Col([
            dbc.Card(
                html.Button('Apply', id='apply_changes_btn', n_clicks=0)
            ),
        ]),

        dbc.Col([
            dbc.Card(
                html.Button('Clear', id='clear_changes_btn', n_clicks=0)
            ),
        ]),

        dbc.Col([
            dbc.Card([
                dbc.CardBody(dbc.RadioItems(
                    id='active_class_selector',
                    className='btn-group',
                    inputClassName='btn-check',
                    labelClassName='btn btn-outline-primary',
                    labelCheckedClassName='active',
                    options=[
                        {'label': f'Ignore', 'value':  - 1} ] + [
                        {'label': f'Class {i}', 'value': i} for i in range(num_classes)
                    ],
                    value=0
                )),
            ]),
        ])
    ])

    """        dbc.Col([
        dbc.Card([
            dbc.CardBody(dcc.Input(
                id='width',
                type='number',
                placeholder='Brush width (2)'
            )),
        ]),
    ]),"""

    hyperimage_panel = dbc.Card([
        dbc.CardHeader('Image panel'),
        dbc.CardBody([
            hyperimage_controls,
            html.Br(),
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(
                        id='x_map',
                        config={'modeBarButtonsToAdd': btns},
                    ))),
                )
            ])
        ])
    ])

    return hyperimage_panel