import dash_bootstrap_components as dbc
from segmentation.models import model_names
from utils.app_modes import App_modes

def make_ml_model_controls(app_mode):
    if app_mode == App_modes.Benchmark:
        """
        dbc.Col([dbc.RadioItems(
            id="show_output_btn",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Show Segmentation", "value": 0},
                {"label": "Show Labels", "value": 1},
                {"label": "Show Spectra", "value": 2},
            ],
            value=2
        )]),
        """
        raise NotImplemented

    options = [
        {'label': name, 'value': val} for name, val in model_names.items()
    ]

    ml_model_controls = dbc.Card([
        dbc.CardHeader('Model panel'),
        dbc.Row([
            dbc.Col([dbc.Button('Show Segmentation', id='show_output_btn', disabled=True)]),
            dbc.Col([dbc.Button('Train Model', id='retrain_btn')]),
            dbc.Col([dbc.Select(
                id='model_identifier',
                placeholder=options[0]['label'],
                options=options,
            )])
        ])
    ])

    return ml_model_controls