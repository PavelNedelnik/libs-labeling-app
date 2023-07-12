import dash_bootstrap_components as dbc
from segmentation.models import model_names
from utils.app_modes import App_modes

def make_ml_model_controls(app_mode):
    output_options = [
        {"label": "Show Spectra", "value": 'show_spectra'},
        {"label": "Show Output", "value": 'show_output'},
    ]
    if app_mode == App_modes.Benchmark:
        output_options.append({"label": "show_true_labels", "value": 2})

    show_button = dbc.RadioItems(
        id="image_output_mode_btn",
        className="btn-group",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary",
        labelCheckedClassName="active",
        options=output_options,
        value=0,
    )
    
    model_options = [
        {'label': name, 'value': val} for name, val in model_names.items()
    ]

    ml_model_controls = dbc.Card([
        dbc.CardHeader('Model panel'),
        dbc.Row([
            dbc.Col(show_button),
            dbc.Col([dbc.Button('Train Model', id='retrain_btn')]),
            dbc.Col([dbc.Select(
                id='model_identifier',
                placeholder=model_options[0]['label'],
                options=model_options,
                value=0
            )])
        ])
    ])

    return ml_model_controls