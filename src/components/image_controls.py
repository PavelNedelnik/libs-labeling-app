import dash_bootstrap_components as dbc
from segmentation.models import model_names
from utils.app_modes import App_modes


def make_control_panel(app_mode):
    output_options = [
        {"label": "Show Spectra", "value": 'show_spectra'},
        {"label": "Show Output", "value": 'show_output'},
    ]
    if app_mode == App_modes.Benchmark:
        output_options.append({"label": "Show True Labels", "value": 'show_true_labels'})
    
    model_options = [
        {'label': name, 'value': val} for name, val in model_names.items()
    ]

    control_panel = dbc.Card([dbc.CardBody(dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.Button('Reset', id='reset_manual_labels_btn', n_clicks=0)
            ),
        ]),

        dbc.Col([
            dbc.Card(
                dbc.Button('Apply', id='apply_changes_btn', n_clicks=0)
            ),
        ]),

        dbc.Col([
            dbc.Card(
                dbc.Button('Clear', id='clear_changes_btn', n_clicks=0)
            ),
        ]),

        dbc.Col([
            dbc.RadioItems(
                id="image_output_mode_btn",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=output_options,
                value=output_options[0]['value'],
            )
        ]),

        dbc.Col([
            dbc.Button('Train Model', id='retrain_btn')
        ]),

        dbc.Col([dbc.Select(
            id='model_identifier',
            placeholder=model_options[0]['label'],
            options=model_options,
            value=0
        )])
    ]))])

    return control_panel