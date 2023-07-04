from dash import html, dcc
import dash_bootstrap_components as dbc

def with_frame(component_body:list) -> html.Div:
    """
    Frame the component
    """
    return html.Div([
        dbc.Card(
            dbc.CardBody(
                component_body
            ),
            color = 'dark',
        )
    ])