import dash_bootstrap_components as dbc
from dash import dcc, html
from segmentation.models import model_names
from utils.app_modes import App_modes

btns = [
    'drawline',
    'drawopenpath',
    'drawclosedpath',
    'drawcircle',
    'drawrect',
    'eraseshape',
]

hyperspectral_image = dbc.Card([
    dbc.CardHeader('Image panel'),
    dbc.CardBody(
        dcc.Graph(
            id='x_map',
            animate = False,
            style={'height':'70vh', 'width':'63vw'},
            config={'modeBarButtonsToAdd': btns, 'responsive': True}
        ),
    )
])