import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from PIL import Image
from matplotlib import cm
from typing import Optional, Iterable


def plot_spectra(spectra: np.ndarray,
                 wavelengths: Optional[Iterable]=None,
                 labels: Optional[Iterable[str]]=None,
                 colormap=px.colors.qualitative.Set2,
                 axes_titles: bool=True,
                 opacity: float = 1.,
                 ):
    if wavelengths is None:
        wavelengths = np.arange(len(spectra[0]))
    if labels is None:
        labels = ["class {}".format(x+1) for x in range(len(spectra))]
    fig = go.Figure()
    for i in range(len(spectra)):
        fig.add_trace(
            go.Scatter(
                x = wavelengths,
                y = spectra[i],
                name = str(labels[i]),
                line = {'color': colormap[i % len(colormap)]},
                opacity=opacity,
            )
        )
    fig.update_layout(
        xaxis_title = "wavelengths (nm)" if axes_titles else "",
        yaxis_title = "relative intensity (-)" if axes_titles else "")
    return fig


def add_legend(fig, num_classes):
    for i in range(num_classes + 1):
        fig.add_trace(go.Scatter(x=[[0, 0, 0]], mode='markers',
                                marker_color=px.colors.qualitative.Set1[i], name=f'class {i}'))
    

def add_colorbar(fig, min_val, max_val):
    # TODO more robust solution
    # TODO tick styling
    fig.add_trace(go.Heatmap(
        x=[0, 1, 2],
        y=[0, 0, 0],
        opacity=0,
        z=[min_val, max_val, max_val],
        colorscale='reds',
        colorbar=dict(
            title="Total Intensity",  # TODO better title
            titleside="top",
            tickmode="array",
            tickvals=np.linspace(min_val, max_val, 10).tolist(),
            ticks="inside"
        )
    ))


def draw_hyperspectral_image(img, zmin, zmax, reset_ui, state, num_classes, colormap):
    fig = go.Figure()
    fig.add_trace(go.Image(z=img, hovertemplate='x: %{x} <br>y: %{y}', colormodel='rgba256'))

    add_colorbar(fig, zmin, zmax)

    num_btns = num_classes + 1
    y_offset = 0.1

    fig.update_layout(
        legend_orientation='h',
        template='plotly_white',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        uirevision=reset_ui,
        updatemenus=[
            dict(
                type = "buttons",
                x = 0,
                y = y_offset + (i / num_btns) * (1 - y_offset),
                bgcolor = colormap.get_plotly_color(i),
                showactive = False,
                buttons = [
                    dict(
                        label = f'Class {i}',
                        method = "relayout", 
                        args = [{'newshape.line.color': colormap.get_plotly_color(i), 'newshape.label.text': str(i)}]
                    )
                ]
            ) for i in range(1, num_btns) 
        ]  + [
            dict(
                type = "buttons",
                x = 0,
                y = y_offset,
                bgcolor = colormap.get_empty_plotly_color(),
                showactive = False,
                buttons = [
                    dict(
                        label = f'Remove label',
                        method = "relayout", 
                        args = [{'newshape.line.color': colormap.get_empty_plotly_color(), 'newshape.label.text': '-1'}]
                    )
                ]
            )
        ]
    )

    if state is None:
        fig.update_layout(
            dragmode='drawclosedpath',
            newshape=dict(
                line=dict(color=colormap.get_plotly_color(1)),
                label=dict(text='1')
            ),
        )
    else:
        fig.update_layout(
            dragmode=state['layout']['dragmode'],
            newshape=state['layout']['newshape'],
        )

    return fig