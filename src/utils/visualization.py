import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from PIL import Image
from matplotlib import cm
from typing import Optional, Iterable

def plot_map(values, colormap=cm.Reds):
    values = np.uint8(colormap(values / values.max(), alpha=1.) * 255)
    fig = px.imshow(img=values, labels={})

    fig.update_traces(
        hovertemplate='<',
        hoverinfo='skip',
    )

    fig.update_layout(
        dragmode='drawopenpath',
        xaxis=dict(visible=False,),
        yaxis=dict(visible=False, scaleanchor='x',),
    )
    return fig

def plot_spectra(spectra: np.ndarray,
                 calibration: Optional[Iterable]=None,
                 labels: Optional[Iterable[str]]=None,
                 colormap=px.colors.qualitative.Set2,
                 axes_titles: bool=True,
                 opacity: float = 1.,
                 ):
    if calibration is None:
        calibration = np.arange(len(spectra[0]))
    if labels is None:
        labels = ["class {}".format(x+1) for x in range(len(spectra))]
    fig = go.Figure()
    for i in range(len(spectra)):
        fig.add_trace(
            go.Scatter(
                x = calibration,
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
        y=[0, 1, 2],
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


def plot_labels_map(y_true, mask, num_classes):
    img = cm.Set1(y_true / (num_classes), alpha=1) * 255
    img[y_true == -2, :] = (128, 128, 128, 255)
    img = np.where(mask == -2, 128, img)

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    add_legend(fig, num_classes)

    return fig


def plot_output_map(y, mask, manual_labels, num_classes):
    manual_labels = cm.Set1(manual_labels / (num_classes), alpha=1.) * 255
    y = cm.Set1(y / (num_classes), alpha=.8) * 255
    img = np.where(mask >= 0, manual_labels, y)
    img = np.where(mask == -2, 128, img)
    
    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    add_legend(fig, num_classes)

    return fig


def plot_values_map(spectra_image, manual_labels, mask, num_classes):
    manual_labels = cm.Set1(manual_labels / (num_classes), alpha=1.) * 255
    spectra_image = cm.Reds((spectra_image - spectra_image.min()) / spectra_image.max(), alpha=1.) * 255
    img = np.where(mask >= 0, manual_labels, spectra_image)
    img = np.where(mask == -2, 128, img)

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    add_legend(fig, num_classes)
    add_colorbar(fig, spectra_image.min(), spectra_image.max())

    return fig


def draw_hyperspectral_image(img, zmin, zmax, reset_ui, state, num_classes):
    fig = go.Figure()
    fig.add_trace(go.Image(z=img, hovertemplate='x: %{x} <br>y: %{y}'))

    add_colorbar(fig, zmin, zmax)

    fig.update_layout(
        legend_orientation='h',
        template='plotly_white',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        uirevision=reset_ui,
        updatemenus=list([
            dict(type = "buttons",
                direction = "down",
                active = 0,
                showactive = True,
                x = 0,
                y = 1,
                buttons = [
                    dict(
                        label = f'Class {i}',
                        method = "relayout", 
                        args = [{'newshape.line.color': px.colors.qualitative.Set1[i], 'newshape.label.text': str(i)}]
                    ) for i in range(1, num_classes + 1)
                ] + [
                    dict(
                        label = f'Remove label',
                        method = "relayout", 
                        args = [{'newshape.line.color': px.colors.qualitative.Set1[-1], 'newshape.label.text': '-1'}]
                    )
                ]
            ),
        ]),
    )

    if state is None:
        fig.update_layout(
            dragmode='drawclosedpath',
            newshape=dict(
                line=dict(color=px.colors.qualitative.Set1[1]),
                label=dict(text='1')
            ),
        )
    else:
        fig.update_layout(
            dragmode=state['layout']['dragmode'],
            newshape=state['layout']['newshape'],
        )

    return fig