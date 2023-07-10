import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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


def plot_values_map(X, wave_mask, mask):
    values = X[:, :, wave_mask].sum(axis=2)
    img = np.where(mask >= 0, mask, values)

    img = np.where(mask == -2, 0, img)