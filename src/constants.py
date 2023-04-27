from matplotlib import cm

DIM = (70, 70) # hyperspectral map dimensions
CLASS_COLORS = cm.Set1
INTENSITY_COLORS = cm.Reds
RANGE_SLIDER_COLORS = ['red']

GRAPH_STYLE = dict()
"""
dict(
    template='plotly_dark',
    plot_bgcolor= 'rgba(0, 0, 0, 0)',
    paper_bgcolor= 'rgba(0, 0, 0, 0)',
    margin=dict(l=0, r=0, b=0, t=0,),
)
"""