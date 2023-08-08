import numpy as np
from matplotlib import cm

class Set1Colormap:
    @classmethod
    def to_plotly(cls, tuple):
        return 'rgba{}'.format(tuple)
    

    def __init__(self, num_classes):
        self.num_classes = num_classes


    def get_empty_color_tuple(self, alpha=1):
        return (cm.Set1(np.zeros((1,)), alpha=alpha) * 255)[0]


    def get_color_tuple(self, idx, alpha=1):
        return (cm.Set1(np.array([idx]) / (self.num_classes + 1), alpha=alpha) * 255)[0]


    def get_color_tuples(self, array, alpha=1):
        return cm.Set1(array / (self.num_classes + 1), alpha=alpha) * 255
    

    def get_plotly_color(self, idx, alpha=1):
        return self.to_plotly(tuple(self.get_color_tuple(idx, alpha)))
    

    def get_empty_plotly_color(self, alpha=1):
        return self.to_plotly(tuple(self.get_empty_color_tuple(alpha)))