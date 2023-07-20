import numpy as np
from skimage import draw
from scipy import ndimage


def svg_path_to_indices(path):
    """
    https://dash.plotly.com/annotations
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(np.int)


def rasterize_and_draw(shape, img):
    if shape['type'] == 'path' and 'fillrule' in shape:
        mask = draw_closedpath_mask(shape['path'], img.shape)
    elif shape['type'] == 'path':
        mask = draw_openpath_mask(shape['path'], img.shape, int(shape['line']['width']))
    elif shape['type'] == 'rect':
        mask = draw_rect_mask(shape, img.shape)
    elif shape['type'] == 'line':
        mask = draw_line_mask(shape, img.shape)
    else:
        raise NotImplemented
    img[mask] = int(shape['label']['text'])
    return img


def clip_coords(rr, cc, dim):
    rr[rr >= dim[0]] = dim[0] - 1
    cc[cc >= dim[1]] = dim[1] - 1
    return rr, cc


def draw_closedpath_mask(path, dim):
    """
    https://dash.plotly.com/annotations
    """
    cols, rows = svg_path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    rr, cc = clip_coords(rr, cc, dim)
    mask = np.zeros(dim, dtype=np.bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


def draw_openpath_mask(path, dim, width):
    # TODO width
    cols, rows = svg_path_to_indices(path).T
    mask = np.zeros(dim, dtype=np.bool)
    for i in range(len(cols) - 1):
        rr, cc = draw.line(rows[i], cols[i], rows[i + 1], cols[i + 1])
        rr, cc = clip_coords(rr, cc, dim)
        mask[rr, cc] = True
    return mask


def draw_rect_mask(shape, dim):
    # coordinates follow different convention
    y0, y1, x0, x1 = shape['x0'], shape['x1'], shape['y0'], shape['y1']
    mask = np.zeros(dim, dtype=np.bool)
    rr, cc = draw.rectangle(start=(x0, y0), end=(x1, y1))
    rr, cc = clip_coords(rr, cc, dim)
    mask[rr, cc] = True
    return mask


def draw_line_mask(shape, dim):
    # coordinates follow different convention
    y0, y1, x0, x1 = round(shape['x0']), round(shape['x1']), round(shape['y0']), round(shape['y1'])
    mask = np.zeros(dim, dtype=np.bool)
    rr, cc = draw.line(x0, y0, x1, y1)
    rr, cc = clip_coords(rr, cc, dim)
    mask[rr, cc] = True
    return mask