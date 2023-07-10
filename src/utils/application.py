import numpy as np

def mouse_path_to_indices(path):
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return list(map(tuple, np.rint(np.array(indices_str, dtype=float)).astype(int).tolist()))


def coordinates_from_hover_data(hover):
    return hover['points'][0]['y'], hover['points'][0]['x'] 