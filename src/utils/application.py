import numpy as np

def mouse_path_to_indices(path):
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return list(map(tuple, np.rint(np.array(indices_str, dtype=float)).astype(int).tolist()))