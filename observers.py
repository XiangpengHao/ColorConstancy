import json
import numpy as np
from color_space import RGB, XYZ


class BaseObserver:
    def __init__(self, file_path):
        self.observer = {
            "x": json.load(open(f"{file_path}/x.json")),
            "y": json.load(open(f"{file_path}/y.json")),
            "z": json.load(open(f"{file_path}/z.json"))
        }

    def spec_to_srgb(self, spectrum: np.ndarray) -> RGB:
        raise NotImplementedError


class TwoDegreeObserver(BaseObserver):
    def spec_to_srgb(self, spectrum) -> RGB:
        x_val: float = np.dot(spectrum, self.observer['x']['data'])
        y_val: float = np.dot(spectrum, self.observer['y']['data'])
        z_val: float = np.dot(spectrum, self.observer['z']['data'])
        return XYZ(x_val, y_val, z_val).to_srgb()


two_degree_observer = TwoDegreeObserver("data/two_degree")
