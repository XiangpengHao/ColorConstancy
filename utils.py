import numpy as np
from typing import Tuple, Union, List
import warnings
import math
import functools

np.seterr(all='raise')

MAX_NATURAL = 4096

MAX_LIGHT_POWER = 120

XYZ2RGB = np.asarray([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
])

RGB2XYZ = np.asarray([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
])

R_xy = np.asarray((0.6400, 0.3300))
G_xy = np.asarray((0.3000, 0.6000))
B_xy = np.asarray((0.1500, 0.0600))


def xyz_in_srgb(xyz: np.ndarray) -> bool:
    xyz = xyz / sum(xyz)
    xy_point = np.asarray((xyz[0], xyz[1]))

    def same_side(p1, p2, a, b) -> bool:
        cp1 = np.cross(b - a, p1 - a)
        cp2 = np.cross(b - a, p2 - a)
        return float(np.dot(cp1, cp2)) >= 0

    return same_side(xy_point, R_xy, G_xy, B_xy) and \
        same_side(xy_point, G_xy, R_xy, B_xy) and \
        same_side(xy_point, B_xy, R_xy, G_xy)


def gamma_correct(v: float) -> float:
    if v <= 0.0031308:
        return 12.92 * v
    else:
        return (1 + 0.055) * (v ** (1 / 2.4)) - 0.055


@functools.lru_cache(maxsize=256)
def gamma_correct_rev(v: float) -> float:
    if v <= 0.04045:
        return v / 12.92
    else:
        return ((v + 0.055) / (1 + 0.055)) ** 2.4


np_gamma_correct_rev = np.vectorize(gamma_correct_rev)
np_gamma_correct = np.vectorize(gamma_correct)


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    rgb_de_gamma = np_gamma_correct_rev(rgb)
    xyz = np.matmul(RGB2XYZ, rgb_de_gamma.T)
    return xyz


def distance_v2(va: Tuple[float, float], vb: Tuple[float, float]):
    return math.sqrt((va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2)


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    if np.min(xyz) < 0 and np.max(xyz) > 1:
        warnings.warn(f'unexpected xyz: {xyz}', stacklevel=2)
    rgb_linear = np.matmul(XYZ2RGB, np.asarray(xyz).T)
    return np.asarray([gamma_correct(c) for c in rgb_linear])


def xyz_to_rgb(xyz: Tuple[float, float, float], magic_n: float) -> Tuple[float, float, float]:
    xyz_array = np.asarray(xyz)
    xyz_array /= magic_n
    if min(xyz_array) < 0 or max(xyz_array) > 1:
        warnings.warn(f"abnormal xyz: {xyz_array}", stacklevel=2)
    rgb = np.matmul(XYZ2RGB, np.asarray(xyz_array).T)

    return (gamma_correct(rgb[0]),
            gamma_correct(rgb[1]),
            gamma_correct(rgb[2]))


def get_natural_spectral(file_name):
    with open(file_name) as f:
        data = [float(x) / MAX_NATURAL for x in f.readlines()[:-1]]
    return np.asarray(data)


def get_light_spectral(file_name):
    with open(file_name) as f:
        data = [float(x) / MAX_LIGHT_POWER for x in f.readlines()]
    return np.asarray(data)



if __name__ == '__main__':
    print(srgb_to_xyz(np.asarray([0.300, 0.300, 0.4])))
