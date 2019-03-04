from __future__ import annotations
from typing import List, Union, Dict
import json
import numpy as np
import utils
from enum import Enum
import logging
import functools

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

color_match_dict = {
    "x": json.load(open("spec/x.json")),
    "y": json.load(open("spec/y.json")),
    "z": json.load(open("spec/z.json"))
}


class SpecType(Enum):
    REFLECTANCE = 0
    ILLUMINANT = 1
    COLOR_MATCH = 2


class RGB:
    def __init__(self, r: float, g: float, b: float, spec_type: SpecType = SpecType.REFLECTANCE):
        self.r: float = r
        self.g: float = g
        self.b: float = b
        self.spec_type: SpecType = spec_type
        self.np_rgb: np.ndarray = np.asarray([r, g, b])

    def to_xyz(self) -> XYZ:
        rgb_gamma_rev = [utils.gamma_correct_rev(v) for v in self.np_rgb]
        xyz = np.matmul(utils.RGB2XYZ, rgb_gamma_rev)
        return XYZ(xyz[0], xyz[1], xyz[2])

    def to_uint8(self, verbose=False) -> np.ndarray:
        return np.rint(self.np_rgb * 255).clip(0, 255).astype(np.uint8)

    def __str__(self):
        return f"({self.r}, {self.g}, {self.b})"

    __repr__ = __str__


class XYZ:
    def __init__(self, x: float, y: float, z: float, spec_type: SpecType = SpecType.REFLECTANCE):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.spec_type: SpecType = spec_type
        self.np_xyz: np.ndarray = np.asarray([x, y, z])

    def norm(self) -> XYZ:
        xyz_sum = sum(self.np_xyz)
        if xyz_sum == 0:
            return XYZ(0, 0, 0)
        return XYZ(self.x / xyz_sum, self.y / xyz_sum, self.z / xyz_sum)

    def to_srgb(self, norm: bool = False) -> RGB:
        if norm:
            xyz = self.norm().np_xyz
        else:
            xyz = self.np_xyz
        rgb = np.matmul(utils.XYZ2RGB, xyz)
        return RGB(utils.gamma_correct(rgb[0]),
                   utils.gamma_correct(rgb[1]),
                   utils.gamma_correct(rgb[2]))

    def to_linear_rgb(self, norm: bool=False)->RGB:
        if norm:
            xyz = self.norm().np_xyz
        else:
            xyz = self.np_xyz
        rbg = np.matmul(utils.XYZ2RGB, xyz)
        return RGB(rbg[0], rbg[1], rbg[2])

    @functools.lru_cache(maxsize=500)
    def in_srgb(self) -> bool:
        from utils import R_xy, G_xy, B_xy
        xyz_norm = self.norm()
        xy_point = xyz_norm.np_xyz[0:2]

        def same_side(p1, p2, a, b) -> bool:
            cp1 = np.cross(b - a, p1 - a)
            cp2 = np.cross(b - a, p2 - a)
            return float(np.dot(cp1, cp2)) >= 0

        return same_side(xy_point, R_xy, G_xy, B_xy) and \
            same_side(xy_point, G_xy, R_xy, B_xy) and \
            same_side(xy_point, B_xy, R_xy, G_xy)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    __repr__ = __str__


class Spectrum:
    def __init__(self, spec_file: Union[str, dict]):
        self.name: str
        self.type: str
        if type(spec_file) == str:
            with open(spec_file) as f:
                spec_data = json.load(f)
        else:
            spec_data = spec_file
        self.name: str = spec_data.get('name')
        self.type: str = spec_data.get('type')
        self.start_nm: int = spec_data.get('start_nm')
        self.end_nm: int = spec_data.get('end_nm')
        self._resolution: int = spec_data.get('resolution')
        self.rgb: RGB = RGB(*spec_data.get('rgb', [0, 0, 0]))
        self.xyz: XYZ = XYZ(*spec_data.get('xyz', [0, 0, 0]))
        self.np_xyz: np.ndarray = self.xyz.np_xyz
        self.np_rgb: np.ndarray = self.rgb.np_rgb
        self.type_max = spec_data.get('type_max')
        self.data: np.ndarray = np.asarray(spec_data.get('data'))

    @staticmethod
    def make_from_value(spec_data: Union[np.ndarray, List[float]],
                        name: str = "temp", spec_type: SpecType = SpecType.REFLECTANCE,
                        start_nm: int = 400, resolution: int = 5):
        raise NotImplementedError()

    @staticmethod
    def spec_to_xyz(spec_data: np.ndarray) -> XYZ:
        x_data: float = np.dot(spec_data, color_match_dict['x']["data"])
        y_data: float = np.dot(spec_data, color_match_dict['y']["data"])
        z_data: float = np.dot(spec_data, color_match_dict['z']["data"])
        return XYZ(x_data, y_data, z_data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start < self.start_nm or item.stop > self.end_nm:
                raise ArithmeticError("nm not in range")
            start_i = (item.start - self.start_nm) // self.resolution
            end_i = (item.stop - self.start_nm) // self.resolution
            print(self.data[start_i:end_i])
            return self.data[start_i:end_i]

    def to_xyz(self) -> XYZ:
        return self.spec_to_xyz(self.data)

    def to_rgb(self):
        xyz = self.to_xyz()
        return xyz.to_srgb()

    @property
    def dict(self):
        rv = {
            'name': self.name,
            'type': self.type,
            'start_nm': self.start_nm,
            'end_nm': self.end_nm,
            'resolution': self._resolution,
            'rgb': self.np_rgb.tolist(),
            'xyz': self.np_xyz.tolist(),
            'data': self.data.tolist(),
            'type_max': self.type_max
        }
        return rv

    @property
    def json(self):
        rv = self.dict
        return json.dumps(rv)

    def dump(self, output: str):
        pass

    @property
    def resolution(self):
        return self._resolution
