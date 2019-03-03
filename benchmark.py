import OpenEXR
import Imath
from typing import Dict
import numpy as np


float_pixel = Imath.PixelType(Imath.PixelType.FLOAT)
half_pixel = Imath.PixelType(Imath.PixelType.HALF)
channels: [int] = list(range(400, 700, 5))


class SpectralImage:
    def __init__(self, file_path):
        self.img_file = OpenEXR.InputFile(file_path)
        dw = self.img_file.header()['dataWindow']
        self.img_shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        wave_length_map: Dict[int, np.ndarray] = {}
        for c in channels:
            buffed = self.img_file.channel(str(c), half_pixel)
            channel_data = np.frombuffer(buffed, dtype=np.float16)
            channel_data.shape = self.img_shape
            wave_length_map[c] = channel_data
        self.wave_length_map = wave_length_map

    def __getitem__(self, wavelength: int):
        return self.wave_length_map[wavelength]

    @property
    def header(self):
        return self.img_file.header()


class BaseBench:
    def __init__(self):
        pass
