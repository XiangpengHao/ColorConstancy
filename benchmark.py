import OpenEXR
import Imath
from typing import Dict
import numpy as np
from color_space import Spectrum, RGB, XYZ
import os


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
            wave_length_map[c] = channel_data.reshape((*self.img_shape, 1))

        # in case the wave length in the headers are not sorted
        sorted_waves = sorted(wave_length_map.keys())
        spectrum = wave_length_map[sorted_waves[0]]
        for key in sorted_waves[1:]:
            spectrum = np.append(spectrum, wave_length_map[key], axis=2)
        spectrum = np.append(
            spectrum, wave_length_map[sorted_waves[-1]], axis=2)
        self.spectrum = spectrum
        self.wavelength_map = wave_length_map

    def dump_to_sRGB_image(self, output: str):
        rgb_image = np.zeros(
            (self.img_shape[0], self.img_shape[1], 3), dtype=np.double)
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                xyz = Spectrum.spec_to_xyz(self.spectrum[i, j, :])
                rgb_image[i, j, :] = xyz.to_linear_rgb().np_rgb

        header = OpenEXR.Header(self.img_shape[1], self.img_shape[0])
        half_chan = Imath.Channel(half_pixel)
        header['channels'] = dict([(c, half_chan) for c in 'RGB'])
        exr = OpenEXR.OutputFile(output, header)
        exr.writePixels({'R': rgb_image[:, :, 0].astype(np.float16).tostring(),
                         'G': rgb_image[:, :, 1].astype(np.float16).tostring(),
                         'B': rgb_image[:, :, 2].astype(np.float16).tostring()})
        exr.close()

    @property
    def header(self):
        return self.img_file.header()

    def __getitem__(self, wavelength: int):
        return self.wavelength_map[wavelength]


class BaseBench:
    def __init__(self, test_img: SpectralImage, truth_img: SpectralImage):
        pass

    def get_test_reflectance(self)->SpectralImage:
        pass

    def get_test_emission_map(self)->SpectralImage:
        pass

    def get_score(self)->float:
        pass
