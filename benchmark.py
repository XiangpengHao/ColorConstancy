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
    def __init__(self, img_file=None, spectrum=np.ndarray):
        self.img_file = img_file
        self.img_shape = spectrum.shape[:-1]
        self.spectrum = spectrum

    @classmethod
    def NewFromSpectrum(cls, spectrum: np.ndarray):
        return SpectralImage(None, spectrum)

    @classmethod
    def NewFromFile(cls, file_path: str):
        img_file = OpenEXR.InputFile(file_path)
        dw = img_file.header()['dataWindow']
        img_shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        wave_length_map: Dict[int, np.ndarray] = {}
        for c in channels:
            buffed = img_file.channel(str(c), half_pixel)
            channel_data = np.frombuffer(buffed, dtype=np.float16)
            wave_length_map[c] = channel_data.reshape((*img_shape, 1))

        # in case the wave length in the headers are not sorted
        sorted_waves = sorted(wave_length_map.keys())
        spectrum = wave_length_map[sorted_waves[0]]
        for key in sorted_waves[1:]:
            spectrum = np.append(spectrum, wave_length_map[key], axis=2)
        # pad to 400, 700 inclusive
        spectrum = np.append(
            spectrum, wave_length_map[sorted_waves[-1]], axis=2)
        return SpectralImage(img_file, spectrum)

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
        return self.spectrum[:, :, (wavelength-400)//5]


class BaseBench:

    NAME: str

    def __init__(self, test_img: SpectralImage, truth_img: SpectralImage):
        self.test_img = test_img
        self.ground_truth = truth_img

    def get_test_reflectance(self) -> SpectralImage:
        raise NotImplementedError("function should be overrided")

    def get_test_emission_map(self) -> SpectralImage:
        raise NotImplementedError("function should be overrided")

    # probably we should have a evalute
    # interface for different evalute methods
    def get_score(self) -> float:
        # currently we do euclidean distance
        pass
