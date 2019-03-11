import numpy as np
import OpenEXR
import Imath
from typing import Dict
from observers import BaseObserver, two_degree_observer


float_pixel = Imath.PixelType(Imath.PixelType.FLOAT)
half_pixel = Imath.PixelType(Imath.PixelType.HALF)
channels: [int] = list(range(400, 700, 5))


class sRGBImage:
    def __init__(self, img_data: np.ndarray, file_path: str = None):
        self.file_path = file_path
        self.img_data = img_data
        self.img_shape = img_data.shape[:-1]

    @classmethod
    def NewFromArray(cls, img_data: np.ndarray):
        return sRGBImage(img_data, None)

    @classmethod
    def NewFromFile(cls, file_path: str):
        img_file = OpenEXR.InputFile(file_path)
        dw = img_file.header()['dataWindow']
        img_shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        img_data = np.zeros((*img_shape, 3), dtype=float)
        for i, c in enumerate("RGB"):
            buffed = img_file.channel(c, half_pixel)
            channel_data = np.frombuffer(buffed, dtype=np.float16)
            img_data[:, :, i] = channel_data.reshape(img_shape)
        return sRGBImage(img_data)

    def dump_file(self, output: str):
        header = OpenEXR.Header(self.img_shape[1], self.img_shape[0])
        half_chan = Imath.Channel(half_pixel)
        header['channels'] = dict([(c, half_chan) for c in 'RGB'])
        exr = OpenEXR.OutputFile(output, header)
        exr.writePixels({'R': self.img_data[:, :, 0].astype(np.float16).tostring(),
                         'G': self.img_data[:, :, 1].astype(np.float16).tostring(),
                         'B': self.img_data[:, :, 2].astype(np.float16).tostring()})
        exr.close()

    @property
    def r(self):
        return self.img_data[:, :, 0]

    @property
    def g(self):
        return self.img_data[:, :, 1]

    @property
    def b(self):
        return self.img_data[:, :, 2]

    def __getitem__(self, pixel):
        return self.img_data[pixel[0], pixel[1], :]


class SpectralImage:
    def __init__(self, img_file: str = None, spectrum: np.ndarray = None):
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

    def to_srgb(self, observer: BaseObserver = two_degree_observer) -> sRGBImage:
        rgb_img = np.zeros(
            (self.img_shape[0], self.img_shape[1], 3), dtype=np.double)
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                rgb_img[i, j, :] = observer.spec_to_srgb(
                    self.spectrum[i, j, :]).np_rgb
        return sRGBImage(rgb_img)

    @property
    def header(self):
        return self.img_file.header()

    def __getitem__(self, wavelength: int):
        return self.spectrum[:, :, (wavelength-400)//5]
