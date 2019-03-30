import numpy as np
import OpenEXR
import Imath
from typing import Dict
import os
import pickle
import logging
from observers import BaseObserver, two_degree_observer
from joblib import Parallel, delayed
from PIL import Image
from color_space import RGB, XYZ
from numba import jit

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

float_pixel = Imath.PixelType(Imath.PixelType.FLOAT)
half_pixel = Imath.PixelType(Imath.PixelType.HALF)
channels: [int] = list(range(400, 700, 5))


class RGBImage:
    def __init__(self, img_data: np.ndarray, file_path: str = None):
        self.file_path = file_path
        self.img_data = img_data
        self.img_shape = img_data.shape[:-1]

    @classmethod
    def NewFromArray(cls, img_data: np.ndarray):
        return RGBImage(img_data, None)

    @classmethod
    def NewFromFile(cls, file_path: str):
        if(file_path.endswith('exr')):
            # exr file loading
            img_file = OpenEXR.InputFile(file_path)
            dw = img_file.header()['dataWindow']
            img_shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

            img_data = np.zeros((*img_shape, 3), dtype=float)
            for i, c in enumerate("RGB"):
                buffed = img_file.channel(c, half_pixel)
                channel_data = np.frombuffer(buffed, dtype=np.float16)
                img_data[:, :, i] = channel_data.reshape(img_shape)
            return RGBImage(img_data)

        elif(file_path.endswith('png') or file_path.endswith('jpg')):
            # png file loading
            logging.warn(
                'loading png file is experimental, assuming linear RGB...')
            img = Image.open(file_path)
            return RGBImage(np.array(img))

    @jit(parallel=True, fastmath=True)
    def get_xyz_image(self) -> np.array:
        xyz_image = np.zeros((*self.img_shape, 3))
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                xyz_image[i, j, :] = RGB.static_to_normed_xyz(
                    self.img_data[i, j, :])
        return xyz_image

    def dump_file(self, output: str):
        if output.endswith('png'):
            image = Image.fromarray(
                np.rint(self.img_data).clip(0, 255).astype(np.uint8))
            image.save(output)
        elif output.endswith('exr'):
            header = OpenEXR.Header(self.img_shape[1], self.img_shape[0])
            half_chan = Imath.Channel(half_pixel)
            header['channels'] = dict([(c, half_chan) for c in 'RGB'])
            exr = OpenEXR.OutputFile(output, header)
            exr.writePixels({'R': self.img_data[:, :, 0].astype(np.float16).tostring(),
                             'G': self.img_data[:, :, 1].astype(np.float16).tostring(),
                             'B': self.img_data[:, :, 2].astype(np.float16).tostring()})
            exr.close()

    def dump_png(self, output: str):
        # here we use the vanilla method: map values to 0-255
        # save as linear rgb, not for display purpose
        total_pixels = self.img_shape[0]*self.img_shape[1]*3
        top_5_pixels = int(total_pixels*0.05)
        max_5_pixels = np.argpartition(
            self.img_data.ravel(), -top_5_pixels)[-top_5_pixels:]
        max_value = sorted(self.img_data.ravel()[max_5_pixels])[0]
        normalized = self.img_data/max_value
        image = Image.fromarray(
            np.rint(normalized*255).clip(0, 255).astype(np.uint8))
        image.save(output)

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
    def NewFromFile(cls, file_path: str, use_cache=True):
        # lets check if there's a cached file
        file_name = os.path.split(file_path)[-1]
        cached_path = f'dist/__cache__{file_name}.pkl'
        if os.path.exists(cached_path) and use_cache:
            logging.info('load from cached file %s', cached_path)
            spectrum = pickle.load(open(cached_path, 'rb'))
            return SpectralImage(file_path, spectrum)

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

        if use_cache:
            # let's cache the spectrum
            pickle.dump(spectrum, open(cached_path, 'wb'))
        return SpectralImage(file_path, spectrum)

    def to_rgb(self, observer: BaseObserver = two_degree_observer, use_cache=True) -> RGBImage:
        # check if there's a cache
        if use_cache:
            file_name = os.path.split(self.img_file)[-1]
            cached_path = f'dist/__cache__rgb__{file_name}.pkl'
            if os.path.exists(cached_path):
                logging.info('using cached rgb file')
                rgb_img = pickle.load(open(cached_path, 'rb'))
                return RGBImage(rgb_img)

        rgb_img = np.zeros(
            (self.img_shape[0], self.img_shape[1], 3), dtype=np.double)
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                rgb_img[i, j, :] = observer.spec_to_rgb(
                    self.spectrum[i, j, :]).np_rgb

        assert(np.amin(rgb_img) >= 0)
        if use_cache:
            pickle.dump(rgb_img, open(cached_path, 'wb'))
        return RGBImage(rgb_img)

    def apply_illuminant(self, illuminant):
        new_spectral = np.zeros((self.spectrum.shape))
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                new_spectral[i, j, :] = np.multiply(
                    self.spectrum[i, j, :], illuminant)
        return SpectralImage.NewFromSpectrum(new_spectral)

    @property
    def header(self):
        return self.img_file.header()

    def __getitem__(self, wavelength: int):
        return self.spectrum[:, :, (wavelength-400)//5]
