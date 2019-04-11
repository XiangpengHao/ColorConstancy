import OpenEXR
import Imath
import numpy as np
from color_space import Spectrum, RGB, XYZ
from images import RGBImage, SpectralImage
import matplotlib
import matplotlib.pyplot as plt
import os
import functools

from numba import jit

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


@jit(nopython=True, fastmath=True)
def metric_angle(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if(norm_v1 == 0 or norm_v2 == 0):
        return 0
    return np.degrees(np.arccos(np.dot(v1, v2) /
                                (norm_v1*norm_v2)))


@jit(nopython=True, fastmath=True)
def metric_chrom_distance(v1, v2):
    if sum(v1) == 0 or sum(v2) == 0:
        return 0
    normed_r1 = v1/sum(v1)
    normed_r2 = v2/sum(v2)
    return np.linalg.norm(normed_r1[:-1]-normed_r2[:-1])


@jit(fastmath=True, nopython=True)
def get_error(img1: np.ndarray, img2: np.ndarray, metric):
    img_shape = img1.shape
    result = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            result[i, j] = metric(img1[i, j], img2[i, j])
    return result


class BaseBench:

    NAME: str = ""

    angular_error: np.ndarray = None

    def __init__(self, input_dir: str, output_dir: str, groundtruth_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.groundtruth_dir = groundtruth_dir

        self.img_list = os.listdir(input_dir)

        if not os.path.exists(self.output_dir):
            os.mkdir(output_dir)

        self.curr_idx: int = -1
        self.curr_refl: RGBImage = None

        self._groundtruth_cache = {}
        self._refl_cache = {}

    def get_emission(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    def run(self):
        raise NotImplementedError("function should be overrided")

    @property
    def curr_img_name(self):
        return self.img_list[self.curr_idx]

    def has_next(self):
        return (self.curr_idx + 1) < len(self.img_list)

    def get_next_refl(self) -> RGBImage:
        if not self.has_next():
            return None
        self.curr_idx += 1
        file_name = self.img_list[self.curr_idx]
        if file_name not in self._refl_cache:
            next_file_path = f'{self.output_dir}/{file_name}'
            self.curr_refl = RGBImage.NewFromFile(next_file_path)
            self._refl_cache[file_name] = self.curr_refl
        return self.curr_refl

    def get_error(self, metric=metric_angle) -> np.ndarray:
        if(self.curr_refl == None):
            return None
        ground_truth = self.get_groundtruth(self.curr_img_name)
        return get_error(self.curr_refl.img_data, ground_truth.img_data, metric)

    def get_groundtruth(self, curr_img_name: str):
        if curr_img_name not in self._groundtruth_cache:
            curr_img_prefix = curr_img_name.split('.')[0]
            self._groundtruth_cache[curr_img_name] = RGBImage.NewFromFile(
                f'{self.groundtruth_dir}/{curr_img_prefix}_truth.exr')
        return self._groundtruth_cache[curr_img_name]

    def draw_heatmap(self, error_map, output_prefix: str):
        plt.figure()
        plt.imshow(error_map, cmap='hot')
        plt.title(f'mean error: {np.average(error_map)}')
        plt.colorbar()
        # plt.show()
        plt.savefig(f'{self.output_dir}/{output_prefix}_{self.NAME}_hm.png')
