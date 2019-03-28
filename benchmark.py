import OpenEXR
import Imath
import numpy as np
from color_space import Spectrum, RGB, XYZ
from images import RGBImage, SpectralImage
import matplotlib
import matplotlib.pyplot as plt
import os

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_angle(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if(norm_v1 == 0 or norm_v2 == 0):
        return 0
    return np.degrees(np.arccos(np.dot(v1, v2) /
                                (np.linalg.norm(v1)*np.linalg.norm(v2))))


def get_chrom_distance(v1, v2):
    # this is by no means reasonable, but it happens
    if sum(v1) == 0 or sum(v2) == 0:
        return 0
    normed_r1 = v1/sum(v1)
    normed_r2 = v2/sum(v2)
    return np.linalg.norm(normed_r1[:-1]-normed_r2[:-1])


class BaseBench:

    NAME: str = ""

    reflectance_map: RGBImage = None
    angular_error: np.ndarray = None

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_list = os.listdir(input_dir)

        self.curr_idx = 0
        self.curr_img = None

    def get_emission(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    def run(self):
        raise NotImplementedError("function should be overrided")

    def has_next(self):
        return self.curr_idx < len(self.img_list)

    def get_next_refl(self) -> RGBImage:
        if not self.has_next():
            return None
        self.curr_img = RGBImage.NewFromFile(
            f'{self.output_dir}/{self.img_list[self.curr_idx]}')
        return self.curr_img

    def get_best_single_adjustment(self, ground_truth: RGBImage) -> np.array:
        shape = self.curr_img.img_shape
        xyz_img = self.curr_img.get_xyz_image()
        xyz_truth = ground_truth.get_xyz_image()

        rv = []
        for i in range(3):
            A = xyz_img[:, :, i].flatten()
            B = xyz_truth[:, :, i].flatten()
            rv.append(np.dot(A, A)/np.dot(B, A))

        adjusted_img = np.zeros((*shape, 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                raw_xyz = RGB(*self.curr_img[i, j, :], True).to_xyz()
                adjusted_xyz = np.multiply(xyz_img[i, j, :], rv) * \
                    (raw_xyz.y/xyz_img[i, j, 1])
                adjusted_img[i, j, :] = XYZ(*adjusted_xyz).to_rgb().np_rgb

        self.reflectance_map = RGBImage.NewFromArray(adjusted_img)
        return self.reflectance_map

    def adjust_single_illuminant(self, illuminant: np.array) -> RGBImage:
        shape = self.curr_img.img_shape

        reflectance = np.zeros((*shape, 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.curr_img.img_data[i,
                                                              j, :]/illuminant
        self.reflectance_map = RGBImage.NewFromArray(reflectance)
        return self.reflectance_map

    def get_error(self, ground_truth: RGBImage, metric=get_angle) -> np.ndarray:
        if(self.reflectance_map == None):
            self.reflectance_map = self.get_reflectance()
        img_shape = self.reflectance_map.img_shape
        result = np.zeros(img_shape)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                result[i, j] = metric(
                    self.reflectance_map[i, j], ground_truth[i, j])

        self.angular_error = result
        return result

    def draw_heatmap(self, error_map, output_prefix: str):
        plt.figure()
        plt.imshow(error_map, cmap='hot')
        plt.title(f'mean error: {np.average(error_map)}')
        plt.colorbar()
        # plt.show()
        plt.savefig(f'dist/{output_prefix}_{self.NAME}_hm.png')
