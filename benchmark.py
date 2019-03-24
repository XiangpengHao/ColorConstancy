import OpenEXR
import Imath
import numpy as np
from color_space import Spectrum, RGB, XYZ
from images import RGBImage, SpectralImage
import matplotlib
import matplotlib.pyplot as plt

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

    def __init__(self, test_img: RGBImage):
        self.test_img = test_img

    def get_reflectance(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    def get_emission(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    def get_best_single_illuminant(self, ground_truth: RGBImage) -> np.array:
        rv = []
        for i in range(3):
            A = self.test_img[:, :, i].flatten()
            B = ground_truth[:, :, i].flatten()
            rv.append(np.dot(A, B)/np.dot(B, B))
        return np.array(rv)

    def adjust_single_illuminant(self, illuminant: np.array) -> RGBImage:
        shape = self.test_img.img_shape
        reflectance = np.zeros((*shape, 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.test_img.img_data[i,
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
