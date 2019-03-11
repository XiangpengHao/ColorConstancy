from benchmark import BaseBench
import numpy as np
from images import sRGBImage, SpectralImage

SPECTRUM_LENGTH = 61


class AverageRGB(BaseBench):
    NAME = 'average_rgb'

    def get_test_reflectance(self) -> sRGBImage:
        estimated = np.zeros(3)
        for i in range(3):
            estimated[i] = np.average(self.test_img.img_data[:, :, i])

        shape = self.test_img.img_shape
        reflectance = np.zeros((*shape, 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.test_img.img_data[i,
                                                              j, :]/estimated
        return sRGBImage.NewFromArray(reflectance)


class MaxRGB(BaseBench):
    NAME = 'max_rgb'

    def get_test_reflectance(self) -> sRGBImage:
        estimated = np.zeros(3)
        for i in range(3):
            estimated[i] = np.max(self.test_img.img_data[:, :, i])

        shape = self.test_img.img_shape
        reflectance = np.zeros((*shape, 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.test_img.img_data[i,
                                                              j, :]/estimated
        return sRGBImage.NewFromArray(reflectance)


class PNorm(BaseBench):
    NAME = 'p_norm'

    P = 5

    def get_test_reflectance(self) -> sRGBImage:
        estimated = np.zeros(3)

        img_shape = self.test_img.img_shape

        N = img_shape[0]*img_shape[1]
        N_1_p = np.power(N, 1/self.P)
        for i in range(3):
            rgb_value = self.test_img.img_data[:, :, i]
            rgb_value.shape = (img_shape[0]*img_shape[1],)
            rgb_value = rgb_value.astype(np.float64)
            estimated[i] = np.linalg.norm(rgb_value, self.P)/N_1_p

        shape = self.test_img.img_shape
        reflectance = np.zeros((*shape, 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.test_img.img_data[i,
                                                              j, :]/estimated
        return sRGBImage.NewFromArray(reflectance)
