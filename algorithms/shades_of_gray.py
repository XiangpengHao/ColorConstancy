from benchmark import BaseBench
import numpy as np
from images import RGBImage, SpectralImage

SPECTRUM_LENGTH = 61


class AverageRGB(BaseBench):
    NAME = 'average_rgb'

    def run(self):
        for img in self.img_list:
            cur_img = RGBImage.NewFromFile(f'{self.input_dir}/{img}')
            estimated = np.zeros(3)
            for i in range(3):
                estimated = np.average(cur_img.img_data[:, :, i])
            shape = cur_img.img_shape
            reflectance = np.zeros((*shape, 3))
            for i in range(shape[0]):
                for j in range(shape[1]):
                    reflectance[i, j, :] = cur_img.img_data[i,
                                                            j, :]/estimated
            RGBImage.NewFromArray(reflectance).dump_file(
                f'{self.output_dir}/{img}')


class MaxRGB(BaseBench):
    NAME = 'max_rgb'

    def run(self):
        for img in self.img_list:
            cur_img = RGBImage.NewFromFile(f'{self.input_dir}/{img}')
            estimated = np.zeros(3)
            for i in range(3):
                estimated[i] = np.max(cur_img.img_data[:, :, i])

            shape = cur_img.img_shape
            reflectance = np.zeros((*shape, 3))
            for i in range(shape[0]):
                for j in range(shape[1]):
                    reflectance[i, j, :] = cur_img.img_data[i,
                                                            j, :]/estimated
            RGBImage.NewFromArray(reflectance).dump_file(
                f'{self.output_dir}/{img}')


class PNorm(BaseBench):
    NAME = 'p_norm'

    P = 6

    def run(self):
        for img in self.img_list:
            cur_img = RGBImage.NewFromFile(f'{self.input_dir}/{img}')
            estimated = np.zeros(3)
            img_shape = cur_img.img_shape

            N = img_shape[0]*img_shape[1]
            N_1_p = np.power(N, 1/self.P)
            for i in range(3):
                rgb_value = cur_img.img_data[:, :, i]
                rgb_value.shape = (img_shape[0]*img_shape[1],)
                rgb_value = rgb_value.astype(np.float64)
                estimated[i] = np.linalg.norm(rgb_value, self.P)/N_1_p

            reflectance = np.zeros((*img_shape, 3))
            for i in range(img_shape[0]):
                for j in range(img_shape[1]):
                    reflectance[i, j, :] = cur_img.img_data[i,
                                                            j, :]/estimated
            RGBImage.NewFromArray(reflectance).dump_file(
                f'{self.output_dir}/{img}'
            )
