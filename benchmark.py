import OpenEXR
import Imath
import numpy as np
from color_space import Spectrum, RGB, XYZ
from images import RGBImage, SpectralImage
import matplotlib
import matplotlib.pyplot as plt


class BaseBench:

    NAME: str = ""

    reflectance_map: RGBImage = None
    angular_error: np.ndarray = None

    def __init__(self, test_img):
        self.test_img = test_img

    def get_reflectance(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    def get_emission(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    def get_angular_error(self, ground_truth: RGBImage) -> np.ndarray:
        if(self.reflectance_map == None):
            self.reflectance_map = self.get_reflectance()
        img_shape = self.reflectance_map.img_shape
        result = np.zeros(img_shape)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                result[i, j] = BaseBench.get_angle(
                    self.reflectance_map[i, j], ground_truth[i, j])
        self.angular_error = result
        return result

    @staticmethod
    def draw_heatmap(error_map, output: str):
        plt.imshow(error_map, 'hot')
        plt.show()

    @staticmethod
    def get_angle(v1, v2):
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if(norm_v1 == 0 or norm_v2 == 0):
            return 0
        return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

    @staticmethod
    def get_distance_error(scene: RGBImage, ground_truth: RGBImage) -> float:
        pass
