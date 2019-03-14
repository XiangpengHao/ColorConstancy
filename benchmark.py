import OpenEXR
import Imath
import numpy as np
from color_space import Spectrum, RGB, XYZ
from images import RGBImage, SpectralImage


class BaseBench:

    NAME: str = ""

    reflectance_map: RGBImage = None

    def __init__(self, test_img: RGBImage, truth_img: RGBImage):
        self.test_img = test_img
        self.ground_truth = truth_img

    def get_test_reflectance(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    def get_test_emission_map(self) -> RGBImage:
        raise NotImplementedError("function should be overrided")

    @staticmethod
    def get_angluar_error(scene: RGBImage, ground_truth: RGBImage) -> float:
        pass

    @staticmethod
    def get_distance_error(scene: RGBImage, ground_truth: RGBImage) -> float:
        pass
