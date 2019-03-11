import OpenEXR
import Imath
import numpy as np
from color_space import Spectrum, RGB, XYZ
from images import sRGBImage, SpectralImage


class BaseBench:

    NAME: str = ""

    def __init__(self, test_img: sRGBImage, truth_img: sRGBImage):
        self.test_img = test_img
        self.ground_truth = truth_img

    def get_test_reflectance(self) -> sRGBImage:
        raise NotImplementedError("function should be overrided")

    def get_test_emission_map(self) -> sRGBImage:
        raise NotImplementedError("function should be overrided")

    # probably we should have a evaluate
    # interface for different evaluate methods
    def get_score(self) -> float:
        # currently we do euclidean distance
        pass
