from benchmark import SpectralImage, BaseBench
import numpy as np

SPECTRUM_LENGTH = 61


class ShadesOfGray(BaseBench):
    def get_test_reflectance(self)->SpectralImage:
        estimated = np.ndarray(SPECTRUM_LENGTH)
        for i in range(SPECTRUM_LENGTH):
            estimated[i] = np.average(self.test_img.spectrum[:, :, i])

        shape = self.test_img.img_shape
        reflectance = np.zeros((*shape, SPECTRUM_LENGTH))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.test_img.spectrum[i,
                                                              j, :]/estimated
        return SpectralImage.NewFromSpectrum(reflectance)
