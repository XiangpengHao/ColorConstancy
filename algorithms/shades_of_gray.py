from benchmark import SpectralImage, BaseBench
import numpy as np

SPECTRUM_LENGTH = 61


class AverageRGB(BaseBench):
    NAME = 'average_rgb'

    def get_test_reflectance(self) -> SpectralImage:
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


class MaxRGB(BaseBench):
    NAME = 'max_rgb'

    def get_test_reflectance(self) -> SpectralImage:
        estimated = np.ndarray(SPECTRUM_LENGTH)
        for i in range(SPECTRUM_LENGTH):
            estimated[i] = np.max(self.test_img.spectrum[:, :, i])

        shape = self.test_img.img_shape
        reflectance = np.zeros((*shape, SPECTRUM_LENGTH))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.test_img.spectrum[i,
                                                              j, :]/estimated
        return SpectralImage.NewFromSpectrum(reflectance)


class PNorm(BaseBench):
    NAME = 'p_norm'

    P = 5

    def get_test_reflectance(self) -> SpectralImage:
        estimated = np.ndarray(SPECTRUM_LENGTH)

        img_shape = self.test_img.img_shape

        N = img_shape[0]*img_shape[1]
        N_1_p = np.power(N, 1/self.P)
        for i in range(SPECTRUM_LENGTH):
            wavelength = self.test_img.spectrum[:, :, i]
            wavelength.shape = (img_shape[0]*img_shape[1],)
            wavelength = wavelength.astype(np.float64)
            estimated[i] = np.linalg.norm(wavelength, self.P)/N_1_p

        shape = self.test_img.img_shape
        reflectance = np.zeros((*shape, SPECTRUM_LENGTH))
        for i in range(shape[0]):
            for j in range(shape[1]):
                reflectance[i, j, :] = self.test_img.spectrum[i,
                                                              j, :]/estimated
        return SpectralImage.NewFromSpectrum(reflectance)
