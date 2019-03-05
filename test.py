from benchmark import SpectralImage, BaseBench
from algorithms.shades_of_gray import AverageRGB, MaxRGB


algs: [BaseBench] = [AverageRGB, MaxRGB]
img = SpectralImage.NewFromFile('test_fixtures/vp_0_combined.exr')

for algorithm in algs:
    refl = algorithm(img, None).get_test_reflectance()
    refl.dump_to_sRGB_image(f'{algorithm.NAME}_refl.exr')
