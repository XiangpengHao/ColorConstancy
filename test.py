from benchmark import SpectralImage, BaseBench
from algorithms.shades_of_gray import AverageRGB, MaxRGB, PNorm


algs: [BaseBench] = [PNorm]
img = SpectralImage.NewFromFile('fixtures/test0.exr')

for algorithm in algs:
    refl = algorithm(img, None).get_test_reflectance()
    refl.dump_to_sRGB_image(f'dist/{algorithm.NAME}_refl.exr')
