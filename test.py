from benchmark import SpectralImage, BaseBench
from algorithms.shades_of_gray import AverageRGB, MaxRGB, PNorm

test_file = 'test1.exr'

algs: [BaseBench] = [AverageRGB, MaxRGB, PNorm]
img = SpectralImage.NewFromFile(f'fixtures/{test_file}')

for algorithm in algs:
    refl = algorithm(img, None).get_test_reflectance()
    refl.dump_to_sRGB_image(f'dist/{test_file}_{algorithm.NAME}_refl.exr')
