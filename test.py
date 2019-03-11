from benchmark import BaseBench
from algorithms.shades_of_gray import AverageRGB, MaxRGB, PNorm
from images import SpectralImage, sRGBImage

test_file = 'test1'

algs: [BaseBench] = [AverageRGB, MaxRGB, PNorm]
spectral_img = SpectralImage.NewFromFile(f'fixtures/{test_file}.exr')
rgb_img = spectral_img.to_srgb()

for algorithm in algs:
    refl = algorithm(rgb_img, None).get_test_reflectance()
    refl.dump_file(f'dist/{test_file}_{algorithm.NAME}_refl.exr')
