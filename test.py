from benchmark import BaseBench
from algorithms.shades_of_gray import AverageRGB, MaxRGB, PNorm
from images import SpectralImage, RGBImage

test_file = 'test0'

algs: [BaseBench] = [AverageRGB, MaxRGB, PNorm]
spectral_img = SpectralImage.NewFromFile(f'fixtures/{test_file}.exr')
rgb_img = spectral_img.to_rgb()

spectral_groundtruth = SpectralImage.NewFromFile(
    f'fixtures/{test_file}_truth.exr')
rgb_grundtruth = spectral_groundtruth.to_rgb()

for algorithm in algs:
    alg_ins = algorithm(rgb_img)
    refl = alg_ins.get_reflectance()
    refl.dump_file(f'dist/{test_file}_{algorithm.NAME}_refl.exr')
    angular_error = alg_ins.get_angular_error(rgb_grundtruth)
    alg_ins.draw_heatmap(angular_error, 'a')
