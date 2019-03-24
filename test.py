from benchmark import BaseBench, get_angle, get_chrom_distance
from algorithms.shades_of_gray import AverageRGB, MaxRGB, PNorm
from images import SpectralImage, RGBImage

test_file = 'test3.png'
groundtruth_file = 'test3_truth.exr'

algs: [BaseBench] = [AverageRGB, MaxRGB, PNorm]
test_img = RGBImage.NewFromFile(f'fixtures/{test_file}')

groundtruth_img = RGBImage.NewFromFile(f'fixtures/{groundtruth_file}')

for algorithm in algs:
    alg_ins = algorithm(test_img)
    refl = alg_ins.get_reflectance()
    refl.dump_file(f'dist/{test_file}_{algorithm.NAME}_refl.exr')
    angular_error = alg_ins.get_error(groundtruth_img, get_angle)
    distance_error = alg_ins.get_error(
        groundtruth_img, get_chrom_distance)
    alg_ins.draw_heatmap(angular_error, f'{test_file}_ang')
    alg_ins.draw_heatmap(distance_error, f'{test_file}_dis')
