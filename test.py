from benchmark import BaseBench, metric_angle, metric_chrom_distance
from algorithms.shades_of_gray import AverageRGB, MaxRGB, PNorm
from algorithms.mixed_illuminant_kmeans import MixedKmeans
from images import SpectralImage, RGBImage


test_dir = 'fixtures/test'
output_dir = 'dist'
groundtruth_dir = 'fixtures/ground_truth'

algs: [BaseBench] = [AverageRGB, MaxRGB]

for algorithm in algs:
    alg_ins = algorithm(test_dir, output_dir, groundtruth_dir)
    alg_ins.run()
    while(alg_ins.has_next()):
        refl = alg_ins.get_next_refl()
        angular_err = alg_ins.get_error(metric_angle)
        distance_err = alg_ins.get_error(metric_chrom_distance)
        alg_ins.draw_heatmap(angular_err, f'{alg_ins.curr_img_name}_ang')
        alg_ins.draw_heatmap(distance_err, f'{alg_ins.curr_img_name}_dis')
