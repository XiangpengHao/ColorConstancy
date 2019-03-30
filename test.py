from benchmark import BaseBench, metric_angle, metric_chrom_distance
from algorithms.shades_of_gray import AverageRGB, MaxRGB, PNorm
from algorithms.mixed_illuminant_kmeans import MixedKmeans
from algorithms.single_approx import BestSingleApprox
from images import SpectralImage, RGBImage

from joblib import Parallel, delayed


test_dir = 'fixtures/test'
output_dir = 'dist'
groundtruth_dir = 'fixtures/ground_truth'

algs: [BaseBench] = [BestSingleApprox]


def parallel_process(algorithm):
    alg_ins = algorithm(
        test_dir, f'{output_dir}/{algorithm.NAME}', groundtruth_dir)
    alg_ins.run()
    while alg_ins.has_next():
        alg_ins.get_next_refl()
        print(f'working on {alg_ins.curr_img_name}')
        angular_err = alg_ins.get_error(metric_angle)
        distance_err = alg_ins.get_error(metric_chrom_distance)
        alg_ins.draw_heatmap(angular_err, f'{alg_ins.curr_img_name}_ang')
        alg_ins.draw_heatmap(distance_err, f'{alg_ins.curr_img_name}_dis')


# Parallel(n_jobs=4)(delayed(parallel_process)(algorithm) for algorithm in algs)

parallel_process(BestSingleApprox)