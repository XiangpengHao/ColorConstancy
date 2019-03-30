from benchmark import BaseBench
from images import RGBImage
import numpy as np
from color_space import RGB, XYZ
from numba import jit


def get_best_single_adjustment(img: RGBImage, ground_truth: RGBImage):
    shape = img.img_shape
    xyz_img = img.get_xyz_image()
    xyz_truth = ground_truth.get_xyz_image()

    rv = []
    for i in range(3):
        A = xyz_img[:, :, i].flatten()
        B = xyz_truth[:, :, i].flatten()
        rv.append(np.dot(B, A)/np.dot(A, A))

    adjusted_img = np.zeros((*shape, 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if xyz_img[i, j, 1] == 0:
                adjusted_img[i, j, :] = np.array([0, 0, 0])
                continue
            raw_xyz = RGB(*img[i, j, :], True).to_xyz()
            adjusted_xyz = np.multiply(xyz_img[i, j, :], rv) * \
                (raw_xyz.y/xyz_img[i, j, 1])
            adjusted_img[i, j, :] = XYZ.static_to_rgb(adjusted_xyz)
    return RGBImage.NewFromArray(adjusted_img)


class BestSingleApprox(BaseBench):
    NAME = 'best_approx'

    def run(self):
        for img in self.img_list:
            cur_img = RGBImage.NewFromFile(f'{self.input_dir}/{img}')
            groundtruth = self.get_groundtruth(img)
            adjusted = get_best_single_adjustment(cur_img, groundtruth)
            adjusted.dump_file(f'{self.output_dir}/{img}')
            self._refl_cache[img] = adjusted
