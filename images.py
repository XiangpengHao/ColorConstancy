import numpy as np
import OpenEXR
import Imath

float_pixel = Imath.PixelType(Imath.PixelType.FLOAT)
half_pixel = Imath.PixelType(Imath.PixelType.HALF)
channels: [int] = list(range(400, 700, 5))


class sRGBImage:
    def __init__(self, img_data: np.ndarray, file_path: str = None):
        self.file_path = file_path
        self.img_data = img_data
        self.img_shape = img_data.shape[:-1]

    @classmethod
    def NewFromArray(cls, img_data: np.ndarray):
        return sRGBImage(None, img_data)

    @classmethod
    def NewFromFile(cls, file_path: str):
        img_file = OpenEXR.InputFile(file_path)
        dw = img_file.header()['dataWindow']
        img_shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        img_data = np.zeros((*img_shape, 3), dtype=float)
        for i, c in enumerate("RGB"):
            buffed = img_file.channel(c, half_pixel)
            channel_data = np.frombuffer(buffed, dtype=np.float16)
            img_data[:, :, i] = channel_data.reshape(img_shape)
        return sRGBImage(img_data)
