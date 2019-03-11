## Color Constancy Algorithms


#### Load a spectral image
```python
img1 = SpectralImage.NewFromFile('path_to_img.exr') # load the exr image file
img2 = SpectralImage.NewFromSpectrum(np_ndarray) 
print(img[405]) # print out the spectral intensity at 405nm
```

#### Convert a spectral image to a sRGB image
```python
rgb_img = spectral_img.to_srgb()
rgb_img.dump_file('output_rgb.exr')
```


#### Evaluate shades of grey
```python
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

```
