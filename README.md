## Color Constancy Algorithms


#### Load a spectral image
```python
img1 = SpectralImage.NewFromFile('path_to_img.exr') # load the exr image file
img2 = SpectralImage.NewFromSpectrum(np_ndarray) 
print(img[405]) # print out the spectral intensity at 405nm
```

#### Convert a spectral image to a sRGB image
```python
img.dump_to_sRGB_image('output_rgb.exr')
```


#### Evaluate shades of grey
```python
from benchmark import SpectralImage, BaseBench
from algorithms.shades_of_gray import AverageRGB, MaxRGB


algs: [BaseBench] = [AverageRGB, MaxRGB]
img = SpectralImage.NewFromFile('test_fixtures/vp_0_combined.exr')

for algorithm in algs:
    refl = algorithm(img, None).get_test_reflectance()
    refl.dump_to_sRGB_image(f'dist/{algorithm.NAME}_refl.exr')
```
