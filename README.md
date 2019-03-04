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


#### Evalute shades of grey
```python
from benchmark import SpectralImage
from algorithms.shades_of_gray import ShadesOfGray
img = SpectralImage.NewFromFile('vp_0_combine.exr')
b = ShadesOfGray(a, None)
reflectance = b.get_test_reflectance()
reflectance.dump_to_sRGB_image('test_b.exr')
```
