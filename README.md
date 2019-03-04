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
