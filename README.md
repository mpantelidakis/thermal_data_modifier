# Flir Image Extractor

FLIR® thermal cameras like the FLIR ONE® include both a thermal and a visual light camera.
The latter is used to enhance the thermal image using an edge detector.

The resulting image is saved as a jpg image but both the original visual image and the raw thermal sensor data are embedded in the jpg metadata.

This small Python tool/library allows to extract the original photo and thermal sensor values converted to temperatures.

## Requirements

This tool relies on `exiftool`. It should be available in most Linux distributions (e.g. as `perl-image-exiftool` in Arch Linux or `libimage-exiftool-perl` in Debian and Ubuntu).

It also needs the Python packages *numpy* and *matplotlib* (the latter only if used interactively).

```bash
# sudo apt update
# sudo apt install exiftool
# sudo pip install numpy matplotlib xlrd
```

## Usage

This module can be used by calling it as a script:

```bash
python flir_image_extractor.py -act
python flir_image_extractor.py -p -i "images/2019-08-28/Camera_1/img_20190828_121055_010.jpg" -s -csv
```

```bash
usage: flir_image_extractor.py [-h] [-act] [-i INPUT] [-p] [-exif EXIFTOOL]
                               [-csv] [-s] [-d]

Extract and visualize Flir Image data

arguments:
  -h, --help            show this help message and exit
  -act, --actions       Perform all available actions apart from plot for all images.
			Includes the generation of 4 images anda csv file.
			1. Original thermal image (60x80)
			2. Original RGB image (640x480)
			3. Downscaled RGB image (60x80)
			4. Cropped RGB image (494x335)
			5. Thermal data csv file of temperatures and RGB values.
  -i INPUT, --input INPUT
                        Input image. Ex. img.jpg
  -p, --plot            Generate a plot using matplotlib
  -exif EXIFTOOL, --exiftool EXIFTOOL
                        Custom path to exiftool
  -csv EXTRACTCSV, --extractcsv EXTRACTCSV
                        Export the thermal data per pixel encoded as csv file
  -s, --scale		Generate a downscaled rgb image to match the thermal image's dimensions
  -d, --debug           Set the debug flag
```

If the user wants to bypass some of the attached metadata, a weather_data.xlsx file has to be added to the images folder.

## Supported/Tested cameras:

- AX8 (thermal + RGB)

## Credits

Raw value to temperature conversion is ported from this R package: https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
Original Python code from: https://github.com/Nervengift/read_thermal.py

# Thermal Data Modifier

This module combines the mask.txt and *_thermal_values.csv files to further enrich the latter.
To be specific, it maps the content of the mask.txt file to the corresponding pixels of the csv file.
The result is 1 extra column indicating if the given pixel is 'Leaf' or 'Noise'
It also provides an output.csv file with useful metrics regarding the given image

To generate the mask.txt file, open the cropped image in the pynovisao software, run a segmentor and click on the areas of interest.

This module can be used by calling it as a script:

```bash
python thermal_data_modifier.py -act
python thermal_data_modifier.py -dir "images/2019-08-28/Camera_1/img_20190828_121055_010/"
```
Note that the -act command is meant to be used on its own, after the user has generated masks for all the image folders.

```bash
usage: flir_image_extractor.py [-h] [-act] [-dir DIRECTORY] [-d]

arguments:
  -h, --help            show this help message and exit
  -dir DIRECTORY, --directory DIRECTORY
			Path to directory. Ex: images/2019-07-01/Camera_1/img_20190701_121055_011/
  -act, --actions       Performs the action for all images inside folders where both
			a mask.txt and a *_thermal_values.csv exist
  -d, --debug           Set the debug flag
```
