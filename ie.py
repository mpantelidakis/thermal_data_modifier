#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import argparse
import io
import os
import os.path
import subprocess
from PIL import Image

import numpy as np


class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.is_debug_number_of_images = 0
        self.flir_img_filename = ""
        self.rgb_image_np = None

    pass


    def process_image(self, flir_img_filename):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)
        :param flir_img_filename:
        :return:
        """
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename

        self.rgb_image_np = self.extract_embedded_image()


    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        :return:
        """
        return self.rgb_image_np


    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np


    def crop_center(self, img, cropx, cropy):
        """
        Crop the image to the given dimensions
        :return:
        """
        y, x, z = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]
    

    def save_images(self):
        """
        Save the extracted images
        :return:
        """
        rgb_np = self.get_rgb_np()

        cropped_img = self.crop_center(rgb_np, 504, 342)
       
        fn_prefix, _ = os.path.splitext(self.flir_img_filename)

        cropped_img_filename = os.path.join('Visual_Spectrum_images/' + fn_prefix.split('/')[1] + ".jpg")
        cropped_img_visual = Image.fromarray(cropped_img)
        
        if self.is_debug:
            print("DEBUG Saving cropped RGB image to:{}".format(cropped_img_filename))

        cropped_img_visual.save(cropped_img_filename)


class SmartFormatter(argparse.HelpFormatter):


    def _split_lines(self, text, width):

        if text.startswith('R|'):

            return text[2:].splitlines()  

        # this is the RawTextHelpFormatter._split_lines

        return argparse.HelpFormatter._split_lines(self, text, width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data', formatter_class=SmartFormatter)
    parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
                        default='exiftool')
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False,
                        action='store_true')
    args = parser.parse_args()

    fie = FlirImageExtractor(exiftool_path=args.exiftool, is_debug=args.debug)

    if not os.path.isdir('./images'):
        raise Exception('Folder with name "images" does not exist.')

    output_path = 'Visual_Spectrum_images'

    try:
        os.mkdir(output_path)
    except OSError:
        if args.debug:
            print("DEBUG Creation of the directory %s failed" % output_path)
    else:
        if args.debug:
            print("DEBUG Successfully created the directory %s " % output_path)

    image_path_list = glob.glob("images/*.jpg")
    
    for image_path in image_path_list:
        fie.process_image(image_path)
        fie.save_images()
        if args.debug:
            print ("-------------------------------------------------------")
    
    print("Total number of images: ",len(image_path_list))