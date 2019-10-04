import argparse
import sys
import os
import glob

import csv
import numpy as np
import cv2 as cv


# -*- coding: utf-8 -*-

class ThermalDataModifier:

    def __init__(self, is_debug=False, directory=''):
        self.is_debug = is_debug
        self.directory = directory
        self.unmodified_data_suffix = '_thermal_values.csv'

        self.rgb_image_np = None
        self.thermal_image_np = None

    def process_thermal_data(self):
        """
        Maps the extracted mask to the thermal data
        :return:
        """

        if self.is_debug:
            if os.path.exists(self.directory):
                print("DEBUG Working directory exists " + self.directory)
            else:
                print("DEBUG Error! Provided directory does not exist!")

        if os.path.exists(os.path.join(self.directory + 'mask.txt')):
            mask = np.loadtxt(os.path.join(self.directory + 'mask.txt'))
            new_mask = cv.resize(mask, dsize=(80, 60), interpolation=cv.INTER_CUBIC)
            np.savetxt(os.path.join(self.directory + 'mask_60x80.txt'),new_mask, fmt='%d')
            it = np.nditer(new_mask, flags=['multi_index'])
            if self.is_debug:
                print('DEBUG Mask successfully loaded!')
                print('DEBUG Mask was downscaled to 60x80!')
        else:
            print("Mask not found! Please add a mask.txt file to the following folder: "+ self.directory)
            exit()

        os.chdir(self.directory)
        unmodified_data_suffix = '_thermal_values.csv'
        loaded_csv_files = glob.glob('*{}'.format(unmodified_data_suffix))

        if loaded_csv_files.__len__() != 1:
            print('Multiple or no themal values csv files found inside the folder: ' + self.directory)
            print(loaded_csv_files)
            print('Please provide a .csv file to the folder or remove duplicate files')
            print('Exiting..')
            exit()

        if self.is_debug:
            print("DEBUG Using csv file: " + loaded_csv_files[0])

        #Create another csv containing class information
        with open(loaded_csv_files[0], 'r') as csvInput:

            reader = csv.reader(csvInput)

            all = []
            row = next(reader)
            row.append('Class')
            all.append(row)

            for row in reader:
                if not it.finished:
                    #check if the csv has been modified before
                    if (len(row)-1) >=6:
                        if self.is_debug:
                            print('')
                            print('DEBUG csv already contains class information and will not be modified')
                            print('output.csv file recreated')
                            print('')
                        return
                    if it[0] != 0:
                        row.append('Leaf')
                    else:
                        row.append('Noise')
                    it.iternext()
                    all.append(row)

            with open(loaded_csv_files[0], 'w') as csvOutput:
                writer = csv.writer(csvOutput, lineterminator='\n')
                writer.writerows(all)

        if self.is_debug:
            print('')
            print('DEBUG csv modified')
            print('DEBUG The file now also contains information regarding which pixels correspond to leaves')
            print('')

        with open(loaded_csv_files[0], 'r') as csvInput:
            reader = csv.reader(csvInput)
            next(reader)

            total_counter = 0
            leaf_counter = 0
            noise_counter = 0

            total_image_temp = 0
            total_leaf_temp = 0
            total_noise_temp = 0

            lowest_leaf_temp = 9999
            highest_leaf_temp = -9999

            lowest_noise_temp = 9999
            highest_noise_temp = -9999

            for row in reader:
                temp_value = float(row[2])
                total_counter += 1
                total_image_temp += temp_value
                if row[6] == 'Leaf':
                    leaf_counter += 1
                    total_leaf_temp += temp_value
                    if temp_value > highest_leaf_temp:
                        highest_leaf_temp = temp_value
                    if temp_value < lowest_leaf_temp:
                        lowest_leaf_temp = temp_value
                else:
                    noise_counter += 1
                    total_noise_temp += temp_value
                    if temp_value > highest_noise_temp:
                        highest_noise_temp = temp_value
                    if temp_value < lowest_noise_temp:
                        lowest_noise_temp = temp_value

            average_leaf_temp = total_leaf_temp/leaf_counter
            average_image_temp = total_image_temp/total_counter
            average_noise_temp = total_noise_temp/noise_counter
            diff = abs(average_leaf_temp - average_noise_temp)

            if self.is_debug:
                print('')
                print('DEBUG Temperature values that correspond to leaves:', leaf_counter)
                print('DEBUG Filtered out temperature values that do not correspond to leaves:', noise_counter)
                print('Total number of temperature values:', total_counter)
                print('')
                print('DEBUG Metrics successfully exported into output.csv')

            metric_labels = ['Temp avg', 'Leaf Temp avg', 'Noise Temp avg', 'avg diff',
                             'Leaf Temp peak', 'Leaf Temp Low', 'Noise Temp Peak', 'Noise Temp Low']

            metrics = [average_image_temp, average_leaf_temp, average_noise_temp, diff, highest_leaf_temp,
                       lowest_leaf_temp, highest_noise_temp, lowest_noise_temp]

            with open('output.csv', 'w') as csvOutput:
                writer = csv.writer(csvOutput, lineterminator='\n')

                all = []
                all.append(metric_labels)
                all.append(metrics)

                writer.writerows(all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modifies the thermal data and generates metrics')
    parser.add_argument('-dir', '--directory', type=str, help='Path to directory. Ex. images/test2/', required=False)
    parser.add_argument('-act', '--actions', help='Performs the action for all images inside folders with .csv and mask.txt files',required=False,  action='store_true')
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False, action='store_true')
    parsed_args = parser.parse_args()
    
    if parsed_args.actions:
        if parsed_args.debug:
            print("DEBUG All actions will be performed for the following folders:")
        folder_path_list = glob.glob("images/camera_*/*/")

        for folder_path in folder_path_list:

            if parsed_args.debug:
                print (folder_path)
            tdm = ThermalDataModifier(is_debug=parsed_args.debug, directory=folder_path)
            tdm.process_thermal_data()
    else:
        tdm = ThermalDataModifier(is_debug=parsed_args.debug, directory=parsed_args.directory)
        tdm.process_thermal_data()
    
    
