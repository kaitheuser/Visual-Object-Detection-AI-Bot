'''
Author: Kai Chuen Tan
Date  : 28th March 2022
'''

import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
from phone_detection.roipoly import RoiPoly

if __name__ == '__main__':

    # Get current directory
    curr_dir = os.getcwd()

    # Extract the last substring after the last '/' to get the folder name, i.e., "find_phone"
    folder_name = 'find_phone'

    # Define the entire folder directory
    folder_dir = curr_dir + '/' + folder_name

    # Define save directory
    save_dir = 'data_scraping/data/'

    # Get all training image names
    _, _, train_imgs = next(os.walk(folder_dir))

    # Remove elements that are not .jpg file
    train_imgs.remove('labels.txt')

    # Sort the list in an ascending order
    train_imgs = sorted(train_imgs,key=lambda x: int(os.path.splitext(x)[0]))
    
    
    """
        Customizable variable to decide where to start scrapping color data
    """
    start_ID = 12

    for ID, filename in enumerate(train_imgs):

        # Initialize and define color data as an empty list
        color_data = []

        # Completion Status
        completion_status = False

        # Skip to the starting image
        if ID < start_ID - 1:
            continue

        # Read the training image
        img = cv.imread(os.path.join(folder_dir,filename))
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_HSV = cv.cvtColor(img_RGB, cv.COLOR_RGB2HSV)
        img_LAB = cv.cvtColor(img_RGB, cv.COLOR_RGB2LAB)
        img_YCrCb = cv.cvtColor(img_RGB, cv.COLOR_RGB2YCR_CB)

        # While completion status is False
        while completion_status == False:

            # Display the image and use roipoly for labeling
            fig, ax = plt.subplots()
            ax.imshow(img_RGB)
            my_roi = RoiPoly(fig = fig, ax = ax, color = 'r')

            # Get the image mask
            mask = my_roi.get_mask(img_RGB)

            # Ask user whether it is a recycling-bin blue sample
            while True:
                color_type = input("Choose a color [1 - 8]:\n"\
                                        "Black Screen        : 1\n"\
                                        "White Tile          : 2\n"\
                                        "Cement Grey Floor   : 3\n"\
                                        "Tan Wooden Floor    : 4\n"\
                                        "Grey Carpet         : 5\n"\
                                        "Shiny Grey Floor    : 6\n"\
                                        "Grey Tile           : 7\n"\
                                        "Blue Tape           : 8\n")
                # Convert string to integer                        
                color_type = int(color_type)
                # Make sure the input is in between the color selection range
                if 1 <= color_type <= 8:
                    break
                # Ask user to re-select appropriate color number
                print("\nError 404: Color not found. Please try again.\n")

            # Extract color data from the mask
            for height in range(mask.shape[0]):
                for width in range(mask.shape[1]):
                    if mask[height, width]:
                        color_data.append([img_RGB[height, width, 0], img_RGB[height, width, 1], img_RGB[height, width, 2],
                                           img_HSV[height, width, 0], img_HSV[height, width, 1], img_HSV[height, width, 2],
                                           img_LAB[height, width, 0], img_LAB[height, width, 1], img_LAB[height, width, 2], 
                                           img_YCrCb[height, width, 0], img_YCrCb[height, width, 1], img_YCrCb[height, width, 2], 
                                           color_type])

            # Print number of pixels selected
            print('Number of Pixels Selected: ' + str(img[mask,:].shape[0]) + '\n\n')

            # Ask user's completion status
            while True:
                user_status = input("\nDone extracting data from this image?[Y/N]: ")
                if user_status not in ['Y','y','N','n']:
                    print("\nError. Please input valid response.\n")
                    continue
                if user_status in ['Y','y']:
                    completion_status = True
                    break
                if user_status in ['N','n']:
                    break
        
        # Convert list to array
        color_data = np.rint(np.array(color_data))

        # Save as csv file to "bin_detection/data/training'"
        np.savetxt(save_dir + filename.replace("jpg", "csv"), color_data, delimiter=",")