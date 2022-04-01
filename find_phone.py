'''
Author: Kai Chuen Tan
Date  : 28th March 2022
'''

# Import all necessary Python packages
#-------------------------------------
import cv2 as cv
from phone_detection.phone_detector import Phone_Detector
import numpy as np
import os, sys
import pandas as pd
from utils import pix2norm_coords, sort_df

# Load the trained parameters
#----------------------------
# Load parameters
color_classes = np.load('color_classes.npy').astype('int')
trained_weights = np.load('trained_weights.npy')

# Convert back to their original data type instead of numpy.ndarray
color_classes = color_classes.tolist()
trained_weights_list = []
for row in range(0, trained_weights.shape[0]):
    trained_weights_list.append(trained_weights[row, :])

# Get current directory
curr_dir = os.getcwd()

# Get the path input
path_input  = sys.argv[1]

# Extract the second last substring after the second last '/' to get the folder name, i.e., "find_phone_test_images"
folder_name = path_input.split('/')[-2]

# Extract the last substring after the last '/' to get the image name
fname = path_input.split('/')[-1]

# Get image number
img_num = fname[:-4]

# Define the entire file directory
file_dir = curr_dir + '/' + folder_name + '/' + fname

# Read image
img = cv.imread(file_dir)

# Convert the image to RGB
img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Define labels folder name
label_folder = 'find_phone'

# Deine lable text file name
txt_file_name = 'labels.txt'

# Define the text file directory
txt_file_dir = curr_dir + '/' + label_folder + '/' + txt_file_name

# Convert the text file into a dataframe
labels_df = pd.read_csv(txt_file_dir, sep = ' ', header = None, names = ["Image Name", "Normalized X-Coordinate", "Normalized Y-Coordinate"])

# Sort the dataframe in an ascending order based on the image names
sorted_labels_df = sort_df(labels_df)

# Get true normalized pixel coordinates
true_norm_coord = sorted_labels_df.loc[int(img_num), "Normalized X-Coordinate": "Normalized Y-Coordinate"].to_numpy()

# Multiclass Logistic Regression Training Model Process (Prediction)
#-------------------------------------------------------------------
# Create the phone detector object
my_detector = Phone_Detector()

# Segment the image that is black phone screen.
mask_img = my_detector.segment_image(img, color_classes, trained_weights_list)

# Detect phone.
estm_boxes = my_detector.get_bounding_boxes(mask_img)
#print(estm_boxes)

# Initialize accuracy score
accuracy_score = 0

# Print the normalized coordinates and accuracy score.
for idx, box in enumerate(estm_boxes):

    # If not empty 
    if len(box) != 0:
        
        # Top-left corner coordinate of the box (pixel coordinate)
        pix_x_top_left = box[0]
        pix_y_top_left = box[1]
        # Bottom-right corner coordinate of the box (pixel coordinate)
        pix_x_bottom_right = box[2]
        pix_y_bottom_right = box[3]

        # Estimated phone location in pixel coordinate
        est_pix_x =  pix_x_top_left + (pix_x_bottom_right - pix_x_top_left)/2
        est_pix_y =  pix_y_top_left + (pix_y_bottom_right - pix_y_top_left)/2

        # Convert to normalized pixel coordinate
        est_norm_x, est_norm_y = pix2norm_coords(int(round(est_pix_x)), int(round(est_pix_y)), img_RGB)

        # Store the estimated normalized pixel coordinate in an array
        est_norm_coord = np.array([est_norm_x, est_norm_y])

        # Print Estimated Phone location
        print('\nEstimated Phone Normalized Pixel Coordinate: (' + str(round(est_norm_x, 4)) + ', ' + str(round(est_norm_y, 4)) + ')\n')

        # Calculate distance error.
        dist_error = np.linalg.norm(true_norm_coord - est_norm_coord)

        # If the dist_error is less than 0.05
        if dist_error <= 0.05:
            
            # Consider pass
            accuracy_score += 100
            
        # If not
        else:
            
            # Consider fail
            accuracy_score += 0

if len(estm_boxes) == 0:
    print("Accuracy Score: " + str(int(0)) + " %\n")
else:
    print("Accuracy Score: " + str(int(accuracy_score/len(estm_boxes))) + " %\n")

# Display rgb image with bounding box and display segmented image
my_detector.draw_bounding_boxes(mask_img, img_RGB, estm_boxes, img_num)




