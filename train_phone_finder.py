'''
Author: Kai Chuen Tan
Date  : 28th March 2022
'''

# Import all necessary Python packages
#-------------------------------------
from phone_detection.phone_detector import Phone_Detector
import numpy as np
from time import time
from utils import get_train_data

# Input parameters
#-----------------
# Define the labels text file name
fname = 'labels.txt'

# List the bad data training images
bad_data_list = ['9.jpg', '10.jpg', '15.jpg', '18.jpg', '23.jpg',
                     '29.jpg', '31.jpg', '41.jpg', '44.jpg', '46.jpg',
                     '50.jpg', '59.jpg', '60.jpg', '67.jpg', '68.jpg',
                     '73.jpg', '74.jpg', '75.jpg', '77.jpg', '78.jpg',
                     '79.jpg', '84.jpg', '88.jpg', '97.jpg', '99.jpg',
                     '100.jpg', '101.jpg', '102.jpg', '103.jpg', '105.jpg',
                     '107.jpg', '109.jpg', '117.jpg', '118.jpg', '119.jpg',
                     '123.jpg', '126.jpg', '129.jpg', '130.jpg', '132.jpg']

# Set Multiclass Logistic Regression Model Parameters
lr = 0.1                            # Learning Rate
iters = 5000                        # Maximum Number of Iterations
tol = 1e-3                          # Error Tolerance

# Pre-processing Training Data
#-----------------------------
# Get the training data
X_train, y_train = get_train_data(fname, bad_data_list)

# Multiclass Logistic Regression Training Model Process
#------------------------------------------------------
# Create the phone detector object
my_detector = Phone_Detector(lr, iters, tol)

# Train the data
start_timer = time()                                                                                # Start timer
color_classes, trained_weights = my_detector.train(X_train, y_train)                                # Start training
print("Training Process is Completed. Training Time is " + str(round(time() - start_timer, 4)) + ' s.')     # Print training status.

# Save the classes of interest and trained weights
np.save('color_classes', color_classes)
np.save('trained_weights', trained_weights)
print("Parameters Saved.\n")
