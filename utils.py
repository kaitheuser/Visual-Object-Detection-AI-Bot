'''
Author: Kai Chuen Tan
Date  : 28th March 2022
'''

# Import all necessary Python packages
#-------------------------------------
import cv2 as cv
import numpy as np
import os
import pandas as pd
import sys

def load_txt(fname):

    '''
    Load the labels text file and convert it into a data frame.
    
    Input with Data Type: fname (string)        - text file name

    Output with Data Type: df (pd.DataFrame)    - dataframe from the text file.
    '''

    # Check whether the file name input is a string
    assert isinstance(fname, str)

    # Get current directory
    curr_dir = os.getcwd()

    # Get the path input
    path_input  = sys.argv[1]

    # Extract the last substring after the last '/' to get the folder name, i.e., "find_phone"
    folder_name = path_input.split('/')[-1]

    # Define the entire file directory
    file_dir = curr_dir + '/' + folder_name + '/' + fname

    # Convert the text file into a dataframe
    df = pd.read_csv(file_dir, sep = ' ', header = None, names = ["Image Name", "Normalized X-Coordinate", "Normalized Y-Coordinate"])

    return df

def sort_df(labels_df):

    '''
    Sort the labels dataframe in an ascending order based on the image names.

    Inputs with Data Type: labels_df (pd.Datafrme)          - labels dataframe

    Outputs with Data Type: sorted_labels_df (pd.Datafrme)  - sorted labels dataframe
    '''
    ## Check for valid inputs
    # Check inputs data type
    assert isinstance(labels_df, pd.DataFrame)

     # Get the image names and store them in a list.
    img_names_list = list(labels_df['Image Name'])

    # Drop the file type from the img_names_list
    img_nums_list = [os.path.splitext(img_num)[0] for img_num in img_names_list]

    # Convert the strings to integers from the img_nums_list
    img_nums_list = list(map(int, img_nums_list))

    # Add to the img_nums_list to the labels_df
    labels_df['Image Number'] = img_nums_list

    # Set the Image Number column as the index of the labels dataframe
    unsorted_labels_df = labels_df.set_index('Image Number')

    # Sort the labels dataframe index.
    sorted_labels_df = unsorted_labels_df.sort_index()

    return sorted_labels_df

def norm2pix_coords(norm_x, norm_y, img_RGB):
    
    '''
    Convert normalized coordinates to pixel coordinates.

    Inputs with Data Type: 1.) norm_x (float)                  - normalized x-coordinate
                           2.) norm_y (float)                  - normalized y-coordinate
                           3.) img_RGB (np.ndarray)            - RGB image

    Outputs with Data Type: 1.) pix_x (int)                    - pixel x-coordinate
                            2.) pix_y (int)                    - pixel y-coordinate
    '''
    ## Check for valid inputs
    # Check inputs data type
    assert isinstance(norm_x, (float, int))
    assert isinstance(norm_y, (float, int))
    assert isinstance(img_RGB, np.ndarray)
    # Check the RGB image number of dimensions
    assert len(img_RGB.shape) == 3

    # Get the height and width of the image
    img_height, img_width, _ = img_RGB.shape

    # Convert to pixel coordinates
    pix_x = round(norm_x * (img_width - 1))
    pix_y = round(norm_y * (img_height - 1))

    return int(pix_x), int(pix_y)


def pix2norm_coords(pix_x, pix_y, img_RGB):
    
    '''
    Convert pixel coordinates to normalized coordinates.

    Inputs with Data Type: 1.) pix_x (int)                    - pixel x-coordinate
                           2.) pix_y (int)                    - pixel y-coordinate
                           3.) img_RGB (np.ndarray)           - RGB image

    Outputs with Data Type: 1.) norm_x (float)                - normalized x-coordinate
                            2.) norm_y (float)                - normalized y-coordinate
    '''
    # Check inputs data type
    assert isinstance(pix_x, int)
    assert isinstance(pix_y, int)
    assert isinstance(img_RGB, np.ndarray)
    # Check the RGB image number of dimensions
    assert len(img_RGB.shape) == 3

    # Get the height and width of the image
    img_height, img_width, _ = img_RGB.shape

    # Convert to normalized coordinates
    norm_x = pix_x / (img_width - 1)
    norm_y = pix_y / (img_height - 1)

    return norm_x, norm_y

def append_data(pix_x, pix_y, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, color_label):
    '''
    Append features and labels to X_train and y_train, respectively.

    Inputs with Data Type: 1.) pix_x (int)                          - pixel x-coordinate
                           2.) pix_y (int)                          - pixel y-coordinate
                           3.) img_RGB (np.ndarray)                 - RGB image
                           4.) img_HSV (np.ndarray)                 - HSV image
                           5.) img_LAB (np.ndarray)                 - LAB image
                           6.) img_YCrCb (np.ndarray)               - YCrCb image
                           7.) X_train (list)                       - Train features data
                           8.) y_train (list)                       - Train labels data
                           9.) color_label (int)                    - Color label

    Outputs with Data Type: 1.) X_train (np.ndarray)                - Updated train features data
                            2.) y_train (np.ndarray)                - Updated train labels data
    '''
    ## Check for valid inputs
    # Check inputs data type
    assert isinstance(pix_x, int)
    assert isinstance(pix_y, int)
    assert isinstance(img_RGB, np.ndarray)
    assert isinstance(img_HSV, np.ndarray)
    assert isinstance(img_LAB, np.ndarray)
    assert isinstance(img_YCrCb, np.ndarray)
    assert isinstance(X_train, list)
    assert isinstance(y_train, list)
    assert isinstance(color_label, int)
    # Check the images number of dimensions
    assert len(img_RGB.shape) == 3
    assert len(img_HSV.shape) == 3
    assert len(img_LAB.shape) == 3
    assert len(img_YCrCb.shape) == 3

    # Append X_train and y_train
    X_train.append([img_RGB[pix_y, pix_x, 0], img_RGB[pix_y, pix_x, 1], img_RGB[pix_y, pix_x, 2],
                    img_HSV[pix_y, pix_x, 0], img_HSV[pix_y, pix_x, 1], img_HSV[pix_y, pix_x, 2],
                    img_LAB[pix_y, pix_x, 0], img_LAB[pix_y, pix_x, 1], img_LAB[pix_y, pix_x, 2], 
                    img_YCrCb[pix_y, pix_x, 0], img_YCrCb[pix_y, pix_x, 1], img_YCrCb[pix_y, pix_x, 2]])
    y_train.append(color_label)

    return X_train, y_train


def extract_pixels(labels_df, bad_data_list = []):

    '''
    Extract all the black phone screen pixels from the labels dataframe.

    Inputs with Data Type: 1.) labels_df (pd.Datafrme)        - labels dataframe
                           2.) bad_data_list (list)           - list of bad training images 

    Outputs with Data Type: 1.) train_data (np.ndarray)       - Extracted train features and labels data [X, y]
    '''

    ## Check for valid inputs
    # Check inputs data type
    assert isinstance(labels_df, pd.DataFrame)
    assert isinstance(bad_data_list, list)
    if len(bad_data_list) != 0:
        for img_name in bad_data_list:
            assert isinstance(img_name, str)

    ## Sort the dataframe in an ascending order based on the image names
    sorted_labels_df = sort_df(labels_df)

    ## Extract the information from the sorted labels dataframe
    # Get the sorted image names and store them in a list.
    img_names_list = list(sorted_labels_df['Image Name'])

    # Get the normalized coordinates from the sorted_labels_df (2D-array, N x 2)
    normalized_coords = sorted_labels_df[['Normalized X-Coordinate','Normalized Y-Coordinate']].to_numpy()

    # Get current directory
    curr_dir = os.getcwd()

    # Get the path input
    path_input  = sys.argv[1]

    # Extract the last substring after the last '/' to get the folder name, i.e., "find_phone"
    folder_name = path_input.split('/')[-1]

    # Initialize X_train and y_train
    X_train = []
    y_train = []

    # Start extracting the black phone screen pixels from each image
    for idx, img_name in enumerate(img_names_list):

        # If it is a bad training image
        if img_name in bad_data_list:

            # Skip to the next training image
            continue

        # Get the entire image directory
        img_dir = curr_dir + '/' + folder_name + '/' + img_name

        # Read the RGB image
        img = cv.imread(img_dir)
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Get the same images but with different color spaces
        img_HSV = cv.cvtColor(img_RGB, cv.COLOR_RGB2HSV)
        img_LAB = cv.cvtColor(img_RGB, cv.COLOR_RGB2LAB)
        img_YCrCb = cv.cvtColor(img_RGB, cv.COLOR_RGB2YCR_CB)

        # Convert to pixel coordinates
        pix_x, pix_y = norm2pix_coords(normalized_coords[idx, 0], normalized_coords[idx, 1], img_RGB)

        # Append data including the 8 neighbouring pixels 
        X_train, y_train = append_data(pix_x, pix_y, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)             # Center
        X_train, y_train = append_data(pix_x - 1, pix_y - 1, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)     # Top-left
        X_train, y_train = append_data(pix_x, pix_y - 1, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)         # Center-top
        X_train, y_train = append_data(pix_x + 1, pix_y - 1, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)     # Top-right
        X_train, y_train = append_data(pix_x - 1, pix_y, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)         # Center-left
        X_train, y_train = append_data(pix_x + 1, pix_y, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)         # Center-right
        X_train, y_train = append_data(pix_x - 1, pix_y + 1, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)     # Bottom-left
        X_train, y_train = append_data(pix_x, pix_y + 1, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)         # Center-bottom
        X_train, y_train = append_data(pix_x + 1, pix_y + 1, img_RGB, img_HSV, img_LAB, img_YCrCb, X_train, y_train, 1)     # Bottom-right

    # Convert list to array
    X_train = np.rint(np.array(X_train))
    y_train = np.rint(np.array(y_train)).reshape(-1, 1)

    # Horizontally stack the data [X, y]
    train_data = np.hstack((X_train, y_train))

    return train_data

def get_train_data(fname, bad_data_list):

    '''
    Compile and obtain all training data.

    Inputs with Data Type: 1.) fname (string)               - label file name.
                           2.) bad_data_list (list)         - list of images to skip due to bad data.

    Outputs with Data Type: 1.) X_train (np.ndarray)        - Training Features Data
                            2.) y_train (np.ndarray)        - Training Labels Data
    '''
    ## Check for valid inputs
    # Check inputs data type
    assert isinstance(fname, str)
    assert isinstance(bad_data_list, list)
    if len(bad_data_list) != 0:
        for img_name in bad_data_list:
            assert isinstance(img_name, str)

    # Define the additional training pixels data folder path
    folder_path = 'data_scraping/compiled_color_data/'

    # Define the additional training pixels data file name
    file_name_csv = 'magic_color_data.csv'

    # Load the file first to initialize an array
    with open(folder_path + file_name_csv) as file_name:
        
        # Store color data
        other_train_data = np.loadtxt(file_name, delimiter=",")

    # Load the text file and store it as a dataframe
    labels_df = load_txt('labels.txt')
    
    # Extract black phone screen pixel training data.
    phone_train_data = extract_pixels(labels_df, bad_data_list)

    # Compiled the train data
    compiled_train_data = np.vstack((phone_train_data, other_train_data))

    # Print loading process is complete
    print("\nData Loading process is complete.\nTotal number of pixels loaded: " + str(compiled_train_data.shape[0]) + ".\n")

    # Normalize the pixel values
    compiled_train_data[:, :-1] = compiled_train_data[:, :-1].astype(np.float64)/255

    # Sort the color label in an ascending order
    idc = np.argsort(compiled_train_data[:, -1]) 
    sorted_compiled_train_data = compiled_train_data[idc]

    # Extract X_train features and y_train labels
    X_train = sorted_compiled_train_data[:, :-1]
    y_train = sorted_compiled_train_data[:, -1]

    return X_train, y_train