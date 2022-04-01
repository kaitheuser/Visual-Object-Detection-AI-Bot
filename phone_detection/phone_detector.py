'''
Author: Kai Chuen Tan
Date  : 28th March 2022
'''

# Import all necessary Python packages
#-------------------------------------
import copy
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import label, regionprops


# Create a phone detector class
#------------------------------
class Phone_Detector():

    def __init__(self, lr = 0.1, iters = 5000, tol = 1e-3):

        '''
        Initialize phone detector with the multiclass logistic regression model parameters

        Color Classes:
        --------------
        Black Screen        : 1
        White Tile          : 2
        Cement Grey Floor   : 3
        Tan Wooden Floor    : 4
        Grey Carpet         : 5
        Shiny Grey Floor    : 6
        Grey Tile           : 7
        Blue Tape           : 8

        Inputs with its Data Type:  1.) lr (float)       - Learning Rate
                                    2.) iters (int)      - Maximum Number of Iterations
                                    3.) tol (float)      - Error Tolerance

        Outputs with its Data Type: N/A
        '''

        ## Check for valid parameter inputs
        # Check parameters' data type
        assert isinstance(lr, (float, int))
        assert isinstance(iters, int)
        assert isinstance(tol, (float, int))
        # Check parameters' range
        assert 0 < lr <= 1                            
        assert 1000 <= iters <= 10000           
        assert tol <= 1
        assert iters * tol <= 10

        ## Initialize Multiclass Logistic Regression Model Parameters
        self.lr = lr                            # Learning Rate
        self.iters = iters                      # Maximum Number of Iterations
        self.tol = tol                          # Error Tolerance
        self.bias = 1                           # Bias Term
        self.weights = None                     # Weights
        self.classes = None                     # Classes of Interest


    def train(self, X_train, y_train):
        '''
        Apply Multi-class Logistic Regression Training Model to determine the trained weight parameters.

        Inputs with its Data Type:  1.) X_train (np.ndarray)    - 2D array features to train
                                    2.) y_train (np.ndarray)    - 1D array/vector label for training

        Outputs with its Data Type: 1.) self.classes (list)     - Classes of interest
                                    2.) self.weights (list)     - Trained weights
        '''
        ## Check for valid training dataset inputs
        # Check training dataset data type
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        # Check the training dataset number of dimensions
        assert len(X_train.shape) == 2
        assert len(y_train.shape) == 1

        # Initialize the X_train features and y_train labels
        self.X_train = np.insert(X_train, 0, self.bias, axis = 1)    # Add bias term to the X_train
        self.y_train = y_train

        # Initialize the weights as an empty list
        self.weights = []

        # Determine the number of samples and number of features
        num_Samples, num_Features = self.X_train.shape

        # Define the different color classes based on the training labels, y_train
        self.classes = np.unique(self.y_train)

        # Apply One VS All Binary Classification
        for color in self.classes:

            # Get the binary label
            y_binary = np.where(self.y_train == color, 1, 0)

            # Initialize the weight
            weight = np.zeros(num_Features)

            for idx in range(0, self.iters):

                # Calculate the probability using sigmoid function (p.m.f.)
                y_predicted = 1 / (1 + np.exp(-(self.X_train @ weight)))

                # Calculate the gradient change using gradient descent function
                delta_func = 1 / num_Samples * (np.dot((y_binary - y_predicted), self.X_train))

                # Store previous weight
                prev_weight = copy.deepcopy(weight)

                # Update weight
                weight += self.lr * delta_func

                # If less than the error tolerance, break the loop to prevent overtraining/overfitting (early stopping)
                if np.linalg.norm(prev_weight - weight) < self.tol:
                    break

            # Add the trained weight to the weights list
            self.weights.append(weight)
        
        return self.classes, self.weights


    def predict(self, X_test, classes = None, weights = None):
        '''
        Predict the labels of the test dataset with the trained weight parameters

        Inputs with its Data Type:  1.) X_test (np.ndarray)    - 2D array test features set
                                    2.) classes (list)         - Classes of interest
                                    3.) weights (list)         - Trained weights

        Outputs with its Data Type: y_predicted                - 1D array predicted label
        '''

        ## Check for valid test dataset input
        # Check test dataset data type
        assert isinstance(X_test, np.ndarray)
        # Check the test dataset number of dimensions
        assert len(X_test.shape) == 2
        if (classes is not None) and (weights is not None):
            ## Check for valid test dataset input
            # Check test dataset data type
            assert isinstance(classes, list)
            for num in classes:
                assert isinstance(num, int)
                assert 1 <= num <= 8            # Check for valid range
            assert isinstance(weights, list)
            for w_params in weights:
                assert isinstance(w_params, np.ndarray)
            # Assign them
            self.classes = classes
            self.weights = weights

        # Initialize the X_test features
        self.X_test = np.insert(X_test, 0, self.bias, axis = 1)   # Add bias term to the X_test

        # Predicted label (index)
        y = [np.argmax([1 / (1 + np.exp(-(x_test @ weight))) for weight in self.weights]) for x_test in self.X_test]

        # Predict the true label
        y_predicted = np.rint(np.array([self.classes[color] for color in y]))

        return y_predicted


    def segment_image(self, img, classes = None, weights = None):

        '''
			Obtain a segmented image from the Multiclass Logistic Regression predicted labels.
			
			Inputs with Data Type: 1.) img (np.ndarray)         - 3D array original image
                                   2.) classes (list)           - Classes of interest
                                   3.) weights (list)           - Trained weights

			Outputs with Data Type: 1.) mask_img (np.ndarray)   - a binary image with 1 if the pixel in the original image is black phone-screen and 0 otherwise
		'''

        ## Check for valid RGB image input
        # Check RGB image data type
        assert isinstance(img, np.ndarray)
        # Check the RGB image number of dimensions
        assert len(img.shape) == 3
		
		# Get the image pixel information from different color spaces
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_HSV = cv.cvtColor(img_RGB, cv.COLOR_RGB2HSV)
        img_LAB = cv.cvtColor(img_RGB, cv.COLOR_RGB2LAB)
        img_YCrCb = cv.cvtColor(img_RGB, cv.COLOR_RGB2YCR_CB)

		# Normalized the pixels
        img_RGB_norm = img_RGB.astype(np.float64)/255
        img_HSV_norm = img_HSV.astype(np.float64)/255
        img_LAB_norm = img_LAB.astype(np.float64)/255
        img_YCrCb_norm = img_YCrCb.astype(np.float64)/255

		# Get the image dimensions
        img_height, img_width, _ = img.shape

		# Initialize a mask
        mask_img = np.zeros((img_height,img_width), np.uint8) # Black Pixel = 0, White Pixel = 1

		# Reshape the image from H x W x 3 to (H X W) X 3
        img_RGB_norm = img_RGB_norm.reshape(img_RGB_norm.shape[0]*img_RGB_norm.shape[1],img_RGB_norm.shape[2])
        img_HSV_norm = img_HSV_norm.reshape(img_HSV_norm.shape[0]*img_HSV_norm.shape[1],img_HSV_norm.shape[2])
        img_LAB_norm = img_LAB_norm.reshape(img_LAB_norm.shape[0]*img_LAB_norm.shape[1],img_LAB_norm.shape[2])
        img_YCrCb_norm = img_YCrCb_norm.reshape(img_YCrCb_norm.shape[0]*img_YCrCb_norm.shape[1],img_YCrCb_norm.shape[2])

		# Compile the dataset as X_test (N x 12 features)
        X_test = np.concatenate((img_RGB_norm, img_HSV_norm, img_LAB_norm, img_YCrCb_norm), axis = 1)

		# Predict each pixel
        y_predicted = self.predict(X_test, classes, weights) # Number of Pixels x 1
        y_predicted_2D = y_predicted.reshape(img_height, img_width) # Height x Width

		# Unmask the pixel that is a phone black screen
        for height in range(img_height):
            for width in range(img_width):
                if y_predicted_2D[height, width] == 1:
                    mask_img[height, width] = 1 # Phone Black Screen is a white pixel
        
        return mask_img


    def get_bounding_boxes(self, mask_img):
        
        '''
			Get the bounding box(es) of the phone.
			
			Inputs with Data Type: mask_img (np.ndarray)    - mask image

			Outputs with Data Type: boxes (list)            - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				                                              where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''

        ## Check for valid mask image input
        # Check mask image data type
        assert isinstance(mask_img, np.ndarray)
        # Check the mask image number of dimensions
        assert len(mask_img.shape) == 2
		
		# Initialize Kernels for Erosion and Dilation process
        kernel_errosion_shape = 4 
        kernel_dilation_shape = 10
        kernel_errosion = np.ones((kernel_errosion_shape, kernel_errosion_shape), np.uint8)
        kernel_dilation = np.ones((kernel_dilation_shape, kernel_dilation_shape), np.uint8)

		# Erode to filter out noise
        mask_img = cv.erode(mask_img, kernel_errosion, iterations = 2)
		# Dilate to regain the size of phone screen without noise
        mask_img = cv.dilate(mask_img, kernel_dilation, iterations = 2)

		# Labeled array, where all connected regions are assigned the same integer value.
        label_img = label(mask_img)

		# Return list of RegionProperties from the label_img
        regions = regionprops(label_img)
		
		# Initialize the box list
        boxes = []

		# Regions props
        for props in regions:

            #print(props.area)
            #print(str(mask_img.shape[0] * mask_img.shape[1]))

			# Make sure the the size of the phone screen is appropriate
            if 0.01 * mask_img.shape[0] * mask_img.shape[1] > props.area > 0.005 * mask_img.shape[0] * mask_img.shape[1]:

                #print('Potential Phone Detected')

				# Get the bounding box top left and bottom right coordinates
                minr, minc, maxr, maxc = props.bbox

				# Calculate the height and width of the bounding box
                bb_height, bb_width = maxr - minr, maxc - minc

				# Check if the hight-to-width ratio of the phone screen is valid.
                if bb_width * 0.4 <= bb_height <= bb_width * 2.5:

                    # Phone Detected
                    #print('Phone Detected')

					# Add to boxes list
                    boxes.append([minc, minr, maxc, maxr])
		
        return boxes

	
    def draw_bounding_boxes(self, mask_img, rgb_img, boxes, img_num):
        
        '''
        Draw bounding boxes and display the image.
			
		Inputs with Data Type: 1.) mask_img (np.ndarray)    - mask image
                               2.) rgb_img (np.ndarray)     - rgb image
                               3.) boxes (list)             - bounding boxes coordinates (top left and bottom right)
                               4.) img_num (string)         - image number
			
		Outputs with Data Type: None
		'''
        ## Check for valid inputs
        # Check inputs data type
        assert isinstance(mask_img, np.ndarray)
        assert isinstance(rgb_img, np.ndarray)
        assert isinstance(boxes, list)
        assert isinstance(img_num, str)
        # If the boxes list is not empty
        if len(boxes) != 0:
            # Continue checking data types
            for box in boxes:
                assert isinstance(box, list)
                for coord in box:
                    assert isinstance(coord, (int, float))
        # Check images number of dimensions
        assert len(mask_img.shape) == 2
        assert len(rgb_img.shape) == 3

		# Get the image shape
        img_height, img_width, _ = rgb_img.shape

		# Initialize a mask image (3 Dimensions)
        mask_img_3D = np.zeros((img_height,img_width, 3)) # Black Pixel = 0, White Pixel = 1

		# Unmask the pixel that is a phone black screen in black and white
        for height in range(img_height):
            for width in range(img_width):
                if mask_img[height, width] == 1:
                    mask_img_3D[height, width,:] = 1
            
		# Plot subplots
        fig, ax = plt.subplots(1, 2, figsize = (8, 8))
		
		# Plot mask image without labels
        ax[0].imshow(mask_img_3D)
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        ax[0].set_xticks([])
        ax[0].set_yticks([])

		# Plot RGB image with bounding boxes
        ax[1].imshow(rgb_img)
        for box in boxes:
            bx = (box[0], box[2], box[2], box[0], box[0])
            by = (box[1], box[1], box[3], box[3], box[1])
            ax[1].plot(bx, by, '-r', linewidth=2.5)
        ax[1].set_yticklabels([])		
        ax[1].set_xticklabels([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

		# Show plot in tight layout
        plt.tight_layout()
        plt.show()

        # Define save image name
        save_fname = img_num + '_result.png'
        # Save figure.
        fig.savefig(save_fname)
