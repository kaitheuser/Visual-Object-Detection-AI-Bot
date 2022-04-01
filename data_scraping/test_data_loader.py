import numpy as np
import os

### Load Normalized Training Dataset
# Data directory
folder  = 'data_scraping/data/'
save_dir = 'data_scraping/compiled_color_data/'
_, _, color_data_files = next(os.walk(folder))
color_data_files.remove('.DS_Store')
color_data_files = sorted(color_data_files,key=lambda x: int(os.path.splitext(x)[0]))

# Load the first file first to initialize an array
with open(folder + color_data_files[0]) as file_name:

    # Initialize color data
    color_data = np.loadtxt(file_name, delimiter=",")

# Load the rest of the data
for ID, filename in enumerate(color_data_files):

    # Skip the first file since it is loaded
    if ID == 0:

        continue
			
	# Training images (Number of Pixels x Number of Features)	
    with open(folder + filename) as file_name:
        
        new_data = np.loadtxt(file_name, delimiter=",")

    color_data = np.vstack((color_data, new_data))

# Save as csv file to "data_scraping/compiled_color_data/"
np.savetxt(save_dir + 'magic_color_data.csv', color_data, delimiter=",")