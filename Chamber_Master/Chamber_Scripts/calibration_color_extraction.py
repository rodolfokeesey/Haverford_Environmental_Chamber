import cv2
import numpy as np
import pickle

#%% Overview:

# This script is used to extract the RGB values from the Xrite color chart 
# reference for color calibration.

# First the user must take a picture of the Xrite color chart under experimental
# conditions.
# Because of the magnification on the Haverford chamber set up, two images have
# to be taken to get all of the patches on the 6x4 color chart.

# The first image focuses on columns 1:3, and the second image on columns 4:6.

# This script will then ask for the name of the image containing color patches 
# in columns 1:3, and then the name of the second image containing the patches
# in columns 4:6. 

# The user will then be asked to select the area of the first
# image containing the color patches. Select the images in columns 1:3, 
# from left to right, top to bottom. Once the first 12 patches have been selected
# the second image will be displayed. The user will repeat the process for the
# patches in columns 4:6.

# Once complete, the average RGB pixel value of each patch will be saved as a .pkl
# file with the data of image acquisition. These RGB values will be used in the
# spline transformation.

# New calibration images should be taken whenever lighting conditions within 
# the chamber change. If the lighting conditions remain constant, this calibration
# does not need to be run.

# Please fill out the User Input Section before running.

#%% User Input

# Enter in the path to the chamber data folder
chamber_data_path = "C:/Users/Rod/Documents/Chamber_Test_Folder/"

# Enter in the name of the folder containing the two calibration images.
expt_run = "Color_Calibration_5_6_22" 

# Enter in the name the first calibration image. This image should have the
# dark brown patch framed in the top left corner. See Calibration_One_5_6_22.png
# as an example of how the image should look.
image_path_1 = "Calibration_One_5_6_22.png"

# Enter in the name the first calibration image. This image should have the
# black patch framed in the bottom right corner. See Calibration_Two_5_6_22.png
# as an example of how the image should look.
image_path_2 = "Calibration_Two_5_6_22.png"

# Enter the number of calibration patches. The default is 24 (4 rows, 6 columns).
num_patches = 24

#%% Manually cropping the locations of the calibration color patches

cropping = False
cycled = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
roi = []

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, roi, cycled
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            #cv2.imshow("Cropped", roi)
            cycled = True


# This section asks for the user to type in the information handled in the user
# input section. Uncomment if you prefer this method.
#image_path_1 = str(input('Enter the name of the first calibration image file. \n'))
#image_path_2 = str(input('Enter the name of the second calibration image file. \n'))
#num_patches = int(input("Enter the number of calibration patches. The default is 24 (4 rows, 6 columns). \n"))

cal_list = []

# Create empty Xrite array to store the color values for the first image.
Xrite_array = np.zeros((4,6,3))

#The Xrite is too large to be fully captured in one image. This gives the number of color patch columns in image 1 and image 2.
columns_per_image = 3

# Loop across the 24 color patches. Each iteration of the outer loop for patch,
# promts the user to select a bounding box for each patch.
for patch in range(1, num_patches + 1):
    
    # For the first 12 patches, image one is displayed to the user.
    if patch <= 12 :
        # opens image 1
        image = cv2.imread(chamber_data_path + expt_run + '/' + image_path_1)
        oriImage = image.copy()
        # Displays the image
        cv2.namedWindow("image")
        # Runs the mouse crop tracking
        cv2.setMouseCallback("image", mouse_crop)
        
        # Once user has selected a bounding box, the "cycled" variable is set to true, an
        while cycled == False:
            i = image.copy()
            if not cropping:
                cv2.imshow("image", image)
            elif cropping:
                cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("image", i)
                if cycled:
                    break
            cv2.waitKey(1)
        # close all open windows
        
        #Sums all the RGB Values in the cropped Color
        sum_rgb = np.sum(np.sum(roi[:,:,:], axis = 0), axis = 0)
        #Calculate the total number of pixels
        tot_pixels = np.shape(roi[:,:,0])[0] * np.shape(roi[:,:,0])[1]
        # Then divide the summed RGB values by the number of pixels for the average
        # rgb value
        avg_rgb = np.flip(sum_rgb / tot_pixels)
        
        Xrite_array[(patch-1)//columns_per_image, (patch-1) % columns_per_image, :] = avg_rgb
        cal_list.append(roi)
        cv2.destroyAllWindows()
        cycled = False
    
    else:
        image = cv2.imread(chamber_data_path + expt_run + '/' + image_path_2)
        oriImage = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        
        while cycled == False:
            i = image.copy()
            if not cropping:
                cv2.imshow("image", image)
            elif cropping:
                cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("image", i)
                if cycled:
                    break
            cv2.waitKey(1)
        # close all open windows
        
        #Sums all the RGB Values in the cropped Color
        sum_rgb = np.sum(np.sum(roi[:,:,:], axis = 0), axis = 0)
        #Calculate the total number of pixels
        tot_pixels = np.shape(roi[:,:,0])[0] * np.shape(roi[:,:,0])[1]
        # Then divide the summed RGB values by the number of pixels for average
        avg_rgb = np.flip(sum_rgb / tot_pixels)
        
        Xrite_array[((patch-1)//columns_per_image) - np.shape(Xrite_array)[0], ((patch - 1 ) % columns_per_image) + columns_per_image, :] = avg_rgb
        cal_list.append(roi)
        cv2.destroyAllWindows()
        cycled = False


# Saves the RGB value of the Xrite patches as a .pkl file, to be used in color calibrations.
pickle.dump(Xrite_array, open(chamber_data_path + expt_run + '/xrite_rgb.pkl', 'wb'))

