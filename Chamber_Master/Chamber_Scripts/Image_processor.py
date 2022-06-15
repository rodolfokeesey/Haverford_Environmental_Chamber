
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from Color_transformation import haverford_color_calibration

#%% Overview:

# This script is the second step in data processing.

# This script locates each sample well, isolates the bulk crystals from the 
# filter paper, extracts the RGB data, and color calibrates the RGB values using
# the Xrite color passport.

# Please fill in the user inputs to run.

#%% User Inputs

# The path for the chamber's data folder
chamber_data_path = "C:/Users/Rod/Documents/Chamber_Test_Folder/"
# Enter in the experiment run you'd like to sync data for.
expt_run = "RK_37" 
# Enter the name of the video for the run
video_path = "image_0.avi"
# Enter the name of the compiled image file
all_images = "all_frames"
# Enter the number of seconds between frames of image capture
image_sample_rate = 60
# Year of the run
Year = "22 " 
#well_of_interest = 1
# Number of wells in the sample holder
total_wells = 9
# The number of the first frame in which the sample are loaded. This is found by manually
# going into the compiled image file and finding the first image where the samples
# loaded.
first_frame = 1400 # Here, the samples first appear in frame 1400.
# The sample rate in minutes that you'd like to query RGB time series data.
output_sample_rate = 10
# The well number for the blank filter paper. This is used as a reference
# to make the mask. The default is 5, or the middle well of the 9 well plate.
blank_well = 5



#%% Image Processing Functions. These functions are used to locate the sample wells,
### put them in order,

# Gets the radius and center of all the wells with circle detection
def get_well(image):
    
    # Loads the image of the sample holder
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    # Converts to grayscale for circle detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    # Finds all the circles in the image that satisfy the given parameters.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=80, maxRadius=110)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)

    # This shows the detected circles overlayed on the image. Uncomment for 
    # troubleshooting with adjusting the Hough Circle parameters to detect your
    # wells
    #cv2.imshow("detected circles", image)
    #cv2.waitKey(0)
    return(circles)

def well_order(circles):
    # Order in terms of greatest Y value to find which circles are in which row
    sortbyrow = np.argsort(circles[0][:,1])
    # These circles are in row 1
    r1 = circles[0][sortbyrow[0:3]]
    # These are in row 2
    r2 = circles[0][sortbyrow[3:6]]
    # These are in row 3
    r3 = circles[0][sortbyrow[6:9]]
    
    #[0:2] are in row 1
    #[3:5] are in row 2
    #[6:8] are in row 3
    
    # We still don't know column order. Order in terms of least X value. Then
    # within each row we can determine column order
    r1sorted = r1[np.argsort(r1[:,0])]
    r2sorted = r2[np.argsort(r2[:,0])]
    r3sorted = r3[np.argsort(r3[:,0])]
    
    # Stacking r1-3sorted together, gives the final order
    ordered_wells = np.vstack((r1sorted,r2sorted,r3sorted))

    return(ordered_wells)



#%%  Image processing functions that are used to create the mask




def create_mask(t0_image_path, well_of_interest):
    image = cv2.imread(t0_image_path)
    wells = get_well(t0_image_path) # Extracts the wells from the image
    if wells is not None:
        ordered_wells = well_order(wells) #Gets the order of the wells
    else: 
        ordered_wells = []

    
    # Get the bounding box from circle center and radius from image at t0
    
    well_of_interest = well_of_interest - 1
    
    x1 = ordered_wells[well_of_interest][0] - (ordered_wells[well_of_interest][2] - 5)
    y1 = ordered_wells[well_of_interest][1] - (ordered_wells[well_of_interest][2] - 5)
    x2 = ordered_wells[well_of_interest][0] + (ordered_wells[well_of_interest][2] - 5)
    y2 = ordered_wells[well_of_interest][1] + (ordered_wells[well_of_interest][2] - 5)
    
    
    # With the bounding box, create the mask to isolate the well of interest
    image = Image.open(t0_image_path, 'r')
    height,width = image.size
    lum_img = Image.new('L', [height,width] , 0)
    
    #
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(x1,y1), (x2,y2)], 0, 360, 
                  fill = 255, outline = "white")
    lum_img_arr =np.array(lum_img)
    
    mask = np.dstack((lum_img_arr/255,lum_img_arr/255,lum_img_arr/255))
    
    return mask

def test_ignore(to_test,blank_ref):
    if abs(to_test[0] - blank_ref[0]) <= 40 and abs(to_test[1] - blank_ref[1]) <= 40 and abs(to_test[2] - blank_ref[2]) <= 40:
        return True
    else:
        return False

# After the mask has been created to isolate the samples, counts the total
# number of sample pixels (or unmasked pixels), and sums up the values for 
# each of the RGB channels.
def color_detection(mask, image_path):
    
    # opens the image to be masked
    image = Image.open(image_path, 'r')
    # converts image to an array
    img_arr =np.array(image)

    pix_num = np.sum(mask[:,:,1])
    final_image = img_arr * mask

    
    red = np.sum(final_image[:,:,0]) / pix_num
    green = np.sum(final_image[:,:,1]) / pix_num
    blue = np.sum(final_image[:,:,2]) / pix_num
    
    print(image_path)
    
    return [red, green, blue]

def color_detection_for_filter(mask, image_path):
    
    image = Image.open(image_path, 'r')
    img_arr =np.array(image)

    pix_num = np.sum(mask[:,:,1])
    final_image = img_arr * mask

    
    print(image_path)
    
    return final_image


def ignore_filter(t0_image_path, well_of_interest):
    mask = create_mask(t0_image_path, well_of_interest)
    to_ignore = color_detection(mask, t0_image_path)
    return to_ignore

# This function takes in the path of the first image with the samples, the well the user
# wishes to analyze, and the image path of the blank well image, as well as
# which wells is being used as a blank. From those inputs, a mask is created
# that isolates the samples in the well of interest.
def filtered_mask_2(t0_image_path, well_of_interest, blank_path, blank_well):
    
    
    #image = cv2.imread(t0_image_path)
    wells = get_well(t0_image_path) # Extracts the wells from the image
    if wells is not None:
        ordered_wells = well_order(wells) # Gets the order of the wells
    else: 
        ordered_wells = [] # if there are no wells detected, set the wells to empty
    
    
    ## Get the bounding box from circle center and radius from image at t0
    
    # Subtract one from well_of_interest for 0 indexing
    well_of_interest = well_of_interest - 1
    
    # Reduce the radius of the bounding circle by 10 pixels, this makes sure the
    # darker sample holder does not get included in the image analysis.
    x1 = ordered_wells[well_of_interest][0] - (ordered_wells[well_of_interest][2] - 10)
    y1 = ordered_wells[well_of_interest][1] - (ordered_wells[well_of_interest][2] - 10)
    x2 = ordered_wells[well_of_interest][0] + (ordered_wells[well_of_interest][2] - 10)
    y2 = ordered_wells[well_of_interest][1] + (ordered_wells[well_of_interest][2] - 10)
    
    
    ## With the bounding box, create the mask to isolate the well of interest
    #Image.display(image)
    image = Image.open(t0_image_path, 'r')
    height,width = image.size
    lum_img = Image.new('L', [height,width] , 0)
      
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(x1,y1), (x2,y2)], 0, 360, 
                  fill = 255, outline = "white")
    lum_img_arr =np.array(lum_img)
    
    mask = np.dstack((lum_img_arr/255,lum_img_arr/255,lum_img_arr/255))
    
    
    # Then, use the filter paper blank well to subtract the filter paper background from the sample wells.
    raw = color_detection_for_filter(mask, t0_image_path)
    print(blank_path)
    to_ignore = ignore_filter(blank_path, blank_well)
    
    
    new_mask = np.empty([1080,1440])
    for y in range(len(raw[:,1,1])):
        for x in range(len(raw[1,:,1])):
            if test_ignore(raw[y,x,:], to_ignore) == True:
                new_mask[y,x] = 0
            else:
                new_mask[y,x] = mask[y,x,1]
                
    plt.figure(0)   
    plt.imshow(new_mask)
    new_mask_stacked = np.dstack((new_mask, new_mask, new_mask))
    
    return new_mask_stacked


#%% For each of the 9 sample wells, extracts the RGB values
for well_run in range(1, total_wells + 1):
    # Gets the total number of images
    image_num = os.listdir(chamber_data_path + expt_run + "/" + all_images + "/")
    # Creates the mask isolating the samples for the current well, using the circle detection 
    # and then the filter paper color to isolate the samples within 
    # those circles.
    mask = filtered_mask_2(chamber_data_path + expt_run + "/" + all_images + "/frame9000.jpg", well_run, chamber_data_path + expt_run + "/" + all_images + "/frame9000.jpg", blank_well )
    # Applies the mask to each frame, the gets the average RGB value of the sample
    all_colors = [color_detection(mask, chamber_data_path + expt_run + "/" + all_images + "/frame" + str(ims) + ".jpg") for ims in range(first_frame, len(image_num), output_sample_rate)]
    
    # For the first well, find the total 
    if well_run == 1:
        all_sample_rgb = np.ones((total_wells,len(all_colors),3))
    
    all_sample_rgb[well_run - 1,:,:] = np.array(all_colors)
    

#%% Loads the RGB data, runs the calibration, and then regraphs.

# Calls the calibration function stored in the Color_transformation script.
# Performs a 3D thin spline transformation. The input is an np array with 
# dimensions (n_samples, {optional dimension: n_times}, n_color_coordinates=3).
sample_rgb_calibrated = haverford_color_calibration(all_sample_rgb)[0]

# Iterates through each well graphing the RGB values. Saves the RGB values to 
# a csv.
for well_run in range(1, total_wells + 1):
    
    # Pulls the RGB values for the current well
    red_tseries = sample_rgb_calibrated[well_run - 1,:,0]
    green_tseries = sample_rgb_calibrated[well_run - 1,:,1]
    blue_tseries = sample_rgb_calibrated[well_run - 1,:,2]
    
    # Plots the RGB values
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(red_tseries[:], 'r')
    ax.plot(green_tseries[:], 'g')
    ax.plot(blue_tseries[:], 'b')
    ax.set_ylim([0,255]) #80, 130
    
    plt.title('Color Time Series for Well ' + str(well_run))
    plt.xlabel('Time: (min x ' + str(output_sample_rate) +')')
    plt.ylabel('RGB Values')
    
    plt.show()
    
    # Saves the RGB Values as a CSV, and then saves the plot.
    np.savetxt(chamber_data_path + expt_run + '/' + expt_run + '_Color_Time_Series_for_Well ' + str(well_run) + '.csv', sample_rgb_calibrated[well_run - 1,:,:], delimiter=",", header="R,G,B", comments='')
    plt.savefig(chamber_data_path + expt_run + '/' + expt_run + '_Calibrated_Color_Time_Series_for_Well ' + str(well_run) + '.png')
    
    plt.close()