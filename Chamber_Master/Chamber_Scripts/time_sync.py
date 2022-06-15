import os
import time
import math
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

#%% Overview:

# This script is the first step in aging chamber data processing. 

# It breaks the time-lapse .avi video recorded from the Thorlabs application into 
# individual frames for RGB analysis.

# It then synchronizes the start times and sample rates of all the raw data 
# from the humidity/temp sensor, the Solo temperature controller for the 
# sample tray, and the image capture. This data is then outputted as a graph
# to give an overview of the run.


#%% User Inputs

# Enter in the file names for the time sync, and the start parameters for the aging run.

# The path for the chamber's data folder
chamber_data_path = "C:/Users/Rod/Documents/Chamber_Test_Folder/"
# Enter in the experiment file you'd like to sync data for.
expt_run = "RK_37"
# Enter the name of the video for the run
video_path = "image_0.avi"
# Enter the name of the compiled image file
all_images = "all_frames" 
# Number of seconds between frames of image capture
image_sample_rate = 60
# Year of the experiment run
Year = "22 " 

#%% Image Processing Functions. These are used to extract the well locations, the orders etc.


# Gets the radius and center of all the wells with circle detection
def get_well(image):
    
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
    gray = cv2.medianBlur(gray, 5)
        
        
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=70, maxRadius=110)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)


    #cv2.imshow("detected circles", image)
    #cv2.waitKey(0)
    return(circles)

def well_order(circles):
    #Order in terms of greatest Y value,
    sortbyrow = np.argsort(circles[0][:,1])
    r1 = circles[0][sortbyrow[0:3]]
    r2 = circles[0][sortbyrow[3:6]]
    r3 = circles[0][sortbyrow[6:9]]
    #[0:2] are in row 1
    #[3:5] are in row 2
    #[6:8] are in row 3
    
    #Order in terms of least X value,
    r1sorted = r1[np.argsort(r1[:,0])]
    r2sorted = r2[np.argsort(r2[:,0])]
    r3sorted = r3[np.argsort(r3[:,0])]
    
    #Get final order from x coordinate, Stack altogether, get final order
    
    ordered_wells = np.vstack((r1sorted,r2sorted,r3sorted))

    
    return(ordered_wells)

def get_coldist(image,wellx,welly,wellradius):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask,(wellx,welly),wellradius,255,-1)
    masked = cv2.bitwise_and(image, image, mask=mask)

    maskedhisto = cv2.calcHist([masked],[0],None,[256],[0,256]) 
    #cv2.imshow("Mask Applied to Image", masked)
    #cv2.waitKey(0)
    return(maskedhisto)



#%% Takes the time lapse .avi video, extracts the frames as images and saves 
### them a to file
if os.path.isdir(chamber_data_path + expt_run + '/' + all_images) != True:
    # Makes a new file to save the extracted frames, then navigates to the file
    os.mkdir(chamber_data_path + expt_run + '/' + all_images)
    os.chdir(chamber_data_path + expt_run + '/' + all_images)
    
    # Finds the video, then loops through extracting each frame
    vidcap = cv2.VideoCapture(chamber_data_path + expt_run + "/" + video_path)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1

#%% Finds the start time of ambient/humidity and sample holder temp data sets. 
### then puts ambient temperature, humidity, and sample holder temperature into 
### three different lists.

# Reads all the files in the experiment directory so we can find...
all_files = os.listdir(chamber_data_path + expt_run + "/")

# ... the file containing the extracted images
image_file = [x for x in all_files if "image" in x][0]
# ... the file containing the humidity and ambient temperature data
hum_amb_file = [x for x in all_files if "humidity_temp_data" in x][0]
# ... the file containing the sample holder temperature
s_holder_file = [x for x in all_files if "Sample_Temp" in x][0]



# Opens the humidity/ambient log, reads all the lines, pulls the first time
# as t1, or the experiment start time
with open(chamber_data_path + expt_run + "/" + hum_amb_file) as f:
    hum_amb_lines = f.readlines()
f.close()
# Creates a list with the time of each reading, in datetime format.
hum_amb_time = [datetime.strptime(x.split(", ")[0], "%m/%d/%y:%H:%M:%S") for x in hum_amb_lines] 
# Creates a list of the ambient temperature readings
amb_temp = [x.split(", ")[1] for x in hum_amb_lines]
# Creates a list of the ambient humidity readings
rh = [x.split(", ")[2] for x in hum_amb_lines]
# compiles the time, ambient temperature, and relative humidity into a list
hum_amb_proc = [hum_amb_time, amb_temp, rh]

# Records the time of the first reading from the humdity/temperature sensor.
hum_amb_t1 = hum_amb_time[0]



#Opens the Sample holder log, reads all the lines, pulls the first time as t1

with open(chamber_data_path + expt_run + "/" + s_holder_file) as f:
    s_holder_lines = f.readlines()
f.close()
# Removes the first line, which is unneeded
s_holder_lines = s_holder_lines[1:] 
# Creates a list with the time of each reading, in datetime format.
s_holder_time = [datetime.strptime(Year + ' '.join(x.split(" ")[1:4]), "%y %b %d %H:%M:%S") for x in s_holder_lines]
# Creates a list of the sample holder temperature readings
s_holder_temp = [x.split(" ")[5].split(",")[1] for x in s_holder_lines]
# Creates a list of the target sample holder temperature
s_holder_settemp = [x.split(" ")[5].split(",")[2] for x in s_holder_lines]
# compiles the time, sample holder temperature, and set temperature into a list
s_holder_proc = [s_holder_time,s_holder_temp,s_holder_settemp]

s_holder_t1 = s_holder_time[0]



# Compares the start times of the humidity/ambient temperature logger, and the
# sample holder controller. Finds the latest start time, and sets that time
# as the start of the experiment
if hum_amb_t1 < s_holder_t1: # If the humidity was recorded first
    run_t1 = s_holder_t1
    image_t1 = s_holder_t1
else: # If the sample holder temperature was recorded first
    run_t1 = hum_amb_t1
    image_t1 = hum_amb_t1

#%% Image loading and timing

# Loads the image frames of the samples, and counts the number. Creates a key 
# linking each image to a time, with the first image assumed to be recorded at
# the experiment start time. 

# Gets the number of images
image_num = os.listdir(chamber_data_path + expt_run + "/" + all_images + "/")
# Creates image timekey
image_time_key = [image_t1 + timedelta(seconds = image_sample_rate * x) for x 
                  in range(len(image_num))] 



#%% Convert all the times to elapsed seconds, truncate the start times

s_holder_elapsed = [(s_holder_time[x] - run_t1)/timedelta(seconds=1) for x 
                    in range(len(s_holder_time))]

s_holder_elapsed = [x for x in s_holder_elapsed if x > 0]
s_holder_tc = s_holder_temp[len(s_holder_temp) - len(s_holder_elapsed): len(s_holder_temp)]


hum_amb_elapsed = [(hum_amb_time[x] - run_t1)/timedelta(seconds=1) for x in range(len(hum_amb_time))]
hum_amb_elapsed = [x for x in hum_amb_elapsed if x > 0]
amb_tc = amb_temp[len(amb_temp) - len(hum_amb_elapsed) : len(amb_temp)]
hum_tc = rh[len(rh) - len(hum_amb_elapsed) : len(rh)]

img_elapsed = [(image_time_key[x] - run_t1)/timedelta(seconds=1) for x in range(len(image_time_key))]
img_elapsed = [x for x in img_elapsed if x > 0]


#%% We need to create a function for linearly interpolating between the data points
### as each sensor records at different rates.
def interp_readings(target_time, times, data):

    direct = [x for x in range(len(times)) if times[x] == target_time]
    
    if bool(direct) == False:
        before = [x for x in range(len(times)) if times[x] < target_time]#[-1]
        after = [x for x in range(len(times)) if times[x] > target_time]#[0]
        
        if bool(before) == False:
            yout = np.interp(target_time, [float(times[after[0]]), float(times[after[1]])], [float(data[after[0]]), float(data[after[1]])])
        elif bool(after) == False:
            yout = np.interp(target_time, [float(times[before[-2]]), float(times[before[-1]])], [float(data[before[-2]]), float(data[before[-1]])])
        else:
            yout = np.interp(target_time, [float(times[before[-1]]), float(times[after[0]])], [float(data[before[-1]]), float(data[after[0]])])
    else:
        yout = data[direct[0]]
    print(target_time)
        
    return[target_time, float(yout)]


#%% Finalizing the data, and syncing it to the same sample rate. The sample rate
### is ultimately dictated by the frame rate of the time lapse photos. This
### is because the time lapse has the slowest sample rate, for our experiments
### we set it to 1/60 hz, or one frame per minute

# x is the elapsed time in minutes, we're setting it to start from 0, 
# to the number of images
x = np.linspace(0,len(image_num)-1,len(image_num))


# first checks if we've already run this test and saved it. If so, just load 
# in the old results
if os.path.isfile(chamber_data_path + expt_run + "/" + 'final_data.pkl') != True: 
    # Creates a list of the elapsed times in seconds, as the sensors read in hz.
    sampletimes = list(x * image_sample_rate) 
    # Queries the sensor data at each frame an image is captured. If there is 
    # no data for that exact time, linearly interpolates the data from the
    # closest readings
    sync_sh = [interp_readings(x, s_holder_elapsed, s_holder_tc)[1] for x in sampletimes]
    sync_amb = [interp_readings(x, hum_amb_elapsed, amb_tc)[1] for x in sampletimes]
    sync_rh = [interp_readings(x, hum_amb_elapsed, hum_tc)[1] for x in sampletimes]
    
    # Once the data is synced, store it in a dictionary and pickle it for easy access.
    # Storing the data prevents the need to rerun this step, as it can be time consuming.
    final_data = {
        "time_sec" : sampletimes,
        "sample_holder_temp" : sync_sh,
        "ambient_temp" : sync_amb,
        "humidity" : sync_rh,
        }
    
    pickle.dump(final_data, open(chamber_data_path + expt_run + "/" + 'final_data.pkl', 'wb'))
else:
    final_data = pickle.load(open(chamber_data_path + expt_run + "/" + 'final_data.pkl', 'rb'))
    sampletimes = final_data.get('time_sec')
    sync_sh = final_data.get('sample_holder_temp')
    sync_amb = final_data.get('ambient_temp')
    sync_rh = final_data.get('humidity')

#%% Graphing


sampletimes_min = [x/60 for x in sampletimes] # Sets sample time to minutes

# More versatile wrapper
fig, host = plt.subplots(figsize=(10,6)) # (default 8,5 (width, height) in inches
# (see https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.subplots.html)
    
par1 = host.twinx()
par2 = host.twinx()
#par3 = host.twinx()
    
host.set_xlim(sampletimes_min[0], sampletimes_min[-1]) 
host.set_ylim(0, max(sync_rh))
par1.set_ylim(21, max(sync_amb))
par2.set_ylim(0, 1.1 * max(sync_sh))

    
host.set_xlabel("Time (min)")
host.set_ylabel("Chamber %RH")
par1.set_ylabel("Ambient Temperature (C)")
par2.set_ylabel("Sample Holder Temperature (C)")


color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.7)
color3 = plt.cm.viridis(.9)
color4 = plt.cm.viridis(1.2)

p1, = host.plot(sampletimes_min, sync_rh, color=color1, label="Relative Humidity")
p2, = par1.plot(sampletimes_min, sync_amb,    color=color2, label="Ambient Temperature (C)")
p3, = par2.plot(sampletimes_min, sync_sh, color=color3, label="Sample Holder Temperature (C)")


lns = [p1, p2, p3]
host.legend(handles=lns, loc='lower left')#loc='best')

# right, left, top, bottom
par2.spines['right'].set_position(('outward', 60))


host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())
#par3.yaxis.label.set_color(p4.get_color())

# Adjust spacings w.r.t. figsize
fig.tight_layout()

plt.savefig(chamber_data_path + expt_run + '/' + 'run_graph_' + expt_run + '.png')