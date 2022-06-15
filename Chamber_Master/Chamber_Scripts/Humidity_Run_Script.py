from datetime import datetime
import serial
import os

#%% Overview:

# This is the data logging script. Every experiment starts by running this script
# on the host computer while the chamber is connected. Run this script using Thonny,
# and choose the default interpreter. This script will create an experiment file
# where all of the sensor data will be recorded. Start this script first, and
# then start the Solo Configuration application, and the Thorlabs camera application.


#%% User input:

#Enter in the root folder path for data storage
data_path = "C:/Users/Enviro_Chamber/Desktop/Chamber_Data/"


#%%
now = datetime.now()
current_time = now.strftime("%D:%H:%M:%S")
print("Current Time =", current_time)
filename = input('Please enter a filename for the experiment \n')

#Creates the new files
os.mkdir(data_path + filename)

#Creates a text file to store the humidity and ambient temp data
f = open(data_path + filename + "\humidity_temp_data.txt", "a")




ser = serial.Serial('COM6')  # open serial port

# reads the sensor serial inputs, stores them in the 
while True:
    f = open(data_path + filename + "\humidity_temp_data.txt", "a")
    now = datetime.now()
    current_time = now.strftime("%D:%H:%M:%S")
    line = ser.readline().decode("utf-8")
    data = line.rsplit(" ")
    temp = data[0]
    humidity = data[1].replace("\r\n","")
    f.write(current_time + ", " + temp + ", " + humidity + "\n")
    print(temp, ", ", humidity)
    f.close()

