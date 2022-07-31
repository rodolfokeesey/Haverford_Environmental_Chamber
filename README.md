This repository is for the Generation 2 Haverford chamber in an open-hardware project for developing degradation chambers for perovskite materials (see figure below). It contains build instructions, 3D assets, and the codebase for replicating the design. 
This chamber has the capability of controlling sample temperature, humidity, and illumination of bulk perovskite samples. Sample degredation is recorded via automated color calibrated imaging.

![Degradation chamber generations in the open-hardware project. This repository describes Haverford Gen. 2 chamber.](https://github.com/PV-Lab/hte_degradation_chamber/blob/main/Chamber_generations.png)

HARDWARE BUILD GUIDE:
- For the hardware build guide, please see the build guide located within the Documentation folder

PREREQUISITE SOFTWARE
- ThorLabs Camera Software (For time-lapse image collection)
	https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam
- Thonny (For ambient temperature and humidity logging via raspberry pi pico)
	https://thonny.org/
- Automations Direct Solo Configuration Software (For sample temperature recording)
	https://www.automationdirect.com/adc/shopping/catalog/software_products/process_control_-a-_measurement_software/sl-soft

Example experimental data and calibration data has been included as en example of expected input and folder hierarchy.


PATHING

These chamber scripts are meant to work with the following folder structure:

Chamber Data # root for all the chamber data. This is the path you enter for chamber_data_path. You can place this folder wherever. <br />
             # A sample folder is included in the repository as Chamber_Test_folder with one example experiment.<br />
-> RK_37 # example experiment run<br />
-> RK_38 # example experiment tun<br />
-> Color_Calibration_5_6_22 # Folder storing the two Xrite Color Checker Images<br />
-> Video_Queue # The temporary file that where the time lapse .avi video from the Thorlabs application is saved<br />
-> Sample_Temp_Queue # The temporary file that where the .txt log from the Automation's Direct Solo Configuration application is saved<br />

ENVIRONMENT

These codes have been tested in Windows 10 in anaconder using Spyder.

1) Download and install anaconda for windows

2) Download (if you do not already have) the chamber data processing codes.

3) Navigate to the location of the downloaded files

4) Deactivate any current conda instance

5) Create a new python 3.7 environment:

	conda create -n aging_chamber python=3.7

6) Activate the new environment

	conda activate aging_chamber

7) Install the necessary python packages using pip

	pip install -r requirements.txt

8) Through the Anaconda Navigator, install Spyder 5.1.5.

CALIBRATION (calibration_color_extraction.py)

1) Before beginning chamber use, the color calibration reference photos of the Xrite Color Checker must be taken.

2) Take two photos of the Xrite Color Checker. The orientation of the photos must match the included examples.

3) Once the two calibration photos have been taken, run the calibration_color_extraction.py script, and follow the prompts.
After calibration is completed, the Aging Chamber is ready for use.

DATA LOGGING (Humidity_Run_Script.py)

1) To begin an experiment, start by connecting the aging chamber and then running the Humidity_Run_Script in Thonny. This
script creates an experiment folder, and records the ambient temperature and humidity within the chamber. The sample holder
temperature and images are recorded by the Solo Configuration application and the ThorCam application, respectively. 

2) End the script once the desired aging time has been reached. Then transfer the file from the Solo Configuration, and the video
from the ThorCam application into the newly created experiment folder. After the data logging for a run is completed, there should
be three files within an experiment folder (for example see, RK_37): 
- An .avi video file, the default filename is "image_0.avi"
- A humidity log .txt file, the default filename is "humidity_temp_data.txt"
- A sample holder temperature log .txt file, the default filename is "Sample_Temp_Queue"

DATA PROCESSING (1. time_sync.py, 2. Image_processer.py, Color_transformation.py)

1.) After terminating the Humidity_Run_Script, and moving the video file and the sample holder log file into the experiment folder, open the 
time_sync.py script in an environment with the chamber dependencies. Fill out the User Inputs section, then run the script. This script will 
read all of the log files, break the timelapse video into individual frames for RGB analysis, and sync the sensor timing with the 
image capture. It will also generate a graph with the chamber's ambient temperature, relative humidity, and the sample holder temperature
for the duration of the experiment.

2.) Next open the Image_processor script. Fill out the User Inputs section, then run the script. This script will go well by well of the
sample holder, create a mask using the blanked well as reference, isolate the bulk crystals in the well, then extract the RGB Values.
Once all the wells have been processed, the raw RGB values are transformed using the Color_transformation.py script. The final output is
a .png image of the RGB change over time and a CSV of the RGB data for each well.