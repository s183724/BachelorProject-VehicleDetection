#!/usr/bin/python3
import datetime
from datetime import time
from csv import reader
import numpy as np
import cv2
import os

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)			
	return np.array(dataset) 

def filename_to_image(filename):
# converts filename of type: "02_20211102143212_309270.jpg" 
# to filename of type: "NVR_ch2_main_20211031000000_20211031010000.mp4"
	
	channel = int(filename[1:2])
	year = int(filename[3:7])
	month = int(filename[7:9])
	day = int(filename[9:11])
	hour = int(filename[11:13])
	minutes = int(filename[13:15])
	seconds = int(filename[15:17])

	inverseChannel = 0 
	if (channel == 2):
		inverseChannel = 4
	if (channel == 4):
		inverseChannel = 2

	time = datetime.datetime(year,month,day,hour,minutes,seconds)
	time2 = time + datetime.timedelta(hours=1)
	camera_channel = "_ch" + str(inverseChannel)
	videoName = ("NVR" + camera_channel + "_main_2021"+ str(time.month).zfill(2) + "" + str(time.day).zfill(2) + "" + str(time.hour).zfill(2) + "0000_2021" + str(time2.month).zfill(2) + "" + str(time2.day).zfill(2) + "" + "" + str(time2.hour).zfill(2)+ "0000.mp4")


	# convert a videoname to a jpg and saves it in the specified folder

	cam = cv2.VideoCapture(videoFolderPath +  videoName)  # Read the video from specified path
	secondsOffset = -1
	framenumber = (time.minute*60 + time.second + secondsOffset)*FPS
	cam.set(1,framenumber) # set the correct frame to save to
	ret,frame = cam.read() 

	if (ret): # if video is still left continue creating images
		filenameOut = "0"+ str(inverseChannel) +"_" + filename[3:]
		pictureDest = savePath + filenameOut

		print ('Saving frame to: ' + pictureDest)
		cv2.imwrite(pictureDest, frame) # saving the file
		return 0
	else:
		print("no frame in path at file: " + filename[3:] + ".jpg")
		return 1

	
FPS = 25

# load csv with filenames 
videoFolderPath = "/media/gustav/Clever/Videos/"
imageFolderPath = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/"
savePath = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/JPEGImages_other_channel/"

#csv_loaded = load_csv(csvFolderPath + "DBSCAN-reliability.csv")
#filenames = csv_loaded[1:,0]
#annotations = csv_loaded[1:,1]


arq = sorted(os.listdir(imageFolderPath + "JPEGImages/"))

counter = 0

for filename in arq:
	counter = counter + filename_to_image(filename)

print(counter)




