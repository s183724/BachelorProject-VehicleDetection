#!/usr/bin/python3
# script that takes images as input, detects the number of vehicles in each frame, 
# and outputs the each analyzed frame in data folder, and number of vehicles in a csv. 


import cv2
import jetson.inference
import jetson.utils
import os
import csv
import xml.etree.ElementTree as ET
import argparse

BOTTOM_THRESHHOLD = 1410
MID_THRESHHOLD = 680 # threshhold to not detect under [pixels] can be determined using data-visualization.py
DISPLAY_OPTION = 0 # controls wether or not the images are rendered to the gui

def doFilesExist(filename):
	# see if both files exist
	jpg_file_exists = os.path.splitext(dataPath + "JPEGImages/" + filename + ".jpg")[1] == ".jpg" 
	xml_file_exists = os.path.splitext(dataPath + "Annotations/" + filename + ".xml")[1] == ".xml" 
	return (jpg_file_exists and xml_file_exists)

def numOfCarsJPEG(filename,display):
	img = jetson.utils.loadImage(dataPath + filename + ".jpg")
	detections = net.Detect(img)
	parkingSpot1 = 0
	parkingSpot2 = 0
	# if list is not empty count the number of cars (ID = 1) where care is placed over threshhold		
	for detection in detections:
		if(detection.ClassID == 1 and detection.Bottom > MID_THRESHHOLD and detection.Bottom < BOTTOM_THRESHHOLD):
			#print("bottom box coordinate: " + str(detection.Bottom))
			parkingSpot1 = 1
		elif(detection.ClassID == 1 and detection.Bottom >= BOTTOM_THRESHHOLD):
			parkingSpot2 = 1

	return parkingSpot1+parkingSpot2, img

def numOfCarsXML(filename):
	#crawl xml tree 
	tree = ET.parse(dataPath + "Annotations/" + filename + ".xml")
	root = tree.getroot()
	q_xml = 0
	for name in root.iter("name"):
		if name.text == "Car":
			q_xml = q_xml+1
	return q_xml

def saveImage(img,filename):
	#display.Render(img)
	#img = jetson.utils.loadImage(dataPath + filename + ".jpg")
	jetson.utils.saveImage(savePath + filename + ".jpg", img)



# Path to new angle model and pictures
#dataPath = "/home/gustav/direc_mount/new-angle-pics-not-annotated/ch4/"
#modelPath = '/home/gustav/direc_mount/Annotations/Annotated_models/new-angle-ch4_run2/'
#savePath = '/home/gustav/direc_mount/new-angle-pics-not-annotated/ch4/detectnet-new-angle-2/'

# Path to old angle model and pictures
dataPath = "/home/gustav/direc_mount/Annotations/annotations3_ch2/JPEGImages/"
modelPath = '/home/gustav/direc_mount/General model/Car_models/'
savePath = '/home/gustav/direc_mount/old-angle-pics/detectnet-no-model-out/'

# use model 
#modelString = ['--model=' + modelPath + 'ssd-mobilenet.onnx', '--input-blob=input_0','--output-cvg=scores', '--output-bbox=boxes','--labels=' + modelPath + 'labels.txt' ]

# dont use model 
modelString = ['--input-blob=input_0','--output-cvg=scores', '--output-bbox=boxes','--labels=' + modelPath + 'labels.txt' ]


arq = sorted(os.listdir(dataPath)) # list of all filenames
net = jetson.inference.detectNet("ssd-mobilenet-v2", modelString) #load the model

if(DISPLAY_OPTION==1):
	display = jetson.utils.videoOutput(savePath) # 'my_video.mp4' for file

# create a csv writer
f = open(savePath + "csv.csv","w") 
writer = csv.writer(f) 
header = ['frame_id','annotated number of cars','Total number of cars','Number of cars 2','Number of cars 4']
writer.writerow(header)


q_detect = 0
q_xml = 0
file  = 0
XMLNotTheSame = 0
while True:

	filename1 = os.path.splitext(arq[file])[0] # filename ("frame1345_ch2")

	numOfCars1,img1 = numOfCarsJPEG(filename1,DISPLAY_OPTION)
#	writer.writerow(data) # only write to csv if XML files are same number of annotated cars
	saveImage(img1,filename1)
	file = file + 1
	if file >= len(arq):
		break

print("number fo XML that are not the same: " + str(XMLNotTheSame))
print("EOP")
