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

# from DBSCAN.py:
from csv import reader
import numpy as np
from sklearn.cluster import DBSCAN


BOTTOM_THRESHHOLD = 1410
MID_THRESHHOLD = 680 # threshhold to not detect under [pixels] can be determined using data-visualization.py
DISPLAY_OPTION = 0 # controls wether or not the images are rendered to the gui

def doFilesExist(filename,folderPath):
	# see if both files exist
	jpg_file_exists = os.path.splitext(folderPath + "JPEGImages/" + filename + ".jpg")[1] == ".jpg" 
	xml_file_exists = os.path.splitext(folderPath + "Annotations/" + filename + ".xml")[1] == ".xml" 
	return (jpg_file_exists and xml_file_exists)

def numOfCarsJPEG(filename,display,channel,folderPath,writer):
	
	img = jetson.utils.loadImage(folderPath + "JPEGImages/" + filename + ".jpg")
	detections = net.Detect(img)

	Occupied = np.zeros(5) # expected number of classes is four parking spots + 1 for outliers
	for detection in detections:
		#parkingSpot = CarPositionDBSCAN(detection.Center,channel)
		#Occupied[parkingSpot] = 1
		writer.writerow(detection.Center)
	return Occupied, img

def numOfCarsXML(filename,folderPath):
	#crawl xml tree 
	tree = ET.parse(folderPath + "Annotations/" + filename + ".xml")
	root = tree.getroot()
	q_xml = 0
	for name in root.iter("name"):
		if name.text == "Car":
			q_xml = q_xml+1
	return q_xml

def displayImage(img,filename,folderPath):
	#display.Render(img)
	jetson.utils.saveImage(folderPath + "detectnet-out/" + filename + ".jpg", img)

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

def createCsvWriter(dataPath):
	# create a csv writer
	f = open(dataPath + "detectnet-car-positions.csv","w") 
	writer = csv.writer(f) 
	header = ['x','y']
	writer.writerow(header)
	return writer

def CarPositionDBSCAN(carCenter,channel):
	# datasetFormatted_ch2, datasetFormatted_ch4, labelToParkingSpot_ch2, labelToParkingSpot_ch4
	
	carCenter = [carCenter]
	print(carCenter)
	if (channel == 2):
		datasetFormattedWithPoint = np.append(datasetFormatted_ch2,carCenter,axis=0) # append new car to data
		clustering = DBSCAN(eps = EPS2, min_samples = MIN_SAMPLES).fit(datasetFormattedWithPoint)
		parkingSpot = clustering.labels_[-1] # extract predicted car parking spot

		print("parkingSpot: " + str(parkingSpot))

		if (0<=parkingSpot<=3):
			parkingSpot = labelToParkingSpot_ch2[int(parkingSpot)] # map to real world park spot
			return parkingSpot
		else:
			return parkingSpot

	if (channel == 4):
		datasetFormattedWithPoint = np.append(datasetFormatted_ch4,carCenter,axis=0) # append new car to data

		print(datasetFormattedWithPoint)

		clustering = DBSCAN(eps = EPS4, min_samples = MIN_SAMPLES).fit(datasetFormattedWithPoint)
		parkingSpot = clustering.labels_[-1] # extract predicted car parking spot

		print("parkingSpot: " + str(parkingSpot))


		if (1<=parkingSpot<=4):
			parkingSpot = 3-labelToParkingSpot_ch4[int(parkingSpot)]
			return parkingSpot
		else:
			return parkingSpot
	else:
		print("no channel selected")

def csvWriteLoop(folderPath,channel):
	arq = sorted(os.listdir(folderPath + "JPEGImages/")) # list of all filenames
	writer = createCsvWriter(folderPath)
	file  = 0
	while True:
		filename = os.path.splitext(arq[file])[0] # filename ("frame1345_ch2")
		if(doFilesExist(filename,folderPath)):
			data = [filename]
		
			#numOfCarsAnnotated = numOfCarsXML(filename,folderPath) 
			#data.append(numOfCarsAnnotated)

			numOfCars,img = numOfCarsJPEG(filename,DISPLAY_OPTION,channel,folderPath,writer)
			#occupiedSpots = sum(numOfCars[:4])
			
			#data.append(occupiedSpots)
			#data.append(numOfCars)

			#if(occupiedSpots != numOfCarsAnnotated): # if the 
			displayImage(img,filename,folderPath)

#			writer.writerow(data) # only write to csv if XML files are same number of annotated cars

		file = file + 1
		if file >= len(arq):
			break


# Path to model and pictures
modelPath = '/home/gustav/direc_mount/Annotations/Annotated_models/new-angle-ch4_run2/'
savePath = '/home/gustav/direc_mount/Annotations-merged/DBSCAN-reliability/'

modelString = ['--model=' + modelPath + 'ssd-mobilenet.onnx', '--input-blob=input_0','--output-cvg=scores', '--output-bbox=boxes','--labels=' + modelPath + 'labels.txt' ]

# for the DBSCAN to work: 
EPS4 = 129 
EPS2 = 154 # determined experimentally in DBSCAN.py
MIN_SAMPLES = 10
folderPath_ch2 = '/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/'
folderPath_ch4 = '/home/gustav/direc_mount/Annotations/annotations-new-angle_ch4/'
CarPositions_ch2 = load_csv(folderPath_ch2 + "carPositions.csv")
CarPositions_ch4 = load_csv(folderPath_ch4 + "carPositions.csv")
datasetFormatted_ch2 = CarPositions_ch2[1:,1:3].astype(float) # skips the header and gives the right coloumn
datasetFormatted_ch4 = CarPositions_ch4[1:,1:3].astype(float) # skips the header and gives the right coloumn
labelToParkingSpot_ch2 = [3, 1, 0, 2] # determined in DBSCAN.py
labelToParkingSpot_ch4 = [0, 2, 3, 1] # determined in DBSCAN.py




#arq2 = sorted(os.listdir(folderPath_ch2 + "JPEGImages/")) # list of all filenames
#arq4 = sorted(os.listdir(folderPath_ch2 + "JPEGImages/")) # list of all filenames
net = jetson.inference.detectNet("ssd-mobilenet-v2", modelString) #load the model


# loop over both folders
csvWriteLoop(folderPath_ch4,4)
csvWriteLoop(folderPath_ch2,2)


if(DISPLAY_OPTION==1):
	display = jetson.utils.videoOutput(folderPath_ch4 + "detectnet-out/") # 'my_video.mp4' for file



if(DISPLAY_OPTION==1):
	display = jetson.utils.videoOutput(folderPath_ch2 + "detectnet-out/") # 'my_video.mp4' for file


print("EOP")
