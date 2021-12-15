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

	channel = int(filename[1:2])
	inverseChannel = 0 
	if (channel == 2):
		inverseChannel = 4
	if (channel == 4):
		inverseChannel = 2
	inverseFileName = filenameOut = "0"+ str(inverseChannel) +"_" + filename[3:]
	inverse_jpg_file_exists = os.path.splitext(folderPath + "JPEGImages_other_channel/" + inverseFileName + ".jpg")[1] == ".jpg" 

	return (jpg_file_exists and xml_file_exists and inverse_jpg_file_exists)

def numOfCarsJPEG(img,channel,writer):
	detections = net.Detect(img)

	Occupied = np.zeros(5) # expected number of classes is four parking spots + 1 for outliers
	for detection in detections:
		parkingSpot = CarPositionDBSCAN(detection.Center,channel)
		Occupied[parkingSpot] = 1
		data = [channel]
		data.append(detection.Center)
		writer.writerow(data)

	return Occupied

def numOfCarsXML(filename,folderPath):
	#crawl xml tree 
	tree = ET.parse(folderPath + "Annotations/" + filename + ".xml")
	root = tree.getroot()
	q_xml = 0
	for name in root.iter("name"):
		if name.text == "Car":
			q_xml = q_xml+1
	return q_xml

def displayImage(img,filename,folderPath,channel):
	#display.Render(img)
	jetson.utils.saveImage(folderPath + "detectnet-out/" + str(channel) + "/" + filename + ".jpg", img)

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

def createCsvWriter(filePath,header):
	# create a csv writer
	f = open(filePath,"w") 
	writer = csv.writer(f) 
	writer.writerow(header)
	return writer

def CarPositionDBSCAN(carCenter,channel):
	# datasetFormatted_ch2, datasetFormatted_ch4, labelToParkingSpot_ch2, labelToParkingSpot_ch4
	
	carCenter = [carCenter]
	if (channel == 2):
		datasetFormattedWithPoint = np.append(datasetFormatted_ch2,carCenter,axis=0) # append new car to data
		clustering = DBSCAN(eps = EPS2, min_samples = MIN_SAMPLES).fit(datasetFormattedWithPoint)
		parkingSpot = clustering.labels_[-1] # extract predicted car parking spot

		if (0<=parkingSpot<=3):
			parkingSpot = labelToParkingSpot_ch2[int(parkingSpot)] # map to real world park spot
			return parkingSpot
		else:
			parkingSpot = 4
			return parkingSpot

	if (channel == 4):
		datasetFormattedWithPoint = np.append(datasetFormatted_ch4,carCenter,axis=0) # append new car to data
		clustering = DBSCAN(eps = EPS4, min_samples = MIN_SAMPLES).fit(datasetFormattedWithPoint)
		parkingSpot = clustering.labels_[-1] # extract predicted car parking spot

		if (0<=parkingSpot<=3):
			parkingSpot = 3-labelToParkingSpot_ch4[int(parkingSpot)]
			return parkingSpot
		else:
			parkingSpot = 4
			return parkingSpot 
	else:
		print("no channel selected")

def csvWriteLoop(folderPath,channel):
	arq2 = sorted(os.listdir(folderPath + "JPEGImages/")) # list of all filenames
	arq4 = sorted(os.listdir(folderPath + "JPEGImages_other_channel/")) # list of all filenames
	writerHeader = ['frame_id','annotations','occupiedSpots_OR ','occupiedSpots2 ','occupiedSpots4 ','numOfCars_OR ','numOfCars2 ','numOfCars4']	
	writer = createCsvWriter(folderPath + "DBSCAN-reliability-lowest-eps.csv",writerHeader)
	detectnetHeader = ['x2','y2','x4','y4']
	detectnetCenterWriter = createCsvWriter(folderPath + 'detectnet-car-positions.csv',detectnetHeader)
	file  = 0
	if(len(arq2) != len(arq4)):
		print("length of two arrays are different")
		print(len(arq4))
		return

	while True:
		filename2 = os.path.splitext(arq2[file])[0] # filename ("frame1345_ch2")
		filename4 = os.path.splitext(arq4[file])[0] # filename ("frame1345_ch2")
		if(doFilesExist(filename2,folderPath)):
			data = [filename2]
		
			numOfCarsAnnotated = numOfCarsXML(filename2,folderPath) 
			data.append(numOfCarsAnnotated)

			img2 = jetson.utils.loadImage(folderPath + "JPEGImages/" + filename2 + ".jpg")
			img4 = jetson.utils.loadImage(folderPath + "JPEGImages_other_channel/" + filename4 + ".jpg")
			numOfCars2 = numOfCarsJPEG(img2,channel,detectnetCenterWriter)
			numOfCars4 = numOfCarsJPEG(img4,4,detectnetCenterWriter)


			#numOfCars4[:4] = np.flip(numOfCars4[:4])
			numOfCarsTotal = np.logical_or(numOfCars4, numOfCars2)

			occupiedSpotsTotal = sum(numOfCarsTotal[:4])
			occupiedSpots2 = sum(numOfCars2[:4])
			occupiedSpots4 = sum(numOfCars4[:4])
			


			data.append(occupiedSpotsTotal)
			data.append(occupiedSpots2)
			data.append(occupiedSpots4)
			data.append(numOfCars2)
			data.append(numOfCars4)
			data.append(numOfCarsTotal)

#			if(occupiedSpots != numOfCarsAnnotated): # if the 
			#displayImage(img2,filename2,folderPath,2)
			#displayImage(img4,filename4,folderPath,4)

			writer.writerow(data) # only write to csv if XML files are same number of annotated cars

		print(file)
		file = file + 1
		if file >= len(arq2):
			break


# Path to model and pictures
modelPath = '/home/gustav/direc_mount/Annotations/Annotated_models/new-angle-ch4_run2/'
savePath = '/home/gustav/direc_mount/Annotations-merged/DBSCAN-reliability/'

modelString = ['--model=' + modelPath + 'ssd-mobilenet.onnx', '--input-blob=input_0','--output-cvg=scores', '--output-bbox=boxes','--labels=' + modelPath + 'labels.txt' ]

# for the DBSCAN to work: 
#EPS4 = 129 
#EPS2 = 154 # determined experimentally in DBSCAN.py

EPS4 = 74
EPS2 = 75

MIN_SAMPLES = 10
folderPath_ch2 = '/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/'
folderPath_ch4 = '/home/gustav/direc_mount/Annotations/annotations-new-angle_ch4/'
CarPositions_ch2 = load_csv(folderPath_ch2 + "carPositions.csv")
CarPositions_ch4 = load_csv(folderPath_ch4 + "carPositions.csv")
datasetFormatted_ch2 = np.c_[CarPositions_ch2[1:,2],CarPositions_ch2[1:,1]].astype(float) # load it in correctly
datasetFormatted_ch4 = np.c_[CarPositions_ch4[1:,2],CarPositions_ch4[1:,1]].astype(float) # skips the header and gives the right coloumn
labelToParkingSpot_ch2 = [3, 2, 0, 1] # determined in DBSCAN.py
labelToParkingSpot_ch4 = [0, 2, 3, 1] # determined in DBSCAN.py




#arq2 = sorted(os.listdir(folderPath_ch2 + "JPEGImages/")) # list of all filenames
#arq4 = sorted(os.listdir(folderPath_ch2 + "JPEGImages/")) # list of all filenames
net = jetson.inference.detectNet("ssd-mobilenet-v2", modelString) #load the model


# loop over both folders
csvWriteLoop(folderPath_ch2,2)

#csvWriteLoop(folderPath_ch4,4)



print("EOP")
