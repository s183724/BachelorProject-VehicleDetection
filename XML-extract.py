#!/usr/bin/python3
# runs through a folder of XML files and for each car prints the y_mean and x_mean of the car location


import cv2
import jetson.inference
import jetson.utils
import os
import csv
import xml.etree.ElementTree as ET



# Path to model and pictures
#dataPath = "/home/gustav/direc_mount/Annotations_merged/Annotations-merged-new-angle/"
dataPath = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch4/"


# create array containing all filenames 
arq = sorted(os.listdir(dataPath + "JPEGImages/")) # list of all filenames

# create a csv writer
f = open(dataPath + "carPositions.csv","w")
writer = csv.writer(f)

# write a header for the csv file
header = ['frame_id','y_avg','x_avg']
writer.writerow(header)

file = 0
while True:

	#tail is the filename (frame24_ch2.jpg)
	tail = os.path.split(arq[file])[1]
	filename = os.path.splitext(tail)[0]
	xml_file_exists = os.path.splitext(dataPath + "Annotations/" + tail + ".xml")[1] == ".xml" 

	if xml_file_exists:		
		#data = [arq[file]]

		#create XML tree
		tree = ET.parse(dataPath + "Annotations/" + filename + ".xml")
		root = tree.getroot()
		
		
		# go through all attribute called bndbox
		for name in root.iter("bndbox"):
			data = [arq[file]]
			print(dataPath + "Annotations/" + filename + ".xml")
			# holds the four bbox values in order: xmin, ymin, xmax, ymax
			positionVector = []

			# go through all values in bndbox attribute
			for child in name:
				positionVector.append(float(child.text))

			x_avg = (positionVector[0] + positionVector[2])/2
			y_avg = (positionVector[1] + positionVector[3])/2

			data.append(y_avg)
			data.append(x_avg)	
			writer.writerow(data)

	file = file + 1
	if file >= len(arq):
		break

print("EOP")
