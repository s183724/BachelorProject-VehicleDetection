#!/usr/bin/python3
# runs through a folder of XML files and for each car prints the y_mean and x_mean of the car location


import cv2
import numpy as np
import os
import csv
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt



# crawls xml tree and outputs the box data in a numpy array
def xmlCrawl(datapath,x): 
	# create array containing all filenames 
	arq = sorted(os.listdir(datapath + "JPEGImages/")) # list of all filenames

	# create numpy array to be filled with the data
	dataset = list()
	file = 0
	while True:

		#tail is the filename (frame24_ch2.jpg)
		tail = os.path.split(arq[file])[1]
		filename = os.path.splitext(tail)[0]
		xml_file_exists = os.path.splitext(datapath + "Annotations/" + tail + ".xml")[1] == ".xml" 

		if xml_file_exists:		
			#data = [arq[file]]

			#create XML tree
			tree = ET.parse(datapath + "Annotations/" + filename + ".xml")
			root = tree.getroot()
			
			# go through all attribute called bndbox
			for name in root.iter("bndbox"):
				data = [arq[file]]
				#print(datapath + "Annotations/" + filename + ".xml")
				# holds the four bbox values in order: xmin, ymin, xmax, ymax
				positionVector = []

				# go through all values in bndbox attribute
				for child in name:
					positionVector.append(float(child.text))

				x_min = positionVector[0]
				x_max = positionVector[2]
				y_min = positionVector[1]
				y_max = positionVector[3]
				x_avg = (x_min + x_max)/2
				y_avg = ( y_min + y_max)/2

				data.append(x_min)
				data.append(y_min)
				data.append(x_max)
				data.append(y_max)
				data.append(x_avg)
				data.append(y_avg)	
				dataset.append(data)

		file = file + 1
		if file >= len(arq):
			break
	
	datasetFormatted = np.array(dataset)
	datasetFormatted = datasetFormatted[:,1:7].astype(float) # remove filename from data and format as float

	# index:  [  0  ,   1  ,  2   ,  3   ,  4   ,   5  ]
	# data is [x_min, y_min, x_max, y_max, x_avg, y_avg]
	#plt.scatter(datasetFormatted[:,4],datasetFormatted[:,5]) # plot box center 	
	if(x == 1):
		color = 'tab:blue'
	if(x == 2):
		color = 'tab:orange'
	plt.scatter(datasetFormatted[:,0],datasetFormatted[:,3],color = color) # plot box bottom left 
	#plt.scatter(np.ones((1,len(datasetFormatted[:,0])))*x,datasetFormatted[:,3],color = color) # plot only y-axis 
	return datasetFormatted

#plt.rcParams["figure.figsize"] = (3.5,5)

# Path to model and pictures
dataPath3_2 = "/home/gustav/direc_mount/Annotations/annotations3_ch2/"
dataPath2_2 = "/home/gustav/direc_mount/Annotations/annotations2_ch2/"
dataPath2_4 = "/home/gustav/direc_mount/Annotations/annotations2_ch4/"
dataPath1_2 = "/home/gustav/direc_mount/Annotations/annotations1_ch2/"
dataPath1_4 = "/home/gustav/direc_mount/Annotations/annotations1_ch4/"
dataPathNewAngleCH2 = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/"
dataPathNewAngleCH4 = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch4/"

#datasetNewAngle_2 = xmlCrawl(dataPathNewAngleCH2,1)
#datasetNewAngle_4 = xmlCrawl(dataPathNewAngleCH4,2)
#dataset3_2 = xmlCrawl(dataPath3_2,1)
dataset2_4 = xmlCrawl(dataPath2_4,1)
#dataset2_2 = xmlCrawl(dataPath2_2,1)
dataset1_4 = xmlCrawl(dataPath1_4,1)
#dataset1_2 = xmlCrawl(dataPath1_2,1)



## plotting the data 

#plt.scatter(np.ones((1,len(dataset2_2[:,5]))),dataset2_2[:,3],color = 'red')
#plt.scatter(np.ones((1,len(dataset2_4[:,5]))),dataset2_4[:,3],color = 'blue')

plt.gca().invert_yaxis() # invert y axis to represent picture coordinate
#plt.xticks([1,2]) #disable x-ticks for 1D-plot

plt.xlabel("Pixels")
plt.ylabel("Pixels")
#plt.legend(["Ch2","Ch4"])
#plt.title("bottom bbox coordinate distribution \n for old angle images")
#plt.legend(["cam1","cam2"],bbox_to_anchor= (0.75,1.15))

plt.savefig("/home/gustav/direc_mount/data-visualization/old-angle-2D-ch4")
plt.savefig("/media/gustav/Clever/direc_mount/data-visualization/old-angle-2D-ch4")
#plt.show()



