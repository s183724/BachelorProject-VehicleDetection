#!/usr/bin/python3
# script that calculates the 

from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os
import cv2
import csv
import xml.etree.ElementTree as ET


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
	# index:  [  0  ,   1  ,  2   ,  3   ,  4   ,   5  ]
	# data is [x_min, y_min, x_max, y_max, x_avg, y_avg]
	datasetFormatted = np.array(dataset)
	datasetFormatted = datasetFormatted[:,1:7].astype(float) # remove filename from data and format as float

	return datasetFormatted


# Path to model and pictures
dataPath3_2 = "/home/gustav/direc_mount/Annotations/annotations3_ch2/"
dataPath2_2 = "/home/gustav/direc_mount/Annotations/annotations2_ch2/"
dataPath2_4 = "/home/gustav/direc_mount/Annotations/annotations2_ch4/"
dataPath1_2 = "/home/gustav/direc_mount/Annotations/annotations1_ch2/"
dataPath1_4 = "/home/gustav/direc_mount/Annotations/annotations1_ch4/"
dataPathNewAngleCH2 = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/"
dataPathNewAngleCH4 = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch4/"


#dataset3_2 = xmlCrawl(dataPath3_2,1)
#dataset2_4 = xmlCrawl(dataPath2_4,2)
#dataset2_2 = xmlCrawl(dataPath2_2,1)
#dataset1_4 = xmlCrawl(dataPath1_4,2)
#dataset1_2 = xmlCrawl(dataPath1_2,1)
dataset = xmlCrawl(dataPathNewAngleCH2,1)
#dataset = xmlCrawl(dataPathNewAngleCH4,1)


#  gives the right coloumn
# index:  [  0  ,   1  ,  2   ,  3   ,  4   ,   5  ]
# data is [x_min, y_min, x_max, y_max, x_avg, y_avg]
datasetFormatted = np.c_[dataset[:,0],dataset[:,3]].astype(float) # load it in correctly


#initArray = np.array([[600,0],[900,600],[1500,400],[2250,600]],np.float64)
gm = GaussianMixture(n_components=4, random_state=0,covariance_type = 'full').fit(datasetFormatted)
#gm.labels_

y_pred = gm.predict(datasetFormatted)

#print(gm.cluster_centers_)

print(gm.means_)

plt.scatter(datasetFormatted[:,0],datasetFormatted[:,1],c=y_pred)
#plt.scatter(gm.cluster_centers_[0:,0],gm.cluster_centers_[:,1],c='red')
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.gca().invert_yaxis() # invert y axis to represent picture coordinates
#plt.title("GMM classifier on ch4 bottom left bbox coordinate")
plotName = "GMM-ch2"
plt.savefig("/home/gustav/direc_mount/data-visualization/" + plotName)
plt.savefig("/media/gustav/Clever/direc_mount/data-visualization/" + plotName)

