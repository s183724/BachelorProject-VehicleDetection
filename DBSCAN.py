#!/usr/bin/python3
# script that calculates the 

from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from datetime import datetime
import os
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
dataPathNewAngleCH2 = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/"
dataPathNewAngleCH4 = "/home/gustav/direc_mount/Annotations/annotations-new-angle_ch4/"

MIN_SAMPLES = 20

## CH4
folderPath = '/home/gustav/direc_mount/Annotations/annotations-new-angle_ch4/'
plotname = "DBSCAN-ch4 "
dataset = xmlCrawl(dataPathNewAngleCH4,1)
EPS = 129 # number between 129 and 74

## CH2 
#folderPath = '/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/'
#plotname = "DBSCAN-ch2"
#dataset = xmlCrawl(dataPathNewAngleCH2,1)
#EPS = 154 #number between 75 and 154

#  gives the right coloumn
# index:  [  0  ,   1  ,  2   ,  3   ,  4   ,   5  ]
# data is [x_min, y_min, x_max, y_max, x_avg, y_avg]
datasetFormatted = np.c_[dataset[:,4],dataset[:,3]].astype(float) # load it in correctly

# for looking at detectnet boxes on top of the other plot
#filename = folderPath + "detectnet-car-positions.csv"
#detectnetOut = load_csv(filename)[1:,:].astype(float)

# run the DBSCAN
clustering = DBSCAN(eps=EPS,min_samples=MIN_SAMPLES).fit(datasetFormatted)
dataWithLabels = np.c_[datasetFormatted,clustering.labels_] # append labels to each data point
numberOfClasses = max(clustering.labels_) + 1
print("numberOfClasses: " + str(numberOfClasses))


# compute mean position for cars
carPosition_count = np.zeros(numberOfClasses)
carPosition_X = np.zeros(numberOfClasses)
carPosition_Y = np.zeros(numberOfClasses)

for i in range(len(dataWithLabels)):
	label = int(dataWithLabels[i][2])
	carPosition_count[label] = carPosition_count[label] + 1
	carPosition_X[label] = carPosition_X[label] + dataWithLabels[i][0]
	carPosition_Y[label] = carPosition_Y[label] + dataWithLabels[i][1]

for i in range(len(carPosition_count)):
	carPosition_X[i] =  carPosition_X[i]/carPosition_count[i]
	carPosition_Y[i] =  carPosition_Y[i]/carPosition_count[i]
carPositionAvg = np.c_[carPosition_X,carPosition_Y]

print(carPositionAvg)


# map label to parkingspot index:
labelToParkingSpot = np.zeros(numberOfClasses)
sortedArray = np.sort(carPositionAvg[:,1])
for i in range(len(labelToParkingSpot)):
	index = np.where(sortedArray[i] == (carPositionAvg[:,1]))[0][0]
	labelToParkingSpot[index] = i # find the index of lowest to highest value
print(labelToParkingSpot)

# test ch4
#carCenter = [ 703.15847826,  361.1876087 ]
#carCenter = [ 791.68688889,  577.66007937]
#carCenter = [ 552.76017606, 1156.36494718]
#carCenter = [ 699.21072508, 1918.97345921]

#test ch2
#carCenter = [319.84597953,  746.6347807 ]
#carCenter = [467.01316901,  926.19065141]
#carCenter = [381.65530159, 1485.84260317]
#carCenter = [647.34927215, 2253.68939873]

# test on real annotations: 
carCenter = [785.84597953,  584.6347807 ]


#carCenter[0] = carCenter[0] + 10 # offset x coordinate
carCenter = [carCenter]

datasetFormattedWithPoint = np.append(datasetFormatted,carCenter,axis=0) # append new car to data
clustering = DBSCAN(eps = EPS, min_samples = MIN_SAMPLES).fit(datasetFormattedWithPoint)

print(datasetFormattedWithPoint)

parkedCar = clustering.labels_[-1] # extract
print(parkedCar)

parkingSpot = labelToParkingSpot[int(parkedCar)]
print("Parking spot: " + str(parkingSpot))


# plotting options 
#plt.scatter(carCenter[0][0],carCenter[0][1],c='blue')
#plt.scatter(detectnetOut[:,0],detectnetOut[:,1],c='blue')
plt.scatter(datasetFormattedWithPoint[:,0],datasetFormattedWithPoint[:,1],c=clustering.labels_)
#plt.scatter(carPositionAvg[:,0],carPositionAvg[:,1],c='red')



plt.gca().invert_yaxis() # invert y axis to represent picture coordinates
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plotTitle = plotname + " EPS-" + str(EPS) + " min_samples-" + str(MIN_SAMPLES)
#plt.title(plotTitle)
#plt.legend(["Ch2","Ch4"])
plt.savefig(folderPath + plotname + ".png")
plt.savefig("/home/gustav/direc_mount/data-visualization/" + plotTitle)
plt.savefig("/media/gustav/Clever/direc_mount/data-visualization/" + plotTitle)
print("saved file to: " + str(folderPath  + plotname + ".png" ))