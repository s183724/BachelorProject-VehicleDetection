#!/usr/bin/python3
# script that calculates the 

from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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

# main 
filename = '/home/gustav/direc_mount/Annotations/annotations-new-angle_ch2/carPositions.csv'
dataset = load_csv(filename)

# skips the header and gives the right coloumn
X = dataset[1:,1] 
Y = dataset[1:,2] 
datasetFormatted = dataset[1:,1:3]

# converts the dataset to float 
datasetFormatted = datasetFormatted.astype(float)

initArray = np.array([[600,0],[600,900],[400,1500],[600,2250]],np.float64)
kmeans = KMeans(n_clusters=4, init=initArray).fit(datasetFormatted)
#kmeans.labels_

y_pred = kmeans.predict(datasetFormatted)

print(kmeans.cluster_centers_)

print(y_pred)

plt.scatter(datasetFormatted[:,0],datasetFormatted[:,1],c=y_pred)
plt.scatter(kmeans.cluster_centers_[0:,0],kmeans.cluster_centers_[:,1],c='red')

plt.gca().invert_yaxis() # invert y axis to represent picture coordinates
plt.show()
