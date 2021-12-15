#! /usr/bin/python3

import csv


# create a csv writer
dataPath = "/jetson-inference/mount/data/"
f = open(dataPath + "/csv.csv","w")
writer = csv.writer(f)

# write a row 
header = ['frame_id','number_of_cars']
data = ['100','1']

writer.writerow(header)

writer.writerow(data)

f.close
