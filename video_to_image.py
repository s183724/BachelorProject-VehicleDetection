#!/usr/bin/python3
# Importing all necessary libraries
from datetime import date, timedelta
import csv
from math import floor
import cv2
import os


# import csv file
file = open("csv2.csv")
csvreader = csv.reader(file)
rows = []
for row in csvreader:
    rows.append(row)
file.close()

#assumes the year is 2021
t0 = date(2021,1,1)
fps = 25

#should be from 1 to len(rows)
# calculate the 
for i in range(350,len(rows)):
    from_excel =  timedelta(days=float(rows[i][1]) - 2)
    t_now = t0 + from_excel

    str_month= ('%02d' % t_now.month)
    str_day=('%02d' % t_now.day)

    int_hours = floor(6+from_excel.seconds/3600)
    str_hours=('%02d' % int_hours)    

    str_month2= ('%02d' % t_now.month)
    str_day2=('%02d' % t_now.day)    
    str_hours2=('%02d' % (int_hours+1))

    if (int_hours > 23):
        str_hours = '00'
        str_day = ('%02d' % (t_now.day + 1))
        str_hours2 = '00'
        str_day2 = ('%02d' % (t_now.day + 1))  
    
    
    camera_channel = "_ch4"
    filename = ("D:\\NVR" + camera_channel + "_main_2021"+ str_month + "" + str_day + "" + str_hours + "0000_2021" + str_month2 + "" + str_day2 + "" + "" + str_hours2+ "0000.mp4")
    
    print("filename: " + filename)    
        
    # Read the video from specified path
    cam = cv2.VideoCapture(filename)
    
    try:

        # creating a folder named data
        if not os.path.exists('data2'):
            os.makedirs('data2')

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')


    # reading from frame
    #print("total frames: " + str(cam.get(7)))
    framenumber = (from_excel.seconds-(int_hours-6)*3600)*25
    print("time is: " + str(framenumber/25/60))
    #print("set the frame to: " + str(framenumber))
    cam.set(1,framenumber)
    ret,frame = cam.read()

    if (ret):
        # if video is still left continue creating images
        name = './data/frame' + str(i) + camera_channel + '.jpg'
        print ('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

    else:
        print("no frame in path")
        
print("finished")

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()