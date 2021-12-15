import csv



# import csv file
readFile = open("CarsInFrame.csv")
writeFile = open("CarsInFrameOutput.csv","w",newline='') 
csvreader = csv.reader(readFile)
csvWriter = csv.writer(writeFile)
rows = []
for row in csvreader:
    rows.append(row)
csvWriter.writerow(["TestTime","cumCars"])

STARTTIME = 44413.02083 #when the comparison should start running from (05/08/2021 - 00:30:00)

print("starting write")

for p in range(1,len(rows)):
    testTime = STARTTIME + p/48 # update test time with 30 min 
    
    cumCars = 0
    for q in range(1,len(rows)):
        startTime = float(rows[q][1])
        stopTime = float(rows[q][2])
        if(startTime < testTime < stopTime ):
            cumCars = cumCars + 1
    csvWriter.writerow([testTime,cumCars])


print("done")
writeFile.close()
readFile.close() 