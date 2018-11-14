import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, 'C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Model/custom_functions')
from read_data import headerData, readAttendance

X_train = pd.read_csv('C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Model/data/sample/sample1.csv', header = None)

att = X_train.values[1:, 2:-1]
att_flat = att.flatten()

classStart  = [11, 0]   # Time in hh:mm when class starts
classEnd = [12, 0]      # Time in hh:mm when class ends
minAttendance = 10      # Minimun number of students to acknowledge that day's class
windowSize = 3          # Size of sliding window
K = 3                   # K (no of nearest neighbours) in KNN
ClassTime = 60          # Will be automated in future version

added = 0
notAdded = 0

attendanceVectors = []

for day in att.T:
    pTotal = 0
    stayTime = []
    inTime = [] 
    outTime = []
    for s in day:
        isPresent, pMarked, iT, oT = readAttendance(s, classStart)
        if(isPresent):
            pTotal += 1

        if(isPresent):
            if ((oT-iT) > 0) and iT < ClassTime:
                stayTime.append(oT-iT)
                inTime.append(iT)
                outTime.append(oT)

    if(pTotal >= minAttendance):
        population = [0] * (ClassTime + 1)
        for i, x in enumerate(inTime):
            for j, y in enumerate(population):
                if j>inTime[i]:
                    population[j] += 1

            for j, y in enumerate(population):
                if j>outTime[i]:
                    population[j] -= 1
        
        attendanceVectors.append(population)

np.savetxt("population.csv", np.asarray(attendanceVectors).T, delimiter=",")


SlidingWindowVectors = []

for idx, day in enumerate(attendanceVectors[:-(windowSize-1)]):
    newVector = []
    for newDay in attendanceVectors[idx:idx+windowSize]:
        newVector += newDay
    SlidingWindowVectors.append(newVector)

TestSWVector = SlidingWindowVectors[-1]
TrainSWVector = SlidingWindowVectors

print(np.shape(TrainSWVector))

'''
plt.plot(TestVector)
plt.ylabel('fucking shit')
plt.show()
'''

TrainSW2 = TrainSWVector[:][:(windowSize-1)*(ClassTime+1)]

knn = NearestNeighbors(n_neighbors=K+1)
knn.fit(TrainSW2)
nbrs = knn.kneighbors([TrainSW2[-1]], return_distance=False)
nbrs1d = np.asarray(nbrs).ravel()

print(nbrs1d)

predicted = np.asarray([0] * (ClassTime+1))
for i in range(K):
    predicted += TrainSWVector[nbrs1d[i]][(windowSize-1)*(ClassTime+1):]
predicted = predicted/K
actual = TrainSWVector[-1][(windowSize-1)*(ClassTime+1):]

print(len(predicted), len(actual))

mse = (np.square(actual - predicted)).mean(axis=None)
print(mse)

plt.subplot(1,3,1)
plt.plot( range(windowSize*(ClassTime+1)), TrainSW2[-1], '.-', range((windowSize-1)*(ClassTime+1), windowSize*(ClassTime+1)), predicted, '.-')
plt.title('Test Vector')
plt.ylabel('No. of people in class')

plt.subplot(1,3,2)
plt.plot(TrainSW2[nbrs1d[1]], '.-')
plt.title('NN 1')
plt.ylabel('No. of people in class')

plt.subplot(1,3,3)
plt.plot(TrainSW2[nbrs1d[2]], '.-')
plt.title('NN 2')
plt.ylabel('No. of people in class')

plt.show()


'''
plt.plot(predicted)
plt.show()
'''