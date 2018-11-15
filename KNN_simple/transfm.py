import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, 'C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Occupancy-prediction/custom_functions')
from read_data import headerData, readAttendance

distanceMetrics = ['euclidean', 'manhattan', 'minkowski', 'hamming']

X_train = pd.read_csv('C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Occupancy-prediction/data/cvl100.csv', header = None)

att = X_train.values[1:, 2:-1]
att_flat = att.flatten()

classStart  = [8, 0]                        # Time in hh:mm when class starts
classEnd = [9, 0]                           # Time in hh:mm when class ends
minAttendance = 10                          # Minimun number of students to acknowledge that day's class
windowSize = 2                              # Size of sliding window
K = 3                                       # K (no of nearest neighbours) in KNN
DistanceMetric = distanceMetrics[0]         # Distnace metric used in KNN
ClassTime = 60                              # Will be automated in future version

testIdx = 7

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

#np.savetxt("population.csv", np.asarray(attendanceVectors).T, delimiter=",")

def getScores (windowSize, K, DistanceMetric, testIdx):
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

    knn = NearestNeighbors(n_neighbors=K+1, metric=DistanceMetric)
    knn.fit(TrainSW2)
    dists, nbrs = knn.kneighbors([TrainSW2[testIdx]], return_distance=True)
    dists1d = np.asarray(dists).ravel()
    nbrs1d = np.asarray(nbrs).ravel()
    wtT = np.sum(dists1d)

    print('Closest nieghbours to', nbrs1d[0], 'are:', nbrs1d[1:])
    print(DistanceMetric, 'distances are:', dists1d[1:])

    wtd = True
    predicted = np.asarray([0.0] * (ClassTime+1))
    for i in range(K):
        nwa = np.asarray(TrainSWVector[nbrs1d[i]][(windowSize-1)*(ClassTime+1):])

        if(wtd):
            nwa = nwa*((wtT-dists1d[i])/wtT)*(K/(K-1))
        
        predicted += nwa
    predicted = predicted/K
    predicted = predicted.astype(int)
    actual = TrainSWVector[testIdx][(windowSize-1)*(ClassTime+1):]

    #print(len(predicted), len(actual))

    mse = (np.square(actual - predicted)).mean(axis=None)
    r2s = r2_score(actual, predicted)
    print('MSE for this prediction:', mse)
    print('R2 score:', r2s*100, '%')

    return mse, r2s, nbrs1d, predicted,TrainSW2

'''
mse_scores = []
r2_scores = [] 
for KK in range(1, 8):
    m, r = getScores(windowSize, KK, distanceMetrics[0], testIdx)
    mse_scores.append(m)
    r2_scores.append(r)
    print(KK, 'done')

plt.plot(range(1, 8), r2_scores)
plt.show()
'''
mse, r2s, nbrs1d, predicted, TrainSW2 = getScores(windowSize, K, DistanceMetric,testIdx)



plt.subplot(1,3,1)
plt.plot( range(windowSize*(ClassTime+1)), TrainSW2[testIdx], '.-', range((windowSize-1)*(ClassTime+1), windowSize*(ClassTime+1)), predicted, '.-')
plt.title('Test Vector')
plt.ylabel('No. of people in class')
plt.ylim(0, 350)

plt.subplot(1,3,2)
plt.plot(TrainSW2[nbrs1d[1]], '.-')
plt.title('NN 1')
plt.ylabel('No. of people in class')
plt.ylim(0, 350)

plt.subplot(1,3,3)
plt.plot(TrainSW2[nbrs1d[2]], '.-')
plt.title('NN 2')
plt.ylabel('No. of people in class')
plt.ylim(0, 350)

plt.show()


'''
plt.plot(range(122), SlidingWindowVectors[13][:122], '.-', range(122, 183), SlidingWindowVectors[13][122:], '--')
plt.title('Vector with sliding window size 3')
plt.ylabel('No. of students')
plt.xlabel('Time (mins)')
plt.legend(['Incomplete vetor', 'prediction using KNN'])
plt.show()
'''