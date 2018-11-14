import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Model/custom_functions')
#sys.path.insert(1, 'C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Model/data/sample')

from read_data import headerData, readAttendance

""" 
import csv

with open('sample.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        line_count += 1
        if (line_count == 1):
            print(f'{row[2]}')
        if line_count == 11:
            print(f'...')
            
    print(f'---')
    print(f'Processed {line_count} lines.')
 """



X_train = pd.read_csv('C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Model/data/sample/sample1.csv', header = None)



att = X_train.values[1:, 2:-1]
att_flat = att.flatten()

stayTime = []
inTime = []
outTime = []

added = 0
notAdded = 0

for s in att_flat:
    
    isPresent, iT, oT = readAttendance(s, [11, 0])
    #print(s)
    if(isPresent):
        if ((oT-iT) > 0) and iT < 60:
            stayTime.append(oT-iT)
            inTime.append(iT)
            outTime.append(oT)
        #else:
            #print(s)
        added += 1
    else:
        notAdded += 1

print('Added: ', added, 'Not added: ', notAdded)

print(len(stayTime))

#plt.hist(z, bins=60)
#plt.xlim(0,60)
#plt.title("Out Time")

#plt.show()


#np.random.seed(19680801)

# example data
#mu = 100  # mean of distribution
#sigma = 15  # standard deviation of distribution
x = stayTime
y = inTime
z = outTime

num_bins = 60

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(z, num_bins)

# add a 'best fit' line
plt.xlim(0,60)

ax.set_xlabel('Time (mins)')
ax.set_ylabel('Number of attendances')
ax.set_title(r'Time between entrance and exit')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()