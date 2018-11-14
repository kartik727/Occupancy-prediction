import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats as st

import sys
sys.path.insert(0, 'C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Occupancy-prediction/custom_functions')
from read_data import headerData, readAttendance

matplotlib.style.use('ggplot')

X_train = pd.read_csv('C:/Users/Kartik Choudhary/OneDrive/Documents/Study Notes/Sem 7/ELD411 - B. Tech Project/Occupancy-prediction/data/sample/sample1.csv', header = None)

att = X_train.values[1:, 2:-1]
att_flat = att.flatten()

stayTime = []
inTime = []
outTime = []

added = 0
notAdded = 0

for s in att_flat:
    
    isPresent, pm, iT, oT = readAttendance(s, [11, 0])

    if(isPresent):
        if ((oT-iT) > 0) and iT < 60:
            stayTime.append(oT-iT)
            inTime.append(iT)
            outTime.append(oT)

size = 1000

y, x = np.histogram(outTime, bins=60, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0

dist = st.chi2
params = dist.fit(y)

arg = params[:-2]
loc = params[-2]
scale = params[-1]

pdf_fitted = dist.pdf(x, loc=loc, scale=scale, *arg)

start= 0
end = 60

yy = pd.Series(outTime)
yy.plot(kind='hist', bins=60, density=True, alpha=0.5)

plt.plot(pdf_fitted)
plt.xlim(0,60)

plt.show()