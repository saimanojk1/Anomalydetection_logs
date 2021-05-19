
import sys, glob, os
import re
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
import array
import numpy as np
from scipy import spatial
import seaborn as sn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from math import sqrt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time

pdf  = pd.read_csv('FinalData2.csv')
pdf['label'] = pdf['label'].map({1: -1,0 : 1})
X = np.array(pdf[['blk_id', 'logType', 'location','ips','logmessageInfo','logMesType','logCount','ports','time']].astype(float))
Y = np.array(pdf['label'])
start = time.process_time()
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X)
t1 = time.process_time()
print("Training time is: ", (t1-start))
center = kmeans.cluster_centers_
#print(center)

center1 = (center.tolist())[0]
print(center1)

#print(kmeans)
#correct = 0
#for i in range(len(X)):
  #  predict_me = np.array(X[i].astype(float))
   # predict_me = predict_me.reshape(-1, len(predict_me))
    #prediction = kmeans.predict(predict_me)
   # print(prediction)
    #if prediction[0] == Y[i]:
        #correct += 1
#
y_act = Y.tolist()
dist = []
x = X.tolist()
t2 = time.process_time()
for i in range(len(x)):
    sum = 0
    for j in range(len(x[i])):
        sum = sum +(x[i][j] - center1[j])**2
    dist.append(sqrt(sum))
    
plt.hist(dist, bins=50)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
#plt.show()
y = []
for d in dist:
    if d> 250000:
        y.append(-1)
    else:
        y.append(1)

print(y.count(1))
t3 = time.process_time()
print("Prediction time is: ", (t3-t2))
acc = accuracy_score(y_act, y)
f1sc = f1_score(y_act, y)
prec = precision_score(y_act, y)
recal = recall_score(y_act, y)

print("Accuracy Score: ", acc)
print("F1 Score: ", f1sc)
print("Precision Score: ", prec)
print("Recall Score: ", recal)

order_index = np.argsort(dist, axis = 0)
plt.scatter(dist,order_index)
#plt.show()
plt.close()
indx = []
acc_list = []
f1sc_list = []
prec_list = []
recal_list = []

for i in range(100,len(y_act),900):
    acc_list.append(100*accuracy_score(y_act[:i+1], y[:i+1]))
    f1sc_list.append(100*f1_score(y_act[:i+1], y[:i+1]))
    prec_list.append(100*precision_score(y_act[:i+1], y[:i+1]))
    recal_list.append(100*recall_score(y_act[:i+1], y[:i+1]))
    indx.append(i+1)
print(indx[0])
print(indx[1])
print(indx[2])

plt.plot(indx, acc_list)
plt.plot(indx, f1sc_list)
plt.plot(indx, prec_list)
plt.plot(indx, recal_list)
plt.xlabel('Number of records')
plt.ylabel('Metric Value')
plt.legend(["Accuracy", "F1 score","Precision","Recall"])
plt.savefig('KM_metrics.png')
