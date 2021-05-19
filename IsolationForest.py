
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time

pdf  = pd.read_csv('FinalData2.csv')
#pdf['label'] = pdf['label'].map({1: -1,0 : 1})
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.01),max_features=1.0)
t1 = time.process_time()
model.fit(pdf[['blk_id', 'logType', 'location','ips','logmessageInfo','logMesType','logCount','ports','time']])
t2 = time.process_time()
pdf['scores']=model.decision_function(pdf[['blk_id', 'logType', 'location','ips','logmessageInfo','logMesType','logCount','ports','time']])

print('Done 2')
t3 = time.process_time()
pdf['anomaly']=model.predict(pdf[['blk_id', 'logType', 'location','ips','logmessageInfo','logMesType','logCount','ports','time']])
t4 = time.process_time()
pdf['anomaly'] = pdf['anomaly'].map({-1: 1,1 : 0})


plt.figure(figsize=(12,8))
plt.hist(pdf['scores'])
plt.title("Histogram of avg anomaly scores where low score means more anomalous")
#plt.savefig('IsolationForest1.png')
plt.close()

print("Training time is: ", (t2-t1))
print("Prediction time is: ", (t4-t3))


acc = accuracy_score(pdf['label'], pdf['anomaly'])
f1sc = f1_score(pdf['label'], pdf['anomaly'])
prec = precision_score(pdf['label'], pdf['anomaly'])
recal = recall_score(pdf['label'], pdf['anomaly'])




print("Accuracy Score: ", acc)
print("F1 Score: ", f1sc)
print("Precision Score: ", prec)
print("Recall Score: ", recal)



print(pdf.head(20))
print(pdf.loc[pdf['anomaly']!=-1])

indx = []
acc_list = []
f1sc_list = []
prec_list = []
recal_list = []

for i in range(100,len(pdf['label']),900):
    acc_list.append(100*accuracy_score(pdf['label'][:i+1], pdf['anomaly'][:i+1]))
    f1sc_list.append(100*f1_score(pdf['label'][:i+1], pdf['anomaly'][:i+1]))
    prec_list.append(100*precision_score(pdf['label'][:i+1], pdf['anomaly'][:i+1]))
    recal_list.append(100*recall_score(pdf['label'][:i+1], pdf['anomaly'][:i+1]))
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
plt.savefig('ISF_metrics.png')
