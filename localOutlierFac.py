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
from sklearn.neighbors import LocalOutlierFactor


df = pd.read_csv("FinalData2.csv")
df['label'] = df['label'].map({1: -1,0 : 1})
t1 = time.process_time()
lof = LocalOutlierFactor(n_neighbors = 20)
t2 = time.process_time()
t3 = time.process_time()
df['lof'] = lof.fit_predict(df[['blk_id', 'logType', 'location','ips','logmessageInfo','logMesType','logCount','ports','time']])
t4 = time.process_time()
df['negative_outlier_factor'] = lof.negative_outlier_factor_

print(df.head(20))

print("Training time is: ", (t2-t1))
print("Prediction time is: ", (t4-t3))


acc = accuracy_score(df['label'], df['lof'])
f1sc = f1_score(df['label'], df['lof'])
prec = precision_score(df['label'], df['lof'])
recal = recall_score(df['label'], df['lof'])




print("Accuracy Score: ", acc)
print("F1 Score: ", f1sc)
print("Precision Score: ", prec)
print("Recall Score: ", recal)


indx = []
acc_list = []
f1sc_list = []
prec_list = []
recal_list = []

for i in range(100,len(df['label']),900):
    acc_list.append(100*accuracy_score(df['label'][:i+1], df['lof'][:i+1]))
    f1sc_list.append(100*f1_score(df['label'][:i+1], df['lof'][:i+1]))
    prec_list.append(100*precision_score(df['label'][:i+1], df['lof'][:i+1]))
    recal_list.append(100*recall_score(df['label'][:i+1], df['lof'][:i+1]))
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
plt.savefig('LOC_metrics.png')
