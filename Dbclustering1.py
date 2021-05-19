import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
import matplotlib.pyplot as plt
df = pd.read_csv("FinalData2.csv")
df['label'] = df['label'].map({1: -1,0 : 1})
df.head()
print(df.dtypes)



from sklearn.cluster import DBSCAN
t1 = time.process_time()
outlier_detection = DBSCAN(eps = 4, metric="euclidean",min_samples = 2, n_jobs = -1)
t2 = time.process_time()
print("Training time is: ", (t2-t1))
t3 = time.process_time()
clusters_o = outlier_detection.fit_predict(df[['blk_id', 'logType', 'location','ips','logmessageInfo','logMesType','logCount','ports','time']])
t4 = time.process_time()
print("Prediction time is: ", (t4-t3))
for i in clusters_o:
    if i != -1:
        print("\n",i)
print('\n\nClusters:\n')
print(clusters_o)
print('\n\n')
#squarer = lambda t: if (t!=-1): t = 0

cluster_list = clusters_o.tolist()
for i in range(len(cluster_list)):
    if cluster_list[i] != -1:
        cluster_list[i] = 1
#clustersoo[clustersoo != -1] = 0
acc = accuracy_score(df['label'], cluster_list)
f1sc = f1_score(df['label'], cluster_list)
prec = precision_score(df['label'], cluster_list)
recal = recall_score(df['label'], cluster_list)




print("Accuracy Score: ", acc)
print("F1 Score: ", f1sc)
print("Precision Score: ", prec)
print("Recall Score: ", recal)
from matplotlib import cm
#cmap = cm.get_cmap("Set1")

#plt.scatter(x=num[:,0],y=num[:,-1], c=clusters, cmap=cmap,colorbar = False)

#plt.savefig('books_read.png')
#plt.close()



indx = []
acc_list = []
f1sc_list = []
prec_list = []
recal_list = []

for i in range(100,len(df['label']),900):
    acc_list.append(100*accuracy_score(df['label'][:i+1], cluster_list[:i+1]))
    f1sc_list.append(100*f1_score(df['label'][:i+1], cluster_list[:i+1]))
    prec_list.append(100*precision_score(df['label'][:i+1], cluster_list[:i+1]))
    recal_list.append(100*recall_score(df['label'][:i+1], cluster_list[:i+1]))
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
plt.savefig('DBS_metrics.png')
