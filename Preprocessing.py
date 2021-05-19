import sparknlp
#from org.apache.spark.ml.linal import Vector, Vectors
from sparknlp.pretrained import PretrainedPipeline
import sys, glob, os
sys.path.extend(glob.glob(os.path.join(os.path.expanduser("~"), ".ivy2/jars/*.jar")))
from sparknlp.base import *
from sparknlp.annotator import *
import re
import pandas as pd
from pandas import DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, ArrayType
from pyspark.sql.functions import regexp_extract, udf, size, max, explode
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
import array
import numpy as np
import pyspark.sql.types as T
from scipy import spatial
import pickle
from pyspark.sql.functions import monotonically_increasing_id 
spark = sparknlp.start()

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)

#a = spark._conf.get('spark.executor.memory')# Feature 1 to be extracted from the logs
#print("Executor Memory: ", a)
#b = spark._conf.get('spark.driver.memory')
#print("Driver Memory: ", b)
# For calculating the message count vector we identify all types of log messages and count the frequency of each message in the training dataset.

# Feature 2 to be extracted from the logs
# We use pretrained-BERT algorithm, which is the state-of-the-art method for Vectorization in NLP (Natural Language Processing), for vectorization of the log messages.



#pipeline = PretrainedPipeline('recognize_entities_dl', lang = 'en')

#result = pipeline.annotate('Google has announced the release of a beta version of the popular TensorFlow machine learning library.')

#081111 110702 19 INFO dfs.FSDataset: Deleting block blk_-619359440677101740 file /mnt/hadoop/dfs/data/current/subdir37/blk_-619359440677101740

#document = DocumentAssembler().setInputCol('text').setOutputCol('document')
#regex_matcher = RegexMatcher().setInputCols('text') \
    #.setStrategy("MATCH_ALL") \
    #.setOutputCol("blk_id") \
    #.setRules(ExternalResource("blk_id.txt"))
#regex_tokenizer = RegexTokenizer(inputCol='text', outputCol="blk_id", pattern="blk_-\d*") #[^0-9a-z#+_]+
#regex_tokenizer = RegexTokenizer(inputCol='text', outputCol="servers", pattern="\d+\.\d+\.\d+\.\d+")
#regex_tokenizer = RegexTokenizer(inputCol='text', outputCol="server_with_socket", pattern="\d+\.\d+\.\d+\.\d+:\d+")
#tokenizer1 = Tokenizer().setInputCols('document').setOutputCol('server_token').setTargetPattern('\d+\.\d+\.\d+\.\d+')
#tokenizer2 = Tokenizer().setInputCols('document').setOutputCol('serverSock_token').setTargetPattern('\d+\.\d+\.\d+\.\d+:\d+')
#pos = PerceptronModel.pretrained().setInputCols('document', 'token').setOutputCol('pos')

#pipeline = Pipeline().setStages([document, tokenizer1, tokenizer2])

data = spark.read.text('HDFS_1/HDFS.log')
labelDF = pd.read_csv("HDFS_1/anomaly_label.csv") 
#data.toDF("text")
blk_id = 'blk_(.|\\d)\\d+'#||\\v)'
date = '^\\d{6}'
time = '(?<=^\\d{6}\\s)(\\d{6})(?=\\s\\d{2})'
location = '(?<=^\\d{6}\\s\\d{6}\\s)(\\d*)(?=\\s\\w{4})'
log_type = '(?<=^\\d{6}\\s\\d{6}\\s)(\\d*\\s)(\\w*)(?=\\s)'
log_var = '(?<=^\\d{6}\\s\\d{6}\\s)(\\d*\\s\\w{4}\\s)(\\S+)(:)(?=\\s)'
servers = '(\\d{2}\\.\\d{3}\\.[\\d||\\.]*)([:]*)(\\d*)(?=\\s||\\v)' #\\d{2}\\.\\d{3}\\.[\\d||\\.]*(?=\\s||\\v)   #Instead of this, use re.findall to find all in udf
                                                                                                                # Maintain a dictionary and assign values for this
log_message = '(?<=:\\s)(.*)'       
#log_text = '(?<=:\\s)[a-zA-Z]+'                                                                            #Get rid  of everything other than text
                                                                                                                #Maintain a dictionary  with Similarity
#servers = r'\d+\.\d+\.\d+\.\d+'
#server_socket = r'\d+\.\d+\.\d+\.\d+:\d+'

#str = "081111 024259 26 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.15.198:50010 is added to blk_-2278777763790767778 size 67108864"
#re.findall('(\\d{2}\\.\\d{3}\\.[\\d||\\.]*)([:]*)(\\d*)(?=\\s||\\v)', str)


#Parsing the logs into the folowing parts: blk_id, timestamp, date, location, log_type, system, servers, log_message
logsDF = data.select(regexp_extract('value', blk_id, 0).alias('blk_id'),
                         regexp_extract('value', time, 0).cast('integer').alias('timestamp'),
                         regexp_extract('value', date, 0).cast('integer').alias('date'),
                         regexp_extract('value', location, 0).cast('integer').alias('location'),
                         regexp_extract('value', log_type, 2).alias('log_type'),
                         regexp_extract('value', log_var, 2).alias('system'),
                         regexp_extract('value', servers, 1).alias('server_ip'),
                         regexp_extract('value', servers, 3).alias('ports'),
                         regexp_extract('value', log_message, 0).alias('log_message'))
                         #,
                         #regexp_extract('value', servers,6).alias('servers'),
                         #regexp_extract('value', server_socket, 6).alias('server_socket'))



data.unpersist()
del(data)
dateFormat = udf(lambda z: dateToSec(z), IntegerType())
spark.udf.register("dateFormat", dateFormat)

timeFormat = udf(lambda z: timeToSec(z), IntegerType())
spark.udf.register("timeFormat", timeFormat)

blk_id_format = udf(lambda z: udf_blk_id(z), StringType())
spark.udf.register("blk_id_format", blk_id_format)

combineTime = udf(lambda x, y: totalTime(x, y), IntegerType())
spark.udf.register("combineTime", combineTime)

updateIPs = udf(lambda x, y: updateIP(x, y), ArrayType(IntegerType()))
spark.udf.register("updateIPs", updateIPs)

log_text = udf(lambda x: text_ext(x), StringType())
spark.udf.register("log_text", log_text)

blk_id =  dict()
ips = dict()      #Done
log_message = dict()   #Half done converted to text, next look for similar text
message_info = dict() #Done
log_types = dict() #Done
def text_ext(s):
    g = re.findall('[a-zA-Z\\s]+', s)
    str = ''
    for stuff in g:
        str =  str  + stuff
    return str


def dateToSec(s): #YYMDD format to secs
    t = s % 100
    s = int(s / 100)
    if (s % 100) in [1,3,5,7,8,10,12]:
        t = t + 31*(s%100)
    elif (s%100) in [4,6,9,11]:
        t = t + 31*(s%100)
    else:
        year = (int(s/100))%100
        if (year % 4) == 0:  
            t = t + 29*(s%100)
        else:  
            t = t + 28*(s%100)  
    s = int(s / 100)
    t = t + 365 * (s % 100)
    del s
    return (t*24*3600)
    
def timeToSec(s): #HHMMSS to secs
    t = s % 100
    s = int(s  / 100)
    t = t + 60*(s%100)
    s = int(s  / 100)
    t = t + 3600*(s%100)
    del s
    return t
    
def totalTime(t,d):
    f = int(t + d)
    return f

def updateIP(s, t):
    g = re.findall('(\\d{2}\\.\\d{3}\\.[\\d||\\.]*)([:]*)(\\d*)(?=\\s||\\v)', s)
    if g == []:
        return g
    else:
        a = []
        for element in g:
            if element[0] in t.keys():
                if t[element[0]] not in a:
                    a.append(t[element[0]])
            else:
                t[element[0]] = len(t.keys()) + 1
                a.append(t[element[0]])
        return a

def updatelog_type(s, t):                       #updatelog_type() and updatelog_message()  do the same functioning, kept separate to avoid confusion
    if s in t.keys():
        return t[s]
    else:
        t[s] = len(t.keys()) + 1
        return t[s]

def updatelog_message(s, t):
    if s in t.keys():
        return t[s]
    else:
        t[s] = len(t.keys()) + 1
        return t[s]

def udf_types(logtype_dict):                                        #udf_...() functions do the same functioning, kept separate to avoid confusion
    return udf(lambda l: updatelog_type(l, logtype_dict))

def udf_message_info(logmessage_info):
    return udf(lambda l: updatelog_message(l, logmessage_info))

def udf_ips(ip_dict):
    return udf(lambda l: updateIP(l, ip_dict))

def udf_blk_id(s):
    s.replace(' ','')
    return s

def udf_Labels(s):
    return udf(lambda l: updateLabels(l, s))

def updateLabels(s, t):
    if t[s] == "Normal":
        return 0
    else:
        return 1

indexformat = udf(lambda z: format_index(z), IntegerType())
spark.udf.register("indexformat", indexformat)

def format_index(s):
    return int(s)

def udf_index(s):
    return udf(lambda l: updateIndex(l, s))

def updateIndex(s,t):
    t.append(t[-1] + 1)
    print(t[-1])
    return int(t[-1])

logsDF = logsDF.withColumn("ips", udf_ips(ips)(logsDF["log_message"]))
logsDF = logsDF.withColumn("logType", udf_types(log_types)(logsDF["log_type"]))
logsDF = logsDF.withColumn("logmessageInfo", udf_message_info(message_info)(logsDF["system"]))
logsDF = logsDF.withColumn("log_text", log_text(logsDF["log_message"]))
logsDF = logsDF.withColumn( 'date',dateFormat('date'))
logsDF = logsDF.withColumn( 'timestamp',timeFormat('timestamp'))
logsDF = logsDF.withColumn( 'blk_id',blk_id_format('blk_id'))
logsDF = logsDF.withColumn( 'time',combineTime('timestamp','date'))
logsDF = logsDF.drop('timestamp', 'date', 'system', 'log_type','log_message','server_ip')
logsDF = logsDF.withColumn("index", monotonically_increasing_id())
#logsDF = logsDF.withColumn("indexa", udf_index(index)(logsDF["logType"]))
#logsDF = logsDF.withColumn("index", udf_index(index)(logsDF["indexa"]))
#logsDF = logsDF.drop('indexa')
#labelDF = labelDF.withColumn('BlockId',blk_id_format('BlockId'))
#logsDF = logsDF.withColumn( 'ips',updateIPs(logsDF['log_message'],ips))
#logsDF["time"] = logsDF["date"] + logsDF["timestamp"]

#select timestamp, date, cast(timestamp as varchar(7)) + cast(date as varchar(10)) unique_number from logsDF;
#logsDF.show(10)

#row1 = logsDF.agg({"logType": "max"}).collect()[0]
#print(row1)

#row1 = logsDF.withColumn("sizes", size("ips")).agg({"sizes": "max"}).collect()[0]
#print (row1)

label_dict =  labelDF.set_index('BlockId').T.to_dict('list')
del(labelDF)
logsDF = logsDF.withColumn("label", udf_Labels(label_dict)(logsDF["blk_id"]))
del(label_dict)
#del(index)

#logsDF.agg({'index': 'max'}).show()
#logsDF.sort(F.desc("index")).show()

trainingData = logsDF.filter(F.col("index") <10000000)
#testData = logsDF.filter(F.col("index")<10000000001)
#testData = logsDF.filter((F.col("index")>10000000000)&(F.col("index")<20000000001))
#testData = logsDF.filter((F.col("index")>20000000000)&(F.col("index")<30000000001))
#testData = logsDF.filter((F.col("index")>30000000000)&(F.col("index")<40000000001))
#testData = logsDF.filter((F.col("index")>40000000000)&(F.col("index")<50000000001))
#testData = logsDF.filter((F.col("index")>50000000000)&(F.col("index")<60000000001))
#testData = logsDF.filter((F.col("index")>60000000000)&(F.col("index")<70000000001))
#testData = logsDF.filter((F.col("index")>70000000000)&(F.col("index")<80000000001))
testData = logsDF.filter(F.col("index")>80000000000)




#testData = logsDF.filter((F.col("index") > 125390) & (F.col("index") < 245391))
#testData = logsDF.filter((F.col("index") > 245390) & (F.col("index") < 445391))
#testData = logsDF.filter((F.col("index") > 445390) & (F.col("index") < 645391))
#testData = logsDF.filter((F.col("index") > 645390) & (F.col("index") < 845391))
#testData = logsDF.filter((F.col("index") > 845390)&(F.col("index") < 1845390))
#trainingData = logsDF.filter(F.col("index") < 100)
#testData = logsDF.filter((F.col("index") >100)&(F.col("index")<200))
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

document_assembler = DocumentAssembler() \
    .setInputCol("log_text") \
    .setOutputCol("document")
    
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
    
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

lemma = LemmatizerModel.pretrained('lemma_antbnc') \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("lemma")
'''
word_embeddings = BertEmbeddings.pretrained('bert_base_cased', 'en')\
  .setInputCols(["document", "token"])\
  .setOutputCol("embeddings")
'''
word_embeddings =   WordEmbeddingsModel().pretrained()\
 .setInputCols(["document",'lemma'])\
 .setOutputCol("embeddings")\
 .setCaseSensitive(False)
'''

 
word_embeddings = BertEmbeddings\
    .pretrained('bert_base_cased', 'en') \
    .setInputCols(["document",'lemma'])\
    .setOutputCol("embeddings")\
    .setPoolingLayer(-2) # default 0
'''
 #'bert_base_cased', 'en') \
 #.setPoolingLayer(0) # default 0

embeddingsSentence = SentenceEmbeddings() \
      .setInputCols(["document", "embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

embeddings_vector = EmbeddingsFinisher() \
      .setInputCols("sentence_embeddings") \
      .setOutputCols("embedding_vectors") \
      .setCleanAnnotations(True) \
      .setOutputAsVector(True)        
      
'''
classsifierdl = ClassifierDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("category")\
  .setMaxEpochs(3)\
  .setEnableOutputLogs(True)
  #.setOutputLogsPath('logs')
'''
clf_pipeline = Pipeline(
    stages=[document_assembler,
            tokenizer,
            normalizer,
            stopwords_cleaner,
            lemma,
            word_embeddings,
            embeddingsSentence,
            embeddings_vector])
            #classsifierdl])

model = clf_pipeline.fit(trainingData)
print("\n\nDONE 1\n\n")
result = model.transform(testData)

#result = clf_pipeline.transform(testData)
print("\n\nDONE 2\n\n")
logsDF.unpersist()
del(logsDF)

trainingData.unpersist()
del(trainingData)

testData.unpersist()
del(testData)

#result.printSchema()
#result.show(100)
to_array = udf(lambda z: format_vector(z), T.ArrayType(T.FloatType()))
spark.udf.register("to_array", to_array)

def format_vector(s):
    a = []
    for elem in s[0]:
        a.append(float(elem))
    return a


firstelement=udf(lambda v:float(v[0]),T.FloatType())
#to_array =  F.udf(lambda v: list([float(x) for x in v]), T.ArrayType(T.FloatType()))
#F.udf(lambda v: v.toArray().tolist(), T.ArrayType(T.FloatType()))
result = result.withColumn('embedding_vectors', to_array('embedding_vectors'))

#nump = np.array([])
def udf_nparray(s):
    return udf(lambda l: updateArray(l, s))

def updateArray(s,t):
    a = []
    for elem in s[0]:
        a.append(float(elem))
    nump = np.stack([nump,a], axis=0)
    return a

#print(nump)
#result = result.withColumn('embedding_vectors1', udf_nparray(nump)(result["embedding_vectors"]))
#print(nump)



#result.repartition(1).write.csv("cc_out.csv", sep=',')
#result.select('embedding_vectors').write.format("csv").save('./file.csv')

#emb_list = [int(row['embeddings_vector']) for row in result.collect()]
print("\n\n\nDone 3\n\n\n")
pdf = result.toPandas()
#pdf['embedding_vectors'].astype('float')
pdf['embedding_vectors'] = pdf['embedding_vectors'].apply(lambda x: np.array(x))
#pdf.to_csv('abcd5.csv')
data = np.stack(pdf['embedding_vectors'].to_numpy())   
del(pdf)
result.unpersist()
del(result)


#clusters = dict()
#cluster_labels = []
print("\n\nStarted unpickling")

with open('dict_cl.pkl', 'rb') as f:
    clusters = pickle.load(f)


with open('list_cl3.pkl', 'rb') as f:
    cluster_labels = pickle.load(f)



print("\n\nDone Pickling\n\n")
print('Size of list: ', len(cluster_labels))
print('\n\nSize of dict: ', len(clusters))
#os.remove('dict_cl.pkl')
#os.remove('list_cl3.pkl')

def  f(item, clusters, cluster_labels):
    for k in clusters.keys():
        r = (1 - spatial.distance.cosine(item,clusters[k]))
        if  r > 0.95:
            cluster_labels.append(k)
            print(len(cluster_labels))
            print(k)
            return 0
    k = len(clusters) + 1
    clusters[k] = item
    cluster_labels.append(k)
    print(len(cluster_labels))
    print(k)
    return 0

for item in  data:
    if len(clusters) == 0:
        clusters[1] = item
        cluster_labels.append(1)
    else:
        f(item, clusters, cluster_labels)


print('\n\n\nLengthof dict:', len(clusters))

with open('dict_cl.pkl', 'wb') as f:
    pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)

with open('list_cl3.pkl', 'wb') as f:
    pickle.dump(cluster_labels, f)
















print(atr)





#np.save('abcd.npy', data)
print("\n\n\nDone 4\n\n\n")
#pdf.iloc[lambda x: np.array(x.embedding_vectors)]
print("\n\n\n")
print(pdf.head())
print("\n\n\n")
print(pdf.dtypes)
#print("\n\n\n")
#pdf['embedding_vectors'].apply(lambda x: np.array(x))
print("\n\n\n")
print(pdf['embedding_vectors'].iloc[0])
print("\n\n\n")
print(pdf.shape)
print("\n\n\n")

#data = pdf["embedding_vectors"].to_numpy()
#data =  np.array(result.select("embedding_vectors"))
#del(pdf)
#data = pdf.to_numpy()
print("\n\n\n")
print(data.shape)
print("\n\n\n")
print(data)
print("\n\n\n")

'''

def udf_similarity(embVec):                                        #udf_...() functions do the same functioning, kept separate to avoid confusion
    return udf(lambda l: updateVecList(l, embVec))
def updateVecList(s, t):
	b = 0.8                       #updatelog_type() and updatelog_message()  do the same functioning, kept separate to avoid confusion
	for elem in t:
		cosine = scipy.spatial.distance.cosine(s, elem)
    		a = 1-cosine
		if b > a:
			res = elem
'''




#logsDF = logsDF.withColumn("logType", udf_types(log_types)(logsDF["log_type"]))


thresh = 0.05
clusters = hcluster.fclusterdata(data, thresh, criterion="distance",metric='cosine')

print("\n\n\nNumber of clusters:")
print(len(set(clusters)))
print("\n\n\n")

# plotting
plt.scatter(*numpy.transpose(data), c=clusters)
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()






print("\n\nStarted unpickling")

with open('dict_cl.pkl', 'rb') as f:
    clusters = pickle.load(f)

print('\n\nSize of dict: ', len(clusters))
del(clusters)

with open("list_cl.pkl", "rb") as fp:   # Unpickling
    cluster_labels = pickle.load(fp)

print("\n\nDone Pickling\n\n")
print('Size of list: ', len(cluster_labels))


clusters_dict = dict()
for i in range(len(cluster_labels)):
    clusters_dict[i+1] = cluster_labels[i]



def udf_Clusters(s):
    return udf(lambda l: updateClusters(l, s))

def updateClusters(s, t):
    return t[s]

logsDF = logsDF.withColumn("cluster_label", udf_Clusters(cluster_labels)(logsDF["index"]))


del(cluster_labels)
logsDF.show()

