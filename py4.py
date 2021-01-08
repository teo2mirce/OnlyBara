import os
import wavio
import numpy as np
import pyglet
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import datetime
import os 
import time
# import scipy
file_nameCSC = "wav.wav"
file_nameTeo = "Teo.wav"
file_nameVictor = "Victor.wav"
file_nameTeo2 = "Teo2.wav"
file_nameVictor2 = "Victor2.wav"
# FileCSC = wavio.read(file_nameCSC)
# rateCSC = FileCSC.rate
# dataCSC = FileCSC.data[:,0]

# FileTeo = wavio.read(file_nameTeo)
# rateTeo = FileTeo.rate
# dataTeo = FileTeo.data[:,0]

# FileVictor = wavio.read(file_nameVictor)
# rateVictor = FileVictor.rate
# dataVictor = FileVictor.data[:,0]

# print(rateCSC)
# print(rateTeo)
# print(rateVictor)
# assert rateCSC == rateTeo and rateTeo == rateVictor
# rate=5*rateCSC


dataTeo,rate  = librosa.load(file_nameTeo)
dataVictor,rateVictor  = librosa.load(file_nameVictor)
dataTeo2,rateTeo2  = librosa.load(file_nameTeo2)
dataVictor2,rateVictor2 = librosa.load(file_nameVictor2)
dataCSC,rateSCS = librosa.load(file_nameCSC)
print(rate,rateVictor,rateVictor2,rateTeo2,rateSCS)

# MeanTeo=librosa.feature.mfcc(dataTeo).mean(axis=1)
# MeanVictor=librosa.feature.mfcc(dataVictor).mean(axis=1)
# MeanTeo2=librosa.feature.mfcc(dataTeo2).mean(axis=1)
# MeanVictor2=librosa.feature.mfcc(dataVictor2).mean(axis=1)

rate10=rate*5

baseDir="py4"+str(int(time.time()))
os.mkdir(baseDir) 
os.mkdir(baseDir+"/0") 
os.mkdir(baseDir+"/1") 
os.mkdir(baseDir+"/2") 
os.mkdir(baseDir+"/3") 
os.mkdir(baseDir+"/4")
IX=[]

for i in range(0,int(dataCSC.shape[0]/rate10)):
	meanSec=librosa.feature.mfcc(dataCSC[i*rate10:(i+1)*rate10]).mean(axis=1)
	IX.append(meanSec)
kmeans = KMeans(n_clusters=2, random_state=0).fit(IX)
for i in range(0,int(dataCSC.shape[0]/rate10)):
	sf.write(baseDir+"/"+str(kmeans.labels_[i])+'/'+str(i)+'.wav', dataCSC[i*rate10:(i+1)*rate10], rate, 'PCM_24')
	#print(i,clf.predict([librosa.feature.mfcc(dataCSC[i*rate:(i+1)*rate]).reshape(-1,)]))
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# for i in range(0,int(channel_1.shape[0]/rate)):
# for i in range(0,200):
	# avgCh1=channel_1[i*rate:(i+1)*rate].std()
	# avgCh2=channel_2[i*rate:(i+1)*rate].std()
	# print(i,avgCh1,avgCh2)