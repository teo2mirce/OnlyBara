import os
import wavio
import numpy as np
import pyglet

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


dataTeo2,rateTeo2  = librosa.load(file_nameTeo2)
dataTeo,rate  = librosa.load(file_nameTeo)
dataVictor,rateVictor  = librosa.load(file_nameVictor)
dataVictor2,rateVictor2 = librosa.load(file_nameVictor2)
dataCSC,rateSCS = librosa.load(file_nameCSC)
print(rate,rateVictor,rateVictor2,rateTeo2,rateSCS)

MeanTeo=librosa.feature.mfcc(dataTeo).mean(axis=1)
MeanVictor=librosa.feature.mfcc(dataVictor).mean(axis=1)
MeanTeo2=librosa.feature.mfcc(dataTeo2).mean(axis=1)
MeanVictor2=librosa.feature.mfcc(dataVictor2).mean(axis=1)

factorx=3
rate10=factorx*rate
tiem=str(int(time.time()))
for per in np.arange(0.6,0.9,0.1):
	os.mkdir("py5_Teo_"+tiem+"_"+str(round(per,2))) 
	os.mkdir("py5_Victor_"+tiem+"_"+str(round(per,2))) 
	for i in range(0,int(dataCSC.shape[0]/rate10)):
		meanSec=librosa.feature.mfcc(dataCSC[i*rate10:(i+1)*rate10]).mean(axis=1)
		strname='%05d' % i
		if (sum(abs(MeanTeo-meanSec))+sum(abs(MeanTeo2-meanSec)))*per < sum(abs(MeanVictor-meanSec))+sum(abs(MeanVictor2-meanSec)):
			q='fmm'
			#os.system('ffmpeg -i out.mp4 -ss '+str(i*factorx)+" -t "+str(factorx)+" -async 1 py5_Teo_"+tiem+"_"+str(round(per,2))+"/"+strname+".mp4")
		else:
			os.system('ffmpeg -i out.mp4 -ss '+str(i*factorx)+" -t "+str(factorx)+" -async 1 py5_Victor_"+tiem+"_"+str(round(per,2))+"/"+strname+".mp4")
# find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -f concat -i fl.txt -c copy output.mp4; rm fl.txt
# https://stackoverflow.com/questions/28922352/how-can-i-merge-all-the-videos-in-a-folder-to-make-a-single-video-file-using-ffm/37756628