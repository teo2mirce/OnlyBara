import os
import wavio
import numpy
import pyglet

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import datetime
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


dataCSC,rate  = librosa.load(file_nameCSC)
dataTeo  = librosa.load(file_nameTeo)[0]
dataVictor  = librosa.load(file_nameVictor)[0]
dataTeo2  = librosa.load(file_nameTeo2)[0]
dataVictor2  = librosa.load(file_nameVictor2)[0]

# rate=22000

X=[]
y=[]
X2=[]
y2=[]

for i in range(0,int(dataVictor.shape[0]/rate)):
	X.append(librosa.feature.mfcc(dataVictor[i*rate:(i+1)*rate]).reshape(-1,))
	y.append(0)
for i in range(0,int(dataTeo.shape[0]/rate)):
	X.append(librosa.feature.mfcc(dataTeo[i*rate:(i+1)*rate]).reshape(-1,))
	y.append(1)
	
for i in range(0,int(dataVictor2.shape[0]/rate)):
	X2.append(librosa.feature.mfcc(dataVictor2[i*rate:(i+1)*rate]).reshape(-1,))
	y2.append(0)
for i in range(0,int(dataTeo2.shape[0]/rate)):
	X2.append(librosa.feature.mfcc(dataTeo2[i*rate:(i+1)*rate]).reshape(-1,))
	y2.append(1)

#https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
#https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
	
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
X_train=X
X_test=X2
y_train=y
y_test=y2

clf = MLPClassifier(random_state=1, max_iter=300,verbose=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
clf = MLPClassifier(random_state=1, max_iter=300,verbose=1).fit(X_train+X_test, y_train+y_test)


for i in range(0,int(dataCSC.shape[0]/rate)):
	pred= clf.predict_proba([librosa.feature.mfcc(dataCSC[i*rate:(i+1)*rate]).reshape(-1,)])[0][0]
	if(pred > 0.75):#victor
		sf.write('Victor/'+str(i)+'.wav', dataCSC[i*rate:(i+1)*rate], rate, 'PCM_24')
		# librosa.output.write_wav('Victor/'+str(i)+'.wav', dataCSC[i*rate:(i+1)*rate])
	if(pred < 0.25):#Teo
		sf.write('Teo/'+str(i)+'.wav', dataCSC[i*rate:(i+1)*rate], rate, 'PCM_24')
		# librosa.output.write_wav('Teo/'+str(i)+'.wav', dataCSC[i*rate:(i+1)*rate])
	if(pred >= 0.25 and pred <=0.75):#victor
		sf.write('Nesigur/'+str(i)+'.wav', dataCSC[i*rate:(i+1)*rate], rate, 'PCM_24')
		# librosa.output.write_wav('Nesigur/'+str(i)+'.wav', dataCSC[i*rate:(i+1)*rate])
	
	
	#print(i,clf.predict([librosa.feature.mfcc(dataCSC[i*rate:(i+1)*rate]).reshape(-1,)]))
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# for i in range(0,int(channel_1.shape[0]/rate)):
# for i in range(0,200):
	# avgCh1=channel_1[i*rate:(i+1)*rate].std()
	# avgCh2=channel_2[i*rate:(i+1)*rate].std()
	# print(i,avgCh1,avgCh2)