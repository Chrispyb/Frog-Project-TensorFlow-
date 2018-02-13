import numpy as np
import scipy.io.wavfile
from scipy.fftpack import fft, ifft, rfft
from sklearn.preprocessing import normalize
import wave
import math
import random

fftWindowSize = 512
fftResolution = 256
numWavFiles = 1

wavFileLocations = []

trainingFeatures = []
testFeatures = []

trainingLabels = []
testLabels = []

labels = []

#Append wav files to wavFiles array as well as associated target data for those files.
wavFileLocations.append("C:/Users/Chris/Desktop/Frog Calls/Edited Files/BullFrogs/Bull1.wav")
labels.append([0, 1, 0, 0, 0, 0, 0])

#Primary loop for pre-processing.
for i in range(0, numWavFiles):

	#Read the data from wav files array.
	rate, data = scipy.io.wavfile.read("C:/Users/Chris/Desktop/ATP Grant/Research Project/samples/spring3test.wav")
	
	#only get the first channel.
	data0Raw = data[:, 0]

	#calculate the number of windows in the file.
	numWindowsInFile = math.floor(len(data0Raw) / fftWindowSize + 1)
	
	#calculate the amount of data to truncate.
	differenceInSize = numWindowsInFile - len(data0Raw)

	#reshape the data into correct size.
	data0RawReshaped = np.reshape(np.resize(data0Raw, numWindowsInFile * fftWindowSize), [numWindowsInFile, fftWindowSize] )
	
	
	normFFTData0 = []

	#realFFT the data, put it into an array, 80 percent chance to add it to training set, 20 percent for test set.
	for j in range(0, numWindowsInFile):
		realFFTData0 = rfft(data0RawReshaped[j], fftResolution)
		normFFTData0.append(realFFTData0 / np.linalg.norm(realFFTData0))
		randomNumber = random.randint(1, 100)
		
		if(randomNumber < 80):
			trainingFeatures.append(normFFTData0)
			trainingLabels.append(labels[i])
		else:
			testFeatures.append(normFFTData0)
			testLabels.append(labels[i])
	