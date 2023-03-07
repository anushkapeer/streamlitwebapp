#neural networks NLP
import streamlit as st

import pandas as pd
import tensorflow
from tensorflow import keras
import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Flatten, Dropout
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def getAccuracy(X,y):
	pca = PCA(n_components=32) #32 - num features to only be used
	X_pca = pca.fit_transform(X)
	xtrain, xtest, ytrain, ytest_log = train_test_split(X_pca, y, test_size=0.2, random_state = 42)

	model = Sequential()
	model.add(Input(shape=(32,)))
	#hidden layers
	model.add(Dense(16, activation='relu')) #rectified linear unit, if weighted sum >= 0, output will be the same, neg value will be 0
	#sufficient for desired accuracy

	model.add(Dropout(0.5))
	model.add(Dense(8, activation='relu')) #8 is next hidden layer
	model.add(Dense(4, activation='relu'))
	model.add(Dense(1, activation='sigmoid')) #diff activation function, bc we need a y/n answer
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy']) #binary croos entropy for binary classification
	#adam optimization is one of the best

	hist=model.fit(xtrain, ytrain,epochs=5000,verbose=0) #num of times running the samples
	#wrong accuracy = backtracking weights
	ypred = model.predict(xtest)
	sum=0
	predclass=[]
	ylist = ytest_log.tolist()
	for i in range(len(ylist)): #manual accuracy 
		if ypred[i]>=0.5:
			predclass.append(1)
		else:
			predclass.append(0)
	
	print(ylist[i],predclass[i]) 
	sum+=(ylist[i]==predclass[i])
	print("accuracy calculated manually",sum/13*100)
	print(min(hist.history['Accuracy']))
	print(max(hist.history['Accuracy']))
	
	accuracy = accuracy_score(ylist, predclass)
	
	print("Final NLP Accuracy: ", accuracy)
	return model,accuracy,ylist,predclass