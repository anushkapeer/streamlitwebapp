import pandas as pd
import streamlit as st

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

@st.cache_data
def getAccuracy(X,y):

	clf = GaussianNB()

	#PCA Analysis
	pca = PCA(n_components=32) #32 - num features to only be used
	X_pca = pca.fit_transform(X)
	X_train, X_test, y_train, naive_y_test = train_test_split(X_pca, y, test_size=0.2)

	# Fit classfier to training data
	clf.fit(X_train, y_train)

	X_train.shape

	#prediction
	ypred_naive = clf.predict(X_test)
	sum=0
	predclass=[]
	ylist = naive_y_test.tolist()
	for i in range(len(ylist)): #manual accuracy 
		if ypred_naive[i]>=0.5:
			predclass.append(1)
		else:
			predclass.append(0)
	
	print(ylist[i],predclass[i]) 
	sum+=(ylist[i]==predclass[i])
	print("accuracy calculated manually",sum/13*100)
	accuracy = accuracy_score(ylist, predclass)
	print("Final Native Bayes Accuracy: ", accuracy)
	return clf,accuracy,ylist,predclass