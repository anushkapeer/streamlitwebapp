#svm classifier 
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

@st.cache_data
def getResult(X,y):
	#PCA Analysis
	pca = PCA(n_components=32) #32 - num features to only be used
	X_pca = pca.fit_transform(X)
	X_train, X_test, y_train, svm_y_test = train_test_split(X_pca, y, test_size=0.2)
	
	model = SVC(gamma = 'auto')
	model.fit(X_train, y_train)

	#prediction
	ypred_svc = model.predict(X_test)
	sum=0
	predclass=[]
	ylist = svm_y_test.tolist()
	for i in range(len(ylist)): #manual accuracy 
		if ypred_svc[i]>=0.5:
			predclass.append(1)
		else:
			predclass.append(0)
	
	print(ylist[i],predclass[i]) 
	sum+=(ylist[i]==predclass[i])
	print("accuracy calculated manually",sum/13*100)
	
	accuracy = accuracy_score(ylist, predclass)
	print("Final SVM Accuracy: ", accuracy)
	
	return model,accuracy,ylist,predclass