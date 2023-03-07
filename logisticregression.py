
#Logistic Regression Classifier

import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache_data
def getResult(X,y):
	model = LogisticRegression(max_iter = 10000)

	# Split dataset into 80% training, 20% test
	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, log_y_test = train_test_split(X, y, test_size=0.1)

	# Fit classfier to training data
	model.fit(X_train, y_train)

	X_train.shape

	# Predict test data
	ypred_log = model.predict(X_test)
	sum=0
	predclass=[]
	ylist = log_y_test.tolist()
	for i in range(len(ylist)): #manual accuracy 
		if ypred_log[i]>=0.5:
			predclass.append(1)
		else:
			predclass.append(0)
	sum+=(ylist[i]==predclass[i])
	
	accuracy = sum/13*100
	#accuracy = accuracy_score(ylist, predclass)
	
	#for ROC
	y_pred_logistic = model.decision_function(X_test)

	return model,accuracy,ylist,predclass

	