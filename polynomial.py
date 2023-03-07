#polynomial + linear regression
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

@st.cache_data
def getResult(X,y):
	#PCA Analysis
	pca = PCA(n_components=32) #32 - num features to only be used
	X_pca = pca.fit_transform(X)
	X_train, X_test, y_train, poly_y_test = train_test_split(X_pca, y, test_size=0.2)


	poly = PolynomialFeatures(degree=7) 
	x_poly = poly.fit_transform(X_train)
	poly.fit(X_train,y_train)

	model = LinearRegression()
	model.fit(x_poly, y_train)

	#prediction
	ypred_poly = model.predict(poly.fit_transform(X_test))

	sum=0
	predclass=[]
	ylist_poly = poly_y_test.tolist()
	for i in range(len(ypred_poly)): #manual accuracy 
		if ypred_poly[i]>=0.5:
			predclass.append(1)
		else:
			predclass.append(0)
	
	print(ylist_poly[i],predclass[i]) 
	sum+=(ylist_poly[i]==predclass[i])
	print("accuracy calculated manually",sum/13*100)

	accuracy = accuracy_score(ylist_poly, predclass)
	print("Final Polynomial Accuracy: ", accuracy)

	return model,accuracy,ylist_poly,predclass