import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import helper
import logisticregression
import nativebayes
import polynomial
import neural
import svm

import seaborn as sns

def main():
	
	st.title("Binary Classification Web App")
	st.sidebar.title("Binary Classification Web App")
	
	@st.cache_data
	def load_data():
		df_genes = helper.get_GeneData()
		df_meta = helper.get_MetaData()
		X = df_genes.iloc[:, 1:]
		y = df_meta['diagnosis_class']
		return X,y
	
	def plot_metrics(metrics_list):
		st.set_option('deprecation.showPyplotGlobalUse', False)
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			cm = plot_confusion_matrix(y_test, y_pred)
			fig = sns.heatmap(cm, annot=True, cmap='Reds')
			#st.pyplot(fig)
			st.pyplot()
		
		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			plot_roc_curve(model, x_test, y_test)
			st.pyplot()
			
		if 'Precision-Recall Curve' in metrics_list:
			st.subheader("Precision-Recall Curve")
			plot_precision_recall_curve(model, x_test, y_test)
			st.pyplot()
	
	X,y = load_data()
	class_names = ['Positive', 'Nagative']
	
	
	
	st.sidebar.subheader("Choose Classifier")
	classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Neural Networks","Naive bayes","Polynomial & Linear regression"))
	
	if classifier == 'Support Vector Machine (SVM)':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C')
		kernel = st.sidebar.radio("Kernel",("rbf", "linear"), key='kernel')
		gamma = st.sidebar.radio("Gamma (Kernel Coefficient", ("scale", "auto"), key = 'gamma')
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
		
		if st.sidebar.button("Classfiy", key='classify'):
			st.subheader("Support Vector Machine (SVM Results)")
			#model = SVC(C=C, kernel=kernel, gamma=gamma)
			#model.fit(x_train, y_train)
			#accuracy = model.score(x_test, y_test)
			#y_pred = model.predict(x_test)
			model,accuracy, y_test,y_pred= svm.getResult(X,y)
			st.write("Accuracy ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)
		
	if classifier == 'Logistic Regression':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C_LR')
		max_iter = st.sidebar.slider("Maxiumum number of interations", 100, 500, key='max_iter')
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
		
		if st.sidebar.button("Classfiy", key='classify'):
			st.subheader("Logistic Regression Results")
			#model = LogisticRegression(C=C, max_iter=max_iter)
			model,accuracy, y_test,y_pred= logisticregression.getResult(X,y)
			st.write("Accuracy ", accuracy)
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)
	if classifier == 'Neural Networks':
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
		
		if st.sidebar.button("Classfiy", key='classify'):
			st.subheader("Neural Network Model")
			#model = SVC(C=C, kernel=kernel, gamma=gamma)
			#model.fit(x_train, y_train)
			#accuracy = model.score(x_test, y_test)
			#y_pred = model.predict(x_test)
			model,accuracy, y_test,y_pred= neural.getResult(X,y)
			st.write("Accuracy ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)
	if classifier == 'Naive bayes':
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
		
		if st.sidebar.button("Classfiy", key='classify'):
			st.subheader("Gaussian Model Results")
			model,accuracy, y_test,y_pred= nativebayes.getResult(X,y)
			st.write("Accuracy ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)
	if classifier == 'Polynomial & Linear regression':
		metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
		
		if st.sidebar.button("Classfiy", key='classify'):
			st.subheader("Polynomial & Linear regression Results")
			model,accuracy, y_test,y_pred= polynomial.getResult(X,y)
			st.write("Accuracy ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)
if __name__ == '__main__':
	main()