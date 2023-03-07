import streamlit as st
import pandas as pd


def get_MetaData():
	df_meta = pd.read_csv("Data/df_meta.csv")
	return df_meta

def get_GeneData():
	df_genes = pd.read_csv("Data/df_genes.csv")
	return df_genes
