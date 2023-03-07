import streamlit as st
import pandas as pd

@st.cache_data
def get_MetaData():
	df_meta = pd.read_csv("Data/df_meta.csv")
	return df_meta

@st.cache_data(ttl=600)
def get_GeneData():
	df_genes = load_data(st.secrets["public_gsheets_url"]) # pd.read_csv("Data/df_genes.csv")
	return df_genes

def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url)


