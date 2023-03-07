import streamlit as st


st.title("Project: Deep Learning Model....")
st.write("App...")

from PIL import Image
image = Image.open('images/gene.jpg')

st.image(image)

