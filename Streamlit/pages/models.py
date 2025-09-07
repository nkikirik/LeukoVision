import streamlit as st

st.title('Modeling')

section = st.sidebar.radio(
    "Choose Section",
    ["InceptionV3", "ResNet50", "VGG16"]
)

if section == "InceptionV3":
    st.header("InceptionV3")
    

elif section == "ResNet50":
    st.header("ResNet50")
    

elif section == "VGG16":
    st.header("VGG16")
