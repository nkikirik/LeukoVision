import streamlit as st
from streamlit_option_menu import option_menu
st.title('Modeling')

section = option_menu(
    menu_title=None,
    options=["InceptionV3", "ResNet50", "VGG16"],  
    icon=['🔬','🧬','🧪'],
    orientation="horizontal",
)

# section = st.sidebar.radio(
#     "Choose Section",
#     ["InceptionV3", "ResNet50", "VGG16"]
# )

if section == "InceptionV3":
    st.header("InceptionV3")
    

elif section == "ResNet50":
    st.header("ResNet50")
    

elif section == "VGG16":
    st.header("VGG16")
