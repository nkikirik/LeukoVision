import streamlit as st

st.title("🏠 Welcome to LeukoVision")

st.markdown("""
**LeukoVision** is an advanced, AI-powered platform designed to analyze and classify blood cell images with precision.  
Leveraging state-of-the-art deep learning models — **InceptionV3**, **ResNet50**, and **VGG16** — the app provides accurate identification of various blood cell types, including basophils, eosinophils, erythroblasts, lymphocytes, monocytes, neutrophils, and platelets.

With **LeukoVision**, you can:
- Upload blood smear images or select from a curated gallery.
- Predict the correct cell type with confidence scores for each prediction.
- Visualize **Grad-CAM heatmaps** to understand which regions of the image influenced the model’s decision.
- Compare the performance of different models for research and educational purposes.

This interactive tool is perfect for **researchers, educators, and medical professionals** looking to combine computational pathology with intuitive visualization.
""")
