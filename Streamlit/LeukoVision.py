import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
import torch
import torch.nn.functional as F
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import torchvision.models as models
from torchvision.models import Inception_V3_Weights, ResNet50_Weights
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import make_gradcam_heatmap, make_gradcam_heatmap_keras, get_canny_edge

st.title("LeukoVision")
class_names = ['BAS','EOS','EBO','IG','LYT','MON','NGS','PLA']

# --- Load Models ---
inception_model = torch.load('./Streamlit/inceptionv3.pth', map_location='cpu')
resnet_model = torch.load('./Streamlit/resnet_model.pth', map_location='cpu')
vgg16_model = load_model('./Streamlit/vgg16_model.h5')

# Remove DataParallel wrapper if exists
for m in [inception_model, resnet_model]:
    if isinstance(m, torch.nn.DataParallel):
        m = m.module

models_dict = {
    "InceptionV3": inception_model,
    "ResNet50": resnet_model,
    "VGG16": vgg16_model,
}

selected_model_name = st.selectbox("Choose a model", ["None"] + list(models_dict.keys()))
selected_model = models_dict.get(selected_model_name) if selected_model_name != "None" else None
if selected_model:
    st.write(f"### You selected: {selected_model_name}")

# --- Image Selection ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
gallery = {
    "BAS": ['./Streamlit/gallary/BA_580.jpg','./Streamlit/gallary/BA_19779.jpg','./Streamlit/gallary/BA_20201.jpg'],
    "EOS": ['./Streamlit/gallary/EO_29763.jpg','./Streamlit/gallary/EO_24568.jpg','./Streamlit/gallary/EO_25085.jpg'],
    "EBO": ['./Streamlit/gallary/ERB_168152.jpg','./Streamlit/gallary/ERB_170062.jpg','./Streamlit/gallary/ERB_174098.jpg'],
    "IG": ["./Streamlit/gallary/PMY_901117.jpg",'./Streamlit/gallary/MMY_630078.jpg','./Streamlit/gallary/MY_318125.jpg'],
    "LYT": ["./Streamlit/gallary/LY_742481.jpg",'./Streamlit/gallary/LY_731097.jpg','./Streamlit/gallary/LY_743393.jpg'],
    "MON": ["./Streamlit/gallary/MO_849518.jpg",'./Streamlit/gallary/MO_888999.jpg','./Streamlit/gallary/MO_912563.jpg'],
    "NGS": ["./Streamlit/gallary/SNE_746083.jpg",'./Streamlit/gallary/BNE_378921.jpg','./Streamlit/gallary/SNE_790562.jpg'],
    "PLA": ["./Streamlit/gallary/PLATELET_969782.jpg",'./Streamlit/gallary/PLATELET_37710.jpg','./Streamlit/gallary/PLATELET_815342.jpg']
}

class_choice = st.selectbox("Choose a class", ["All"] + list(gallery.keys()))

# Flatten or filter gallery images
paths = [img for label, imgs in gallery.items() for img in (imgs if class_choice=="All" else gallery[class_choice])]
labels = [label for label, imgs in gallery.items() for _ in (imgs if class_choice=="All" else gallery[class_choice])]

selected_gallery = None
with st.expander("Show Gallery", expanded=False):
    selected_gallery = image_select(label=f"Choose from {class_choice} gallery",
                                    images=paths, captions=labels, use_container_width=False)

# Determine final image
image_path = uploaded_file if uploaded_file else selected_gallery
image = Image.open(image_path).convert("RGB").resize((299, 299)) if image_path else None

# --- Helper Functions ---
def preprocess_image(image, model_name):
    if model_name == "VGG16":
        arr = np.expand_dims(np.array(image), axis=0)
        return preprocess_input(arr)
    else:
        weights = Inception_V3_Weights.DEFAULT if model_name=="InceptionV3" else ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        return preprocess(image).unsqueeze(0)

def predict(image_tensor, model_name, model):
    if model_name == "VGG16":
        preds = model.predict(image_tensor, verbose=0)
        pred_idx = np.argmax(preds[0])
        prob = preds[0, pred_idx]
    else:
        with torch.no_grad():
            output = model(image_tensor)
            pred_idx = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1)[0, pred_idx].item()
    return pred_idx, prob

# --- Display ---
if image:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image")
        img_tensor = preprocess_image(image, selected_model_name)
        pred, pred_prob = predict(img_tensor, selected_model_name, selected_model)
        st.write(f"## Prediction: {class_names[pred]}")
        st.write(f"## Probability: {pred_prob*100:.2f}%")
    with col2:
        img_display = np.array(image) / 255.0
        st.image(img_display, caption="Processed Image")

    # --- Grad-CAM ---
    if st.button("Generate Grad-CAM"):
        if selected_model_name == "VGG16":
            pred_class = np.argmax(selected_model.predict(img_tensor, verbose=0)[0])
            heatmap = make_gradcam_heatmap_keras(img_tensor, selected_model, 'block5_conv3', pred_class)
        else:
            with torch.no_grad():
                target_layer = "Mixed_7c" if selected_model_name=="InceptionV3" else "layer4"
                heatmap, _ = make_gradcam_heatmap(img_tensor, selected_model, target_layer)
        img_np = np.array(image)
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]
        overlay = 0.4 * heatmap_color + 0.6 * get_canny_edge(img_np/255.0)
        overlay = np.clip(overlay, 0, 1)
        st.image(overlay, caption="Grad-CAM Result")
