import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
import torch
from utils import make_gradcam_heatmap,get_canny_edge
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt


st.title("LeukoVision")
class_names=['BAS','EOS','EBO','IG','LYT','MON','NGS','PLA']

inception_model = torch.load('./Streamlit/inceptionv3.pth',weights_only=False,map_location=torch.device('cpu'))
if isinstance(inception_model, torch.nn.DataParallel):
    inception_model = inception_model.module

resnet_model=torch.load('./Streamlit/resnet_model.pth',weights_only=False,map_location=torch.device('cpu'))
if isinstance(resnet_model, torch.nn.DataParallel):
    resnet_model = resnet_model.module

models = {
    "InceptionV3": inception_model,
    "ResNet50": resnet_model,
    # "Keras Model": keras_model,
}

selected_model_name = st.selectbox("Choose a model", list(models.keys()))
selected_model = models[selected_model_name]

st.write(f"### You selected: {selected_model_name}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

st.write(f"### OR:")

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

# --- First choose class ---
class_choice = st.selectbox(
    "Choose a class",
    ["All"] + list(gallery.keys())
)

# --- Decide which images to show ---
labels, paths = [], []
if class_choice == "All":
    # flatten all classes
    for label, imgs in gallery.items():
        for img_path in imgs:
            labels.append(label)
            paths.append(img_path)
else:
    # only selected class
    for img_path in gallery[class_choice]:
        labels.append(class_choice)
        paths.append(img_path)

# --- Show images in expander ---
selected_gallery = None
with st.expander("Show Gallery", expanded=False):
    selected_gallery = image_select(
        label=f"Choose from {class_choice} gallery",
        images=paths,
        captions=labels,
        use_container_width=False
    )

# --- Store selection in session state ---
if selected_gallery is not None:
    st.session_state["selected_gallery"] = selected_gallery

# --- Retrieve selected image ---
selected_gallery = st.session_state.get("selected_gallery", None)
if selected_gallery:
    st.success(f"You selected: {selected_gallery}")
    image = Image.open(selected_gallery).convert("RGB").resize((299, 299))

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((299,299))
    #st.image(image, caption="Chosen Image", use_container_width=True)
elif selected_gallery is not None:
    image = Image.open(selected_gallery).convert("RGB").resize((299,299))
    
if image:
    # Put button above the columns
    generate_cam = st.button("Generate Grad-CAM")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image")
        if 'InceptionV3' in selected_model_name or 'ResNet50' in selected_model_name:
            weights = Inception_V3_Weights.DEFAULT
            preprocess = weights.transforms()
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = selected_model(img_tensor)
                pred = output.argmax(dim=1).item()
                probs = F.softmax(output, dim=1)
                pred_prob = probs[0, pred].item()

        st.write(f"## Prediction: {class_names[pred]}")
        st.write(f"## Probability: {pred_prob*100:.2f}%")

    with col2:
        if generate_cam:
            if 'InceptionV3' in selected_model_name:
                heatmap,pred_label=make_gradcam_heatmap(img_tensor, selected_model, target_layer_name="Mixed_7c")
                #heatmap = show_importance_inception(selected_model, img_tensor, target=pred, device='cpu')
            else:
                heatmap,pred_label=make_gradcam_heatmap(img_tensor, selected_model, target_layer_name="layer4")
                #heatmap = show_importance_resnet(selected_model, img_tensor, target=pred, device='cpu')

            img_np = np.array(image)

            # Resize heatmap to match image
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]

            if img_np.max() > 1:
                img_np = img_np / 255.0

            overlay = 0.4 * heatmap_color + 0.6 * get_canny_edge(img_np)
            overlay = np.clip(overlay, 0, 1)

            st.image(overlay, caption="Grad-CAM Result")
            #st.empty()
        else:
            st.empty()  # keep alignment if button not pressed