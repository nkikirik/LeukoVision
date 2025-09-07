import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
import torch
import torch.nn.functional as F
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from torchvision.models import Inception_V3_Weights, ResNet50_Weights
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import make_gradcam_heatmap, make_gradcam_heatmap_keras, get_canny_edge

# --- App Title ---
st.title("LeukoVision")
class_names = ['BAS', 'EOS', 'EBO', 'IG', 'LYT', 'MON', 'NGS', 'PLA']

@st.cache_resource
# --- Load Models ---
def load_inception():
    return torch.load('./Streamlit/inceptionv3.pth', weights_only=False,map_location=torch.device('cpu'))
def load_resnet():
    return torch.load('./Streamlit/resnet_model.pth', weights_only=False,map_location=torch.device('cpu'))
def load_vgg16():
    return load_model('./Streamlit/vgg16_model.h5')
inception_model = load_inception()
resnet_model = load_resnet()
vgg16_model = load_vgg16()

# Remove DataParallel wrapper if exists
for m in [inception_model, resnet_model]:
    if isinstance(m, torch.nn.DataParallel):
        m = m.module

models = {
    "InceptionV3": inception_model,
    "ResNet50": resnet_model,
    "VGG16": vgg16_model,
}

# --- Model Selection ---
selected_model_name = st.selectbox("Choose a model", ["None"] + list(models.keys()))
selected_model = None
if selected_model_name != "None":
    selected_model = models[selected_model_name]
    st.write(f"### You selected: {selected_model_name}")

# --- Only proceed if a model is selected ---
if selected_model:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.write("### OR:")

    # --- Gallery ---
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

    # --- Class Selection ---
    class_choice = st.selectbox("Choose a class", ["All"] + list(gallery.keys()))

    # --- Decide which images to show ---
    labels, paths = [], []
    if class_choice == "All":
        for label, imgs in gallery.items():
            for img_path in imgs:
                labels.append(label)
                paths.append(img_path)
    else:
        for img_path in gallery[class_choice]:
            labels.append(class_choice)
            paths.append(img_path)

    # --- Show images in gallery expander ---
    selected_gallery = None
    with st.expander("Show Gallery", expanded=False):
        selected_gallery = image_select(
            label=f"Choose from {class_choice} gallery",
            images=paths,
            captions=labels,
            use_container_width=False
        )

    # --- Store and retrieve selection in session state ---
    if selected_gallery is not None:
        st.session_state["selected_gallery"] = selected_gallery
    selected_gallery = st.session_state.get("selected_gallery", None)

    # --- Load selected image ---
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB").resize((299, 299))
    elif selected_gallery is not None:
        image = Image.open(selected_gallery).convert("RGB").resize((299, 299))
        st.success(f"You selected: {selected_gallery}")

    # --- Image Prediction and Display ---
    if image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Image")

            # --- Preprocess and predict ---
            if "VGG16" in selected_model_name:
                img_array = np.expand_dims(np.array(image), axis=0)
                img_tensor = preprocess_input(img_array)
                predict_acc = selected_model.predict(img_tensor, verbose=0)
                pred = np.argmax(predict_acc, axis=-1)[0]
                pred_prob = predict_acc[0, pred]
            else:
                weights = Inception_V3_Weights.DEFAULT if 'InceptionV3' in selected_model_name else ResNet50_Weights.DEFAULT
                preprocess = weights.transforms()
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = selected_model(img_tensor)
                    pred = output.argmax(dim=1).item()
                    pred_prob = F.softmax(output, dim=1)[0, pred].item()

            st.write(f"## Prediction: {class_names[pred]}")
            st.write(f"## Probability: {pred_prob*100:.2f}%")

        # --- Processed Image Display ---
        with col2:
            if 'VGG16' in selected_model_name:
                img_display = img_tensor[0][..., ::-1]
                img_display = np.clip(img_display, 0, 255) / 255.0
                st.image(img_display, caption="Processed Image")
            else:
                img_np = img_tensor.squeeze().permute(1, 2, 0).numpy().clip(0, 1)
                st.image(img_np, caption="Processed Image")

        # --- Grad-CAM ---
        generate_cam = st.button("Generate Grad-CAM")
        if generate_cam:
            if 'VGG16' in selected_model_name:
                pred_class = np.argmax(predict_acc[0])
                heatmap = make_gradcam_heatmap_keras(img_tensor, selected_model, 'block5_conv3', pred_class)
            else:
                target_layer = "Mixed_7c" if 'InceptionV3' in selected_model_name else "layer4"
                heatmap, _ = make_gradcam_heatmap(img_tensor, selected_model, target_layer_name=target_layer)

            img_np = np.array(image)
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.4 * heatmap_color + 0.6 * get_canny_edge(img_np/255.0)
            overlay = np.clip(overlay, 0, 1)
            st.image(overlay, caption="Grad-CAM Result")
        else:
            st.empty()
