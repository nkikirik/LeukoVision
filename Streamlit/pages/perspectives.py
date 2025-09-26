import streamlit as st

st.title("Perspectives")
st.markdown(""" <div style="text-align: justify;">

Our models—InceptionV3, ResNet50, and VGG16—demonstrate high accuracy (>98%) in classifying white blood cells (WBCs). Grad-CAM visualizations confirm that the models focus on relevant cell regions, enhancing interpretability. While all models perform comparably, practical considerations like inference speed and computational cost will guide future deployment.

Currently, the dataset contains ~18,000 images across 8 WBC classes. To improve generalization and robustness, we plan to expand the dataset to over a million images, incorporating variations in staining, resolution, and rare cell types. This will allow finer-grained classification and better disease detection.

Another limitation is that the models require single, centered cells per image. We aim to address this by training on multi-cell images and implementing a “Count Lab” feature using YOLO, enabling automated detection, classification, and counting of multiple cells in one image.

Finally, we plan to enhance interpretability through quantitative analysis of Grad-CAM activations and to deploy the models in a scalable web service, integrating with laboratory systems for real-world use. Our platform aims to streamline WBC classification, reduce human error, and support faster, more accurate medical insights for both researchers and clinicians.
</div>
""", unsafe_allow_html=True)