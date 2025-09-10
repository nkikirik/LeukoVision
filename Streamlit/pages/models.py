import streamlit as st
from streamlit_option_menu import option_menu
import sys
from utils import white_bg
import io
from tensorflow.keras.applications import InceptionV3

st.title('Modeling')

st.markdown('LeukoVision leverages state-of-the-art convolutional neural networks (CNNs) ' \
'to classify different types of blood cells. These models—InceptionV3, ResNet50, and VGG16—have ' \
'been widely used in medical image analysis due to their ability to capture subtle patterns in ' \
'microscopy images.')

section = option_menu(
    menu_title=None,
    options=["InceptionV3", "ResNet50", "VGG16"],  
    icons=['🔬','🧬','🧪'],
    orientation="horizontal",
)

# section = st.sidebar.radio(
#     "Choose Section",
#     ["InceptionV3", "ResNet50", "VGG16"]
# )

if section == "InceptionV3":
    st.subheader("InceptionV3 🔬")
    st.markdown("""
    # InceptionV3 Overview

    InceptionV3 is a deep convolutional neural network architecture designed for efficient and accurate image recognition. It is an evolution of the original GoogLeNet (Inception) model, optimized for both computational efficiency and high performance on large-scale image classification tasks.  

    The key idea behind InceptionV3 is the use of **Inception modules**, which allow the network to capture features at multiple scales simultaneously. Each module applies several convolutions of different sizes in parallel and concatenates the results, enabling the model to learn both fine and coarse features from an image.  

    InceptionV3 incorporates several advanced techniques to improve training and reduce overfitting, including:

    - **Factorized convolutions** to reduce computational cost while maintaining performance  
    - **Auxiliary classifiers** that provide additional gradient signals during training  
    - **Batch normalization** to stabilize and accelerate training  
    - **Label smoothing** to improve generalization  

    Thanks to these innovations, InceptionV3 achieves high accuracy on benchmark datasets such as ImageNet, while keeping computational resources manageable. This makes it a popular choice for real-world applications, including medical imaging, object detection, and visual recognition tasks.
    """)

    st.image(white_bg('./Streamlit/pages/images/inceptionv3.png'), caption='Architecture diagram of InceptionV3',use_container_width=True)
    model = InceptionV3(weights='imagenet')
    with st.expander("See Full Model Summary"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.code(stream.getvalue())

elif section == "ResNet50":
    st.subheader("ResNet50 🧬")
    

elif section == "VGG16":
    st.subheader("VGG16 🧪")
