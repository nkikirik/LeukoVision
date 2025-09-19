import streamlit as st
from streamlit_option_menu import option_menu
import sys
from utils import white_bg
import io
import tensorflow as tf
from keras.applications import InceptionV3 # type: ignore
from keras.applications import VGG16 # type: ignore
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, Rescaling, Resizing
from keras.models import Model    
import plotly.graph_objects as go
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


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
    st.markdown(""" ### Overview
    <div style="text-align: justify;">
                
    InceptionV3 is a deep convolutional neural network architecture designed for efficient and accurate image recognition. It is an evolution of the original GoogLeNet (Inception) model, optimized for both computational efficiency and high performance on large-scale image classification tasks.  

    The key idea behind InceptionV3 is the use of **Inception modules**, which allow the network to capture features at multiple scales simultaneously. Each module applies several convolutions of different sizes in parallel and concatenates the results, enabling the model to learn both fine and coarse features from an image.  

    InceptionV3 incorporates several advanced techniques to improve training and reduce overfitting, including:

    - **Factorized convolutions** to reduce computational cost while maintaining performance  
    - **Auxiliary classifiers** that provide additional gradient signals during training  
    - **Batch normalization** to stabilize and accelerate training  
    - **Label smoothing** to improve generalization  

    Thanks to these innovations, InceptionV3 achieves high accuracy on benchmark datasets such as ImageNet, while keeping computational resources manageable. This makes it a popular choice for real-world applications, including medical imaging, object detection, and visual recognition tasks.
                
    </div>
    """,unsafe_allow_html=True)

    st.image(white_bg('./pages/images/inceptionv3.png'), caption='Architecture diagram of InceptionV3',use_container_width=True)
    model = InceptionV3(weights=None)
    with st.expander("See Full Model Summary"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.code(stream.getvalue())
    st.markdown('### Performace')

elif section == "ResNet50":
    st.subheader("ResNet50 🧬")
    

elif section == "VGG16":
    st.subheader("VGG16 🧪")

    st.markdown(""" ### Overview
    <div style="text-align: justify;">
                
    VGG16 is a deep convolutional neural network introduced by the Visual Geometry Group at Oxford in 2014 (Simonyan et al, 2014),
    which gained recognition after performing strongly in the ImageNet Challenge. 
                
    The “16” refers to its 16 weight layers (13 convolutional and 3 fully connected), all built using small 3×3 filters and 2×2 max pooling. By default, its architecture 
    takes a 224×224×3 image as input, passes it through five convolutional blocks with increasing filter sizes 
    (64, 128, 256, and 512), and then through three fully connected dense layers, ending with a softmax classifier for 8 classes.
                
    Some of its advantages over other models are:
    - **Simplicity:** Uses only 3×3 convolution filters and 2×2 pooling, making it more consistent and easier to implement.
    - **Good Generalization:** Performs well on different datasets beyond ImageNet with fine-tuning.
    - **Feature Extraction:** Its intermediate convolutional layers produce strong, reusable features for other vision tasks (e.g., detection, segmentation). 
                 
    Due to the advantages mentioned above VGG16 has been extensively explored in the medical image classification task 
    (e.g., cell classification, cancer diagnosis).
                
    </div>
    """,unsafe_allow_html=True)

    # Insert image of Vgg16 architecture
    st.image(white_bg('./pages/images/vgg16.png'), caption='Model architecture of VGG16',use_container_width=True)

    # Show model summary
    inputs=Input(shape=(None,None,3)) # Input layer
    x=Resizing(224,224)(inputs) # Resize inputs to (224,224,3)
    x=Rescaling(1/255)(x) # Rescale from [-1,1]
    vgg16_model = VGG16(weights=None, include_top=False, input_tensor=x)
    x=vgg16_model.output
    x=GlobalAveragePooling2D()(x)
    # Dense layers for classification
    x=Dense(1024,activation='relu')(x)
    x=Dropout(rate=0.2)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(rate=0.2)(x)
    outputs=Dense(8,activation='softmax')(x)
    model=Model(inputs=inputs,outputs=outputs)


    with st.expander("See Full Model Summary"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.code(stream.getvalue())

    st.markdown('## Performace')

    #Load the output of the training of the vgg16 model
    # Load the history of the model
    with open('./history_model_unfrozen_noweights.pkl', 'rb') as f:
        history_model = pickle.load(f)

    # Plot the loss/acc graphs
    st.markdown(" #### <u>Loss/Accuracy graphs </u>", unsafe_allow_html=True)
    
    acc_training = history_model['accuracy']
    acc_val = history_model['val_accuracy']

    loss_training = history_model['loss']
    loss_val = history_model['val_loss']

    # Create subplot grid: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))

    # First subplot (left)
    fig.add_trace(
        go.Scatter(x=np.arange(len(acc_training)+1), y=acc_training, mode="lines+markers", name="Training", line = dict(color='blue')),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=np.arange(len(acc_val)+1), y=acc_val, mode="lines+markers", name="Validation", line = dict(color='red')),
        row=1, col=1)

    # Second subplot (right)
    fig.add_trace(
        go.Scatter(x=np.arange(len(loss_training)+1), y=loss_training, mode="lines+markers", name="Training", line=dict(color="blue")),
        row=1, col=2)

    fig.add_trace(
        go.Scatter(x=np.arange(len(loss_val)+1), y=loss_val, mode="lines+markers", name="Validation", line=dict(color="red")),
        row=1, col=2)


    # Layout styling
    fig.update_layout(
        title=dict(text="Training Performance", font = dict(color="black"), ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title=dict(text = "Epochs", font = dict(color = "black")), tickfont = dict(color="black")),
        xaxis2=dict(title=dict(text = "Epochs", font = dict(color = "black")), tickfont = dict(color="black")),   # axis for the second subplot
        yaxis=dict(title=dict(text = "Accuracy", font = dict(color = "black")), tickfont = dict(color="black")),
        yaxis2=dict(title=dict(text = "Loss", font = dict(color = "black")), tickfont = dict(color="black")))


     # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    
   
    
    
   




