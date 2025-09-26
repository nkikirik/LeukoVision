import streamlit as st
from streamlit_option_menu import option_menu
import sys
from utils import white_bg
import io
import tensorflow as tf
from keras.applications import InceptionV3,ResNet50,VGG16
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, Rescaling, Resizing
from keras.models import Model    
import plotly.graph_objects as go
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd


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

    st.image(white_bg('./Streamlit/pages/images/inception/inceptionv3.png'), caption='Architecture diagram of InceptionV3',use_container_width=True)
    model = InceptionV3(weights=None)
    with st.expander("See Full Model Summary"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.code(stream.getvalue())
    st.markdown('### Performace')
    st.markdown(""" <div style="text-align: justify;">
    The training accuracy approaches 99.97%, while the validation accuracy reaches 
    98.29%, demonstrating the model’s strong ability to accurately classify different cell types.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show InceptionV3 loss and accuracy plot"):
        st.image(white_bg('./Streamlit/pages/images/inception/loss_acc.png'), caption='Loss and accruacy plot from InceptionV3 training',use_container_width=True)
    st.markdown(""" <div style="text-align: justify;">
                The test set shows a very high accuracy of 98.28% 
                and that is reflected in the digonal form of the confusion matrix.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show InceptionV3 confusion matrix"):
        st.image(white_bg('./Streamlit/pages/images/inception/cm.png'), caption='Confusion matrix of InceptionV3 test set',use_container_width=True)
    report = pd.read_csv("./Streamlit/pages/images/inception/class_report.txt", 
                     sep="\s+", header=0,
                     names=["Class", "Recall", "Specificity", "Precision", "F1-Score"])
    report.index = report.index + 1
    numeric_cols = report.select_dtypes(include="number").columns
    styled = report.style.format({col: "{:.2f}" for col in numeric_cols}) \
                        .set_properties(**{"text-align": "center"}) \
                        .set_table_styles([{
                            "selector": "th",
                            "props": [("text-align", "center"), ("font-weight", "bold")]
                        }])
    st.markdown(""" <div style="text-align: justify;">
                The classification metrics consistently range between 0.96 and 1.00, 
                demonstrating that the InceptionV3 model performs exceptionally well in 
                distinguishing among different WBC subtypes. This highlights both the robustness of 
                the model and its suitability for automated cell classification tasks.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show classification report"):
        st.dataframe(styled)
    
    

# The key idea behind InceptionV3 is the use of **Inception modules**, which allow the network to capture features at multiple scales simultaneously. Each module applies several convolutions of different sizes in parallel and concatenates the results, enabling the model to learn both fine and coarse features from an image.  

#     InceptionV3 incorporates several advanced techniques to improve training and reduce overfitting, including:

#     - **Factorized convolutions** to reduce computational cost while maintaining performance  
#     - **Auxiliary classifiers** that provide additional gradient signals during training  
#     - **Batch normalization** to stabilize and accelerate training  
#     - **Label smoothing** to improve generalization 
elif section == "ResNet50":
    st.subheader("ResNet50 🧬")
    st.markdown("""
        ### Overview  
        <div style="text-align: justify;">
        ResNet50 (Residual Network, 50 layers) is a deep convolutional neural network introduced by Microsoft Research in 2015.  
         
        The key innovation in ResNet50 is the **residual block**, which introduces **skip connections** (or shortcuts) that allow the model to "skip" one or more layers. These shortcuts enable the network to directly pass information forward, making it easier to train very deep architectures.  
 
        ResNet50 became one of the most influential models in deep learning, forming the backbone of many modern architectures. Its ability to combine **depth with stability** makes it highly effective for tasks like blood cell classification.
        </div>
        """,unsafe_allow_html=True)
    st.image(white_bg('./Streamlit/pages/images/resnet50/resnet50.png'), caption='Architecture diagram of ResNet50',use_container_width=True)
    model = ResNet50(weights=None)
    with st.expander("See Full Model Summary"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.code(stream.getvalue())
    st.markdown('### Performace')
    st.markdown(""" <div style="text-align: justify;">
    The training accuracy approaches 99.85%, while the validation accuracy reaches 
    97.90%, demonstrating the model’s strong ability to accurately classify different cell types.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show ResNet50 loss and accuracy plot"):
        st.image(white_bg('./Streamlit/pages/images/resnet50/loss_acc.png'), caption='Loss and accruacy plot from ResNet50 training',use_container_width=True)
    st.markdown(""" <div style="text-align: justify;">
                The test set shows a very high accuracy of 97.63% 
                and that is reflected in the digonal form of the confusion matrix.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show ResNet50 confusion matrix"):
        st.image(white_bg('./Streamlit/pages/images/resnet50/cm.png'), caption='Confusion matrix of ResNet50 test set',use_container_width=True)
    report = pd.read_csv("./Streamlit/pages/images/resnet50/class_report.txt", 
                     sep="\s+", header=0,
                     names=["Class", "Recall", "Specificity", "Precision", "F1-Score"])
    report.index = report.index + 1
    numeric_cols = report.select_dtypes(include="number").columns
    styled = report.style.format({col: "{:.2f}" for col in numeric_cols}) \
                        .set_properties(**{"text-align": "center"}) \
                        .set_table_styles([{
                            "selector": "th",
                            "props": [("text-align", "center"), ("font-weight", "bold")]
                        }])
    st.markdown(""" <div style="text-align: justify;">
                The classification metrics consistently range between 0.94 and 1.00, 
                demonstrating that the ResNet50 model performs exceptionally well in 
                distinguishing among different WBC subtypes. This highlights both the robustness of 
                the model and its suitability for automated cell classification tasks.
                </div>
                """,unsafe_allow_html=True)
    if st.toggle("Show classification report"):
        st.dataframe(styled)



#It was designed to address the **vanishing gradient problem** that arises when training very deep networks. 
# - **Depth:** 50 layers  
# - **Architecture:** Built from convolutional layers, batch normalization, ReLU activations, and residual blocks with skip connections  
# - **Strengths:** Efficient training of very deep networks, strong feature extraction, widely adopted in computer vision tasks  
# - **Applications:** Image classification, object detection, medical imaging, and transfer learning in many domains 

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
    st.image(white_bg('./Streamlit/pages/images/vgg16/vgg16.png'), caption='Model architecture of VGG16',use_container_width=True)

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
    with open('./Streamlit/history_model_unfrozen_noweights.pkl', 'rb') as f:
        history_model = pickle.load(f)

    # Plot the loss/acc graphs
    with st.expander("Loss/Accuracy graphs"):
        st.markdown(" #### <u>Loss/Accuracy graphs </u>\n" \
        "Although we do not use the pretrained weights from ImageNet" \
        ", after 5 epochs, we already have a steep increase in accuracy "
        "(decrease for loss function), which reaches 90% for both training and validation sets. At the last epochs " \
        "of our training, we notice that we have reached a plateau, which corresponds to 99% for the training set and " \
        "98% for the validation set, respectively.", unsafe_allow_html=True)
        
        acc_training = history_model['accuracy']
        acc_val = history_model['val_accuracy']

        loss_training = history_model['loss']
        loss_val = history_model['val_loss']

        # Create subplot grid: 1 row, 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"),vertical_spacing=0.6 )

        # First subplot (left)
        fig.add_trace(
            go.Scatter(x=np.arange(len(loss_training)+1), y=loss_training, mode="lines+markers", name="Training loss", line = dict(color='blue')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=np.arange(len(loss_val)+1), y=loss_val, mode="lines+markers", name="Validation loss", line = dict(color='red')),
            row=1, col=1)

        # Second subplot (right)
        fig.add_trace(
            go.Scatter(x=np.arange(len(acc_training)+1), y=acc_training, mode="lines+markers", name="Training accuracy", line=dict(color="blue")),
            row=1, col=2)

        fig.add_trace(
            go.Scatter(x=np.arange(len(acc_val)+1), y=acc_val, mode="lines+markers", name="Validation laccuracy", line=dict(color="red")),
            row=1, col=2)


        # Layout styling
        fig.update_layout(
            title=dict(text="Training Performance", font = dict(color="black") ),
            showlegend = True, 
            legend=dict( font=dict(size=12,color="black")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title=dict(text = "Epochs", font = dict(color = "black")), tickfont = dict(color="black")),
            xaxis2=dict(title=dict(text = "Epochs", font = dict(color = "black")), tickfont = dict(color="black")),   # axis for the second subplot
            yaxis=dict(title=dict(text = "Loss", font = dict(color = "black")), tickfont = dict(color="black")),
            yaxis2=dict(title=dict(text = "Accuracy", font = dict(color = "black")), tickfont = dict(color="black")))
        
        fig.update_layout(
        annotations=[
            dict(
                text="Loss",
                x=0.22, y=1.05, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=18, family="Arial", color="black")
            ),
            dict(
                text="Accuracy",
                x=0.78, y=1.05, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=18, family="Arial", color="black"))] )
        
            # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        

    with st.expander("Confusion matrix"):
        st.markdown(" #### <u> Confusion matrix </u> \n" \
        " The confusion matrix highlights not only the overall accuracy but also the specific strengths and weaknesses of the" \
        " model in classifying individual cell types. The most important finding is that all classes were predicted " \
        "at a rate of 97% or higher, which highlights the consistency of our model across all cell classes.", unsafe_allow_html=True)

        st.image(white_bg('./Streamlit/pages/images/vgg16/confusion_matrix.png'), caption='Confustion matrix of VGG16',use_container_width=True)


    with st.expander("Classification report"):
        st.markdown(" #### <u> Classification report </u>\n" \
        "As one can observe, our model achieves excellent results in all four evaluation metrics "
        "for every cell class, with scores consistently above 97%.", unsafe_allow_html=True)

        dict = {'BAS': [0.98, 1, 0.99, 0.98 ],
                'EOS': [1,1,1,1],
                'EBO': [0.97, 1, 0.98, 0.98],
                'IG' : [0.97, 0.99, 0.97, 0.97],
                'LYT': [1, 1, 0.99, 0.99],
                'MON': [0.98, 1, 0.97, 0.97],
                'NGS': [0.98, 1, 0.99, 0.98],
                'PLA': [0.99, 1, 1, 1]
                }
        df = pd.DataFrame.from_dict(dict, orient='index', columns=['Recall', 'Specificity', 'Precision', 'F1-score'])
        st.dataframe(df.style.format("{:.2f}"))
        


    
    
   
    
    
   




