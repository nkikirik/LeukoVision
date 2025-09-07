import streamlit as st

st.markdown(
    """
    <style>
    /* Target the nav bar text */
    .st-emotion-cache-1wbqy5l a p {
        font-size:50px !important;   /* Increase/decrease font size */
        font-weight:600;             /* Make bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)

home = st.Page("./pages/home.py", title="LeukoVision", icon="🏠")
data = st.Page("./pages/data.py", title="Data", icon="📊")
pred = st.Page("./pages/prediction.py", title="Vision Lab", icon="🧪")
inception=st.Page("./pages/inception.py", title="InceptionV3")
resnet=st.Page("./pages/resnet.py", title="ResNet50")
vgg16=st.Page("./pages/vgg16.py", title="VGG16")

# Navigation container
pg = st.navigation(
    [home, data,
     {'Models':  [inception, resnet, vgg16]},
     pred
     ],
    position="top",  # 👈 horizontal navbar
)

# Run the active page
pg.run()