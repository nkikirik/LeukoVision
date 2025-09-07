import streamlit as st

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .st-emotion-cache-1wbqy5l a p {
        font-size:50px !important;   /* Increase/decrease font size */
        font-weight:600;             /* Make bold */
    }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

home = st.Page("./pages/home.py", title="LeukoVision", icon="🏠")
data = st.Page("./pages/data.py", title="Data", icon="📊")
pred = st.Page("./pages/visionlab.py", title="Vision Lab", icon="🧪")
models=st.Page("./pages/models.py", title="Modeling", icon='🤖')

# Navigation container
pg = st.navigation(
    [
        home,
        data,
        models,
        pred
    ],
    position="top",
)

# Run the active page
pg.run()