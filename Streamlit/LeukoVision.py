import streamlit as st

st.markdown(
    """
    <style>
    /* Target the nav bar text */
    .st-emotion-cache-1wbqy5l a p {
        font-size:18px !important;   /* Increase/decrease font size */
        font-weight:600;             /* Make bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)

home = st.Page("./pages/Page1.py", title="LeukoVision", icon="🏠")
pred = st.Page("./pages/Prediction.py", title="Predictions", icon="🧬")


# Navigation container
pg = st.navigation(
    [home,pred],
    # {"Main": [home],
    #     "Models": [pred]},
    position="top",  # 👈 horizontal navbar
)

# Run the active page
pg.run()