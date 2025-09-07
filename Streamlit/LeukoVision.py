import streamlit as st

home = st.Page("./pages/Page1.py", title="LeukoVision", icon="🏠")
pred = st.Page("./Streamlit/pages/Prediction.py", title="Predictions", icon="🧬")


# Navigation container
pg = st.navigation(
    {
        "Main": [home],
        "Models": [pred],
    },
    position="top",  # 👈 horizontal navbar
)

# Run the active page
pg.run()