import streamlit as st
import plotly.graph_objects as go
import numpy as np


st.title('Data')


# Load data
cell = np.loadtxt('./Streamlit/pages/count_spanish_german_chinese.txt', usecols=0, dtype=str)
count = np.loadtxt('./Streamlit/pages/count_spanish_german_chinese.txt', usecols=[1,2,3], dtype=int)

# Create a stacked bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=cell,
    y=count[:,0],
    name='Spanish DS'
))


fig.add_trace(go.Bar(
    x=cell,
    y=count[:,1],
    name='German DS'
))

# Update layout for stacked bars
fig.update_layout(
    barmode='stack',
    xaxis_title='Cell type',
    yaxis_title='Population',
    xaxis_tickangle=-45,
    template='plotly_white'
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
