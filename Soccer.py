import streamlit as st
import pickle
import pandas as pd
import numpy as np


clubs = ['Kortrijk', 'St. Gilloise', 'Gent', 'Club Brugge',
       'Oud-Heverlee Leuven', 'Beerschot VA', 'Mechelen', 'Genk',
       'Charleroi', 'Standard', 'St Truiden', 'Dender', 'Antwerp',
       'Westerlo', 'Cercle Brugge', 'Anderlecht', 'Eupen',
       'RWD Molenbeek', 'Waregem', 'Seraing', 'Oostende', 'Mouscron',
       'Waasland-Beveren']

opponent_clubs = ['Kortrijk', 'St. Gilloise', 'Gent', 'Club Brugge',
       'Oud-Heverlee Leuven', 'Beerschot VA', 'Mechelen', 'Genk',
       'Charleroi', 'Standard', 'St Truiden', 'Dender', 'Antwerp',
       'Westerlo', 'Cercle Brugge', 'Anderlecht', 'Eupen',
       'RWD Molenbeek', 'Waregem', 'Seraing', 'Oostende', 'Mouscron',
       'Waasland-Beveren']

home_away = ['Home', 'Away']

st.title('Belgian League Results Prediction')



col1, col2, col3 = st.columns(3)

with col3:
    home_team = st.selectbox('Home Team', sorted(clubs))


opponent_clubs = [club for club in clubs if club != home_team]

with col2:
    away_team = st.selectbox('Away Team', sorted(opponent_clubs))

with col1:
    Date = st.date_input('Match Date')

if st.button('Winner'):
    input_df = pd.DataFrame({
        'away_team': [away_team], 'Home_team': [home_team], 'Date':[Date]
    })
    result = pipe.predict_proba(input_df)
    st.header("Winner" + str(result))


st.markdown("""
    <style>
   
    .custom-selectbox-label {
        font-size: 50px; 
    }
    
    .stSelectbox div {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

import base64

# Ruta de la imagen JPG
ruta_imagen = (r"C:\Users\admin\Documents\Soccer_streamlit\images.jpg")

# Abrir la imagen en modo binario y convertirla a base64
with open(ruta_imagen, "rb") as img_file:
    imagen_base64 = base64.b64encode(img_file.read()).decode("utf-8")

import base64
from io import BytesIO
from PIL import Image


image_data = base64.b64decode(imagen_base64)


image = Image.open(BytesIO(image_data))




st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{imagen_base64}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>

    
    h1, h2, h3, h4, h5, h6, p,  label {
        color: #FFFFFF;
    }
    div {color: #000000}
    .stButton button {
        background-color: #00f900;  
        color: black;

    
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #333333;  
        color: black;  
    }
    </style>
    """,
    unsafe_allow_html=True
)


    