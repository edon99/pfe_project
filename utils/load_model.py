from ultralytics import YOLO
import streamlit as st

@st.cache_data 
def load_players_model():
    return YOLO('../models/players.pt')
@st.cache_data 
def load_keypoints_model():
    return YOLO('../models/keypoints.pt')
def load_models():
    return load_players_model(), load_keypoints_model()