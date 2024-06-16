from ultralytics import YOLO
import streamlit as st

def load_players_model():
    return YOLO('../models/players.pt')
def load_keypoints_model():
    return YOLO('../models/keypoints.pt')
def load_models():
    return load_players_model(), load_keypoints_model()