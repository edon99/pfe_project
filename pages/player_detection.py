import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO
from detect import detect


model = YOLO("models/players.pt")


    ##INTERFACE
st.write("# Player Detection")    
video_path=st.file_uploader("Upload your video")  
if video_path is None:
    st.write("Please import your video")
else:
    start_button = st.button("Start detection","start")
    stop_button = st.button("Stop detection","stop")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_path.read())
    video_path=tfile.name
    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()

    if start_button and not stop_button:
        detect(stframe, cap, model)
    else:
        cap.release()
            
        