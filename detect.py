# Import libraries
import numpy as np
import pandas as pd
import streamlit as st


import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO




def detect_players(stframe,cap, model):
        st.toast(f'Detection Started!')
        while cap.isOpened():
            success, frame = cap.read()
            if success:
            # results = model(frame, conf=0.4)
                results = model.track(frame,persist=True, conf=0.4)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame)

            else:
                print("error detecting")

    
        