# Import libraries
import numpy as np
import pandas as pd
import streamlit as st
from typing import List

import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO

def detect(stframe, cap, model, conf = 0.4):
        while cap.isOpened():
            success, frame = cap.read()
            if success:
            # results = model(frame, conf=0.4)
                results = model.track(frame, persist=True, conf=conf)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame)
            else:
                print("error detecting")

def predict(model, frame, conf, format = 'xyxy'):
    """
    Predicts the bounding boxes of the objects in the image

    Returns
    -------
    pd.DataFrame
        DataFrame containing the bounding boxes
    """
    results = model(frame, conf=conf)
    # return results[0].boxes.xyxy.cpu().numpy()
    return results[0]

    
        