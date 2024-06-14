import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
from detect import detect

model = YOLO("models/keypoints.pt")

    ##INTERFACE
st.write("# Keypoint Detection")    
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
        detect(stframe, cap, model, conf= 0.65)
    else:
        cap.release()
            
        