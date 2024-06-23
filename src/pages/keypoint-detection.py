import streamlit as st
import tempfile
import cv2
from utils.load_model import load_keypoints_model
from utils.detect import track

model = load_keypoints_model()

##INTERFACE
st.write("# Keypoint Detection")
video_path = st.file_uploader("Upload your video")
if video_path is None:
    st.write("Please import your video")
else:
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start detection", "start")
    with col2:
        stop_button = st.button("Stop detection", "stop")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_path.read())
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()

    if start_button and not stop_button:
        track(stframe, cap, model, conf=0.65)
    else:
        cap.release()
