import streamlit as st
import random
import tempfile
import cv2
from mplsoccer import Pitch
from matplotlib.patches import Circle
from utils.load_model import load_models
from utils.labels_utils import get_labels_dics
from detect import predict

players_model, keypoints_model = load_models()
ball_players_dic, pitch_keypoints_dic, keypoints_map_pos = get_labels_dics()

st.write("# 2D Tactical Map")
video_path = st.file_uploader("Upload your video")
if video_path is not None:
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start detection", "start")
    with col2:
        stop_button = st.button("Stop detection", "stop")


    stframe = st.empty()

    if start_button and not stop_button:
        pitch = Pitch(pitch_color='grass', line_color='white',
                      stripe=True)  # optional stripes
        fig, ax = pitch.draw()

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_path.read())
        video_path = tfile.name
        fps = video_path.video_capture.get(cv2.CAP_PROP_FPS)
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            frame_id = 0
            success, frame = cap.read()
            results_players = players_model.predict(frame, imgsz=920, conf=0.4)
            results_keypoints = keypoints_model.predict(frame, imgsz=920, conf=0.6)
            
            # print(results_players, results_keypoints)
            frame_id += 1
        cap.release()
        # detect(stframe, cap, players_model)
        # st.pyplot(fig)
        # detect(stframe, cap, model, conf= 0.65)
        if stop_button:
            cap.release()
