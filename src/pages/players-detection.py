import streamlit as st
import tempfile
import cv2
from utils.detect import track_players
from utils.load_model import load_players_model

model = load_players_model()

##INTERFACE
st.write("# Player Detection")

video_path = st.file_uploader("Upload your video")
if video_path is None:
    st.write("Please import your video")
else:
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.text_input(label="Team 1", value="Team A")
    with col2:
        team2 = st.text_input(label="Team 2", value="Team B")

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
        st.toast(f'Detection Started!')
        track_players(stframe, cap, model, team1, team2)
    elif stop_button:
        st.toast(f'Detection Stopped!')
        cap.release()
