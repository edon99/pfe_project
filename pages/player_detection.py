import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
from detect import detect_test



model = YOLO("models/players.pt")

    ##INTERFACE
st.write("# Player Detection")    

video_path=st.file_uploader("Upload your video")  
if video_path is None:
    st.write("Please import your video")
else:
    team1 = st.text_input(label="Team 1",value="Team A")
    team2 = st.text_input(label="Team 2", value="Team B")


    start_button = st.button("Start detection","start")
    stop_button = st.button("Stop detection","stop")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_path.read())
    video_path=tfile.name
    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()

    if start_button and not stop_button:
        st.toast(f'Detection Started!')
        detect_test(stframe, cap, model, team1, team2)
    elif stop_button:
        st.toast(f'Detection Stopped!')
        cap.release()
            
        