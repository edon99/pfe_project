import streamlit as st
import tempfile
import cv2
from utils.load_model import load_models
from utils.detection import detect


@st.cache_data
def init_data():
    players_model, keypoints_model = load_models()
    tac_map = cv2.imread('src/assets/2d-pitch.png')
    return players_model, keypoints_model, tac_map




def main():
    players_model, keypoints_model, tac_map = init_data()
    st.write("# 2D Tactical Map")
    input_vide_file = st.file_uploader(
        'Upload your match', type=['mp4', 'mov', 'avi', 'm4v', 'asf'])
    if input_vide_file is None:
        st.write("Please import your video")
    else:
        col1, col2 = st.columns(2)
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.text_input(label="Team 1", value="Team A")
        with col2:
            team2 = st.text_input(label="Team 2", value="Team B")
        with col1:
            start_button = st.button("Start detection", "start")
        with col2:
            stop_button = st.button("Stop detection", "stop")

        tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tempf.write(input_vide_file.read())
        stframe = st.empty()
        cap = cv2.VideoCapture(tempf.name)
        status = False

        if start_button and not stop_button:
            st.toast(f'Detection Started!')
            status = detect(cap,
                            stframe,
                            team1, team2,
                            players_model,
                            keypoints_model,
                            tac_map = tac_map)
            cap.release()
        else:
            try:
                cap.release()
            except:
                pass
        if stop_button or status:
            st.toast(f'Detection Stopped!')
            cap.release()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
