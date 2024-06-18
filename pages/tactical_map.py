import streamlit as st
import tempfile
import cv2
from utils.load_model import load_models
from utils.detection import detect


@st.cache_data
def init_data():
    players_model, keypoints_model = load_models()
    tac_map = cv2.imread('assets/2d-pitch.png')
    return players_model, keypoints_model, tac_map


def main():
    players_model, keypoints_model, tac_map = init_data()
    st.write("# 2D Tactical Map")
    input_vide_file = st.file_uploader(
        'Upload your match', type=['mp4', 'mov', 'avi', 'm4v', 'asf'])
    if input_vide_file is None:
        st.write("Please import your video")
    else:
        save_output = st.checkbox(label='Save output', value=False)
        if save_output:
            output_file_name = st.text_input(
                label='File Name (Optional)',
                placeholder='Enter output video file name.')
        else:
            output_file_name = 'out'
        col1, col2 = st.columns(2)
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
                            output_file_name,
                            save_output,
                            players_model,
                            keypoints_model,
                            tac_map = tac_map,
                            num_pal_colors=3)
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
