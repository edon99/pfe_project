import streamlit as st
import cv2

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Football match analysis")

st.sidebar.title("Import your Video here")
with st.sidebar:
    uploaded_video=st.file_uploader("Upload your video")
    if uploaded_video:
        st.write("# Preview")
        st.video(uploaded_video, format="video/mp4")



st.markdown(
    """
    This platform helps users run a vast scale analysis on match videos.
      It provides statistics such as ball posession and concentration heat maps.

      ### How it works?

      this is how it works im not sure yet but we'll figure it out.
"""
)
