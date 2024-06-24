import streamlit as st


def set_config():
    st.set_page_config(layout="wide")
    st.logo('src/assets/TacticX.png')


def main():
    st.write("# TactiX")

    st.markdown("""
      AI-Powered Software as a Service Solution designed to improve football team performance.

      A solution that can automatically analyze patterns, identify trends, and provide insightful metrics allowing coaches and analysts to make decisions based on evidence-backed insights.
    """)

    st.write("## Conditions")
    st.markdown("""
    1. The uploaded video must be taken using a tactical POV camera that captures the field from an upper perspective.
    2. The uploaded video must show the field from a horizontal view.
    3. The quality of the video must be at least 720p to ensure the detection of all players on the field.
    """)

    st.write("## Tutorial")
    st.markdown("""
    1. Upload the video that respects the conditions mentioned above under a format that's cited in the video field.
    2. once the video is uploaded, click on the Start Detection button to start.
    """)

    col1, col2 = st.columns(2)
    

    with col1:
        st.image("src/assets/ss2.png")

    with col2:
        st.image("src/assets/ss1.png")

    
    st.write("## Coming Soon")
    st.markdown("""
        The Stats tab is a representation of how we plan on visualising data obtained from analysing the videos.
        Currently it is a data visualisation mini-app of AFCON data that was made available by StatsBomb.
        The graphs, pitches and visuals were made using MplSoccer.
    """)

if __name__ == '__main__':
    try:
        set_config()
        main()
    except SystemExit:
        pass
