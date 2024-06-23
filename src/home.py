import streamlit as st


def set_config():
    st.set_page_config(layout="wide")
    st.logo('src/assets/TacticX.png')


def main():
    st.write("# Home page")

    st.markdown("""
      AI-Powered Software as a Service Solution designed to improve football team performance.

      A solution that can automatically analyze patterns, identify trends, and provide insightful metrics allowing coaches and analysts to make decisions based on evidence-backed insights.
    """)


if __name__ == '__main__':
    try:
        set_config()
        main()
    except SystemExit:
        pass
