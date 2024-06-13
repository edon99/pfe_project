import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import random

from mplsoccer import Pitch, VerticalPitch
from PIL import Image, ImageColor
from ultralytics import YOLO
from detect import detect
from matplotlib.patches import Circle, Arrow


pitch = Pitch(pitch_color='grass', line_color='white',
              stripe=True)  # optional stripes
fig, ax = pitch.draw()

start=st.button(label="Start")

if start:

    for _ in range(10):
        center_x = random.randint(20, 100)
        center_y = random.randint(20, 100)
        radius = 3
        circle = Circle(xy=(center_x, center_y), radius=radius, color='red')
        ax.add_patch(circle)
    st.pyplot(fig)

    