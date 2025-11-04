import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load your trained model
model = load_model("handwritten_model.h5")

st.title("✍️ Handwritten Alphabet Recognition (A–Z)")

# Canvas for drawing
canvas_result = st.canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img  # invert
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        st.success(f"🧠 Predicted Character: {chr(pred_class + 65)}")
    else:
        st.warning("Please draw a letter first!")
