import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------- #
#                                 PAGE CONFIG                                  #
# ---------------------------------------------------------------------------- #
st.set_page_config(layout="wide", page_title="Digit Recognizer")
st.title("Draw a number (0-9)")
st.markdown("Draw on the canvas, and the computer will guess what your number is.")

left, right = st.columns(2)

# ---------------------------------------------------------------------------- #
#                                CANVAS SETUP                                  #
# ---------------------------------------------------------------------------- #
with left:
    image_data = st_canvas(
        stroke_width=28,
        fill_color="#ffffff",
        stroke_color="#ffffff",
        background_color="#000000",
        height=280,
        width=280,
        key="canvas",
        return_image_data=True,  # Fixes: RuntimeError (image_data not requested)
    )

# ---------------------------------------------------------------------------- #
#                            MODEL & PREDICTION                                #
# ---------------------------------------------------------------------------- #
@st.cache_resource
def load_keras_model():
    # Fixes: FileNotFoundError on Streamlit Cloud by deriving absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Dense_Model.keras")

    if not os.path.exists(model_path):
        root_files = os.listdir(base_dir)
        raise FileNotFoundError(
            f"Model not found at: {model_path}. Files present in directory: {root_files}"
        )

    return load_model(model_path)

model = load_keras_model()

def predict(X):
    pred_array = (
        MinMaxScaler()
        .fit_transform(model.predict(X, verbose=0).reshape(-1, 1))
        .reshape(-1)
    )
    pred_num = np.argmax(pred_array)
    score = np.max(pred_array)
    return (pred_num, score, pred_array)

# ---------------------------------------------------------------------------- #
#                             INFERENCE & OUTPUT                               #
# ---------------------------------------------------------------------------- #
# Fixes: cv2.cvtColor assertion error by ensuring image array exists and is non-empty
if (
    image_data is not None 
    and image_data.image_data is not None 
    and image_data.image_data.size > 0
):
    # Ensure correct data type (uint8)
    raw_img = image_data.image_data.astype("uint8")

    # Fixes: st_canvas produces 4-channel RGBA data
    grey = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2GRAY)

    # Only run prediction if the user has actually drawn something
    if np.any(grey > 0):
        # Resize to model input size (28x28)
        input_array = cv2.resize(grey, (28, 28), interpolation=cv2.INTER_AREA)

        # Preprocess array
        X = (input_array / 255.0).reshape(1, -1)

        # Run model inference
        predicted_number, score, prediction_array = predict(X)

        # Plot confidence histogram and image preview
        fig = make_subplots(1, 2)
        fig.add_trace(
            px.histogram(x=list(range(10)), y=prediction_array, nbins=20).data[0], 
            1, 1
        )
        fig.add_trace(px.imshow(input_array[::-1, :]).data[0], 1, 2)
        fig.update_layout(height=500, width=1000)

        with right:
            st.markdown(f"# Predicted number: {predicted_number}")
            st.markdown(f"# Confidence: {score * 100:.3f} %")
            st.plotly_chart(fig, use_container_width=True)
    else:
        with right:
            st.info("Draw a digit on the canvas to see predictions.")
else:
    with right:
        st.info("Draw a digit on the canvas to see predictions.")
