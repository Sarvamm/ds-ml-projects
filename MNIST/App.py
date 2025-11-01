# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
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
st.set_page_config(layout="wide")
st.title("Draw a number (0-9)")
st.markdown("""
Draw on the canvas, The computer will guess what your number is 
""")
left, right = st.columns(2)

with left:
    # Create a canvas component
    image_data = st_canvas(
        stroke_width=28,
        fill_color="#ffffff",
        stroke_color="#ffffff",
        background_color="#000000",
        height=280,
        width=280,
        key="canvas",
    )


grey = cv2.cvtColor(image_data.image_data, cv2.COLOR_BGR2GRAY)

input_array = cv2.resize(grey, (28, 28), interpolation=cv2.INTER_AREA)

model = load_model("Dense_Model.keras")


def predict(X):
    pred_array = (
        MinMaxScaler()
        .fit_transform(model.predict(X).reshape(-1, 1))
        .reshape(
            -1,
        )
    )
    pred_num = np.argmax(pred_array)
    score = np.max(pred_array)
    return (pred_num, score, pred_array)


# --------------------------------- Inference -------------------------------- #
X = (input_array / 255).reshape(1, -1)

predicted_number, score, prediction_array = predict(X)


fig = make_subplots(1, 2)
fig.add_trace(px.histogram(x=range(10), y=prediction_array, nbins=20).data[0], 1, 1)
fig.add_trace(px.imshow(input_array[::-1,]).data[0], 1, 2)
fig.update_layout(height=500, width=1000)

with right:
    st.markdown(f"""# Predicted number: {predicted_number}
# Confidence: {score * 100: .3f} % """)
    st.plotly_chart(fig)
# ---------------------------------------------------------------------------- #
#                                      END                                     #
# ---------------------------------------------------------------------------- #
