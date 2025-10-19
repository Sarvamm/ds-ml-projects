# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from typing import Callable
import plotly.express as px
from plotly.subplots import make_subplots

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
        stroke_width=55,
        fill_color="#ffffff",
        stroke_color="#ffffff",
        background_color="#000000",
        height=560,
        width=560,
        key="canvas",
    )


grey = cv2.cvtColor(image_data.image_data, cv2.COLOR_BGR2GRAY)

input_array = cv2.resize(grey, (28, 28), interpolation=cv2.INTER_AREA)


# -------------------------- Loading trained weights ------------------------- #
data = np.load("model_parameters.npz")

W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]
W3 = data["W3"]
b3 = data["b3"]
W4 = data["W4"]
b4 = data["b4"]

# --------------------------- ACTIVATION FUNCTIONS --------------------------- #


# Sigmoid
def sigmoid(z: float | int) -> float:
    return 1 / (1 + np.exp(-z))


# ReLU
def relu(z: float | int) -> float:
    return np.maximum(0, z)


# Softmax
def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(np.exp(z))


# ------------------------------ DENSE LAYER --------------------------------- #
def dense(
    A_in: np.ndarray,  # (ndarray (m,n))
    W: np.ndarray,  # (ndarray (n,j))
    B: np.ndarray,  # (ndarray (1,j))
    g: Callable = relu,
) -> np.ndarray:
    """
    Computes dense layer
    Args:
    A_in (ndarray (m,n)) : Data, m examples, n features each
    W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
    b    (ndarray (1,j)) : bias vector, j units
    g    activation function (e.g. sigmoid, relu..)
    Returns
    A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """

    return g(A_in @ W + B)

# ---------------------- Contructing the Neural Network ---------------------- #
def sequential(X, W1, b1, W2, b2, W3, b3, W4, b4):
    a1 = dense(X, W1, b1, relu)
    a2 = dense(a1, W2, b2, relu)
    a3 = dense(a2, W3, b3, relu)
    a4 = dense(a3, W4, b4, softmax)
    return a4


def predict(X):
    pred_array = sequential(X, W1, b1, W2, b2, W3, b3, W4, b4)
    pred_num = np.argmax(pred_array)
    score = np.max(pred_array)
    return (pred_num, score, pred_array)

# --------------------------------- Inference -------------------------------- #
X = (input_array / 255).reshape(1, -1)

predicted_number, score, prediction_array = predict(X)


fig = make_subplots(1, 2)
fig.add_trace(px.histogram(x=range(10), y=prediction_array[0], nbins=20).data[0], 1, 1)
fig.add_trace(px.imshow(input_array[::-1,]).data[0], 1, 2)
fig.update_layout(height=500, width=1000)

with right:
    st.markdown(f"""# Predicted number: {predicted_number}
# Confidence: {score * 100: .3f} % """)
    st.plotly_chart(fig)
# ---------------------------------------------------------------------------- #
#                                      END                                     #
# ---------------------------------------------------------------------------- #