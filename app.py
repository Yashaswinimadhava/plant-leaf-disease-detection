
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = Path("models/plant_disease_model.keras")
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Leaf Disease Detection")
st.write(
    "Upload a plant leaf image to get a prediction "
    "from your trained image classification model."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    if st.button("Predict disease"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize(IMAGE_SIZE)
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # The training model includes its own Rescaling layer.
            predictions = model.predict(image_array, verbose=0)[0]

            class_index = int(np.argmax(predictions))
            confidence = float(predictions[class_index]) * 100

        st.subheader("Prediction")
        st.success(f"Predicted class: {class_index}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.caption(
            "This is a model prediction, not a definitive diagnosis. "
            "Results may be unreliable for images unlike the training data."
        )
