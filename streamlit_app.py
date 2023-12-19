import streamlit as st
from DentalCore.dentalcore import DentalCore

@st.cache_resource
def load_model():
    model_load_state = st.info(f"Loading model for...")
    yolo_model = DentalCore()
    model_load_state.empty()
    return yolo_model

yolo_model = load_model()

st.title("Demo for recognizing types of dental implants")

uploaded_file = st.sidebar.file_uploader("Choose the image file for Model to inference from")

if uploaded_file is not None:
    filename = uploaded_file.name
    preprocess_image = yolo_model.resize_image(uploaded_file.getvalue())
    inferenced_image = yolo_model.inference_xray(preprocess_image)
    st.write("Input Image")
    st.image(uploaded_file.getvalue(), caption='x-ray input')
    st.write("Output Image")
    st.image(inferenced_image, caption='model output')

