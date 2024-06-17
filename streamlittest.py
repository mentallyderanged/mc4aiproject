import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image

# Import preprocessing and model modules
from Readandprep import load_dataset, processingds
from Modeltraining import trainmodel

def main():
    st.title("Handwritten Digit Classification Web App")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    activities = ["Program", "Credits"]
    choices = st.sidebar.selectbox("Select Option", activities)

    if choices == "Program":
        st.subheader("Draw a digit below and click 'Predict Now'")

        # User input for dataset path and test size
        ds_path = st.text_input("Enter the path to the sample dataset directory:")
        testsize = st.slider("Enter the test size (between 0 and 1):", 0.0, 1.0, 0.2)

        if st.button("Train Model"):
            st.write('test')
            if st.button("Predict Now"):
                st.write('test')
    elif choices == 'Credits':
        st.write("test")

if __name__ == "__main__":
    main()
