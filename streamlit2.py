
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Import functions from other files
from randomsampleselection import randomsampleselection
from Readandprep import load_dataset, prep_dataset
from Modeltraining import trainmodel

# Page selection sidebar
page = st.sidebar.selectbox("Select a page", ["Dataset Selection & Training", "Prediction"])

# Initialize session state for the model
if 'model' not in st.session_state:
    st.session_state.model = None
if page == "Dataset Selection & Training":
    st.title("Dataset Loader, Processor & Model Training")

    # Dataset Selection
    option = st.selectbox("Select Dataset Source:", ["Default Dataset", "Custom Dataset"])
    dataset_path = "default_dataset" if option == "Default Dataset" else None
    if option == "Custom Dataset":
        dataset_path = st.file_uploader("Upload Dataset (zip file)", type="zip")

    # Data Preprocessing Parameters
    test_size = st.number_input("Test Set Size (0.1 - 0.5)", min_value=0.1, max_value=0.5, value=0.2, step=0.1)

    # Random Sample Selection (Optional)
    if option == "Custom Dataset":
        use_random_sample = st.checkbox("Use Random Sample of Images", value=True)
        if use_random_sample:
            num_samples_per_class = st.number_input("Number of Samples per Class", min_value=1, value=500)

    # Model Training Parameters
    epochs = st.number_input("Number of Training Epochs", min_value=1, value=100)

    # Load, Preprocess, and Train
    if st.button("Load, Preprocess & Train Model"):
        if dataset_path is not None:
            with st.spinner("Loading and Preprocessing Dataset..."):
                X, y = load_dataset(dataset_path)
                if option == "Custom Dataset" and use_random_sample:
                    X, y = randomsampleselection(X, y, num_samples_per_class)
                X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, test_size)

            st.success("Dataset Loaded and Preprocessed!")
            st.write(f"Training Set Shape: {X_train.shape}")
            st.write(f"Test Set Shape: {X_test.shape}")

            with st.spinner(f"Training Model..."):
                st.session_state.model = trainmodel(X_train, y_train_ohe, epochs)

            st.success("Model Trained!")

            # Evaluate the model on the test set
            loss, accuracy = st.session_state.model.evaluate(X_test, y_test_ohe, verbose=0)
            st.write("## Evaluation on Test Set:")
            st.write(f"Loss: {loss:.4f}")
            st.write(f"Accuracy: {accuracy:.4f}")
elif page == "Prediction":
    st.title("Make a Prediction")
    if st.session_state.model is not None:
        # Streamlit canvas setup
        stroke_width =  30
        stroke_color = "#eee"
        bg_color = "black"
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:", ("freedraw", "line")
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode=drawing_mode,
            key="canvas",
        )

        if st.button("Predict"):
            if canvas_result.image_data is not None:
                img = canvas_result.image_data

                # Preprocess the input image
                img = cv2.resize(img, (32, 32))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = img_gray.astype(np.float32) / 255.0
                img_gray = img_gray.reshape(1, 32, 32, 1)  # Reshape for the model

                # Make prediction using the loaded model
                prediction = st.session_state.model.predict(img_gray)
                predicted_class = np.argmax(prediction)

                # test labels 4 default alphabet ONLY! NEED TO CHANGE IF DIFFERENT DATASET IS USED!
                labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                predicted_label = labels[predicted_class]

                st.write(f"Predicted Letter: {predicted_label}")
    else:
        st.write("Please train the model on the 'Dataset Selection & Training' page first.")
