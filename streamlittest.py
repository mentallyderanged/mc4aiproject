import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

# Import functions from other files
from randomsampleselection import randomsampleselection
from Readandprep import load_dataset, prep_dataset
from Modeltraining import trainmodel

def main():
    st.title("Handwritten Digit Classification Web App")
    activities = ["Program", "Credits"]
    choices = st.sidebar.selectbox("Select Option", activities)

    if choices == "Program":
        st.subheader("Handwritten Digit Classification")

        # Random Sample Selection
        st.header("Step 1: Create Random Sample Dataset")
        source_dir = st.text_input("Enter the path to the source directory:")
        new_dataset_dir = st.text_input("Enter the desired path to the sample dataset directory:")
        number_of_images = st.number_input("Enter the number of images to be sampled per folder:", min_value=1, value=10)

        if st.button("Create Random Sample Dataset"):
            randomsampleselection(source_dir, new_dataset_dir, int(number_of_images))
            st.success("Random sample dataset created successfully!")

        # Load and Prepare Dataset
        st.header("Step 2: Load and Prepare Dataset")
        ds_path = st.text_input("Enter the path to the sample dataset directory:")
        testsize = st.slider("Enter the test size (between 0 and 1):", 0.0, 1.0, 0.2)

        if st.button("Load and Prepare Dataset"):
            X, y = load_dataset(ds_path)
            X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, testsize)
            st.session_state['prepared_data'] = (X_train, X_test, y_train_ohe, y_test_ohe)
            st.success("Dataset loaded and prepared successfully!")

        # Train Model
        if 'prepared_data' in st.session_state:
            st.header("Step 3: Train Model")
            epochs = st.number_input("Enter the number of epochs:", min_value=1, value=10, step=1)

            if st.button("Train Model"):
                X_train, _, y_train_ohe, _ = st.session_state['prepared_data']
                model = trainmodel(X_train, y_train_ohe, epochs)
                st.session_state['trained_model'] = model
                st.success("Model trained successfully!")

        # Drawing Canvas and Prediction
        if 'trained_model' in st.session_state:
            st.header("Step 4: Test the Model")
            st.subheader("Draw a digit below and click 'Predict Now'")
            canvas_result = st_canvas(
                stroke_width=20,
                stroke_color='rgb(255, 255, 255)',
                background_color='rgb(0, 0, 0)',
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas"
            )

            if canvas_result.image_data is not None:
                img = canvas_result.image_data
                img = Image.fromarray((img[:, :, 0] * 255).astype(np.uint8))
                img = img.resize((28, 28))
                st.image(img, caption='Resized Image', width=150)
                img = np.array(img) / 255.0
                img = img.reshape(1, 28, 28, 1)

                if st.button("Predict Now"):
                    model = st.session_state['trained_model']
                    prediction = model.predict(img)
                    predicted_class = np.argmax(prediction)
                    st.success(f"Predicted digit: {predicted_class}")

    elif choices == 'Credits':
        st.write("Created by Daniel - qvu with love <3")

if __name__ == "__main__":
    main()