import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

# Import functions from other files
from randomsampleselection import randomsampleselection
from Readandprep import load_dataset, prep_dataset
from Modeltraining import trainmodel

# Page selection sidebar
page = st.sidebar.selectbox("Select a page", ["Dataset selection - Model training", "Prediction"])

if page == "DDataset selection - Model training":
    st.title("Dataset Loader and Processor")

    option = st.selectbox("Select an option", ["Custom dataset","Default dataset"])
    
    if option == "Deafult dataset":
        # Load the default dataset
        
    if option == "Custom dataset":
        # File uploader
        dataset_path = st.file_uploader("Upload dataset (zip file)", type="zip")
        
    
    
    # Test size slider
    test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)

    if dataset_path is not None:
        # Load and preprocess the dataset
        X, y = load_dataset(dataset_path)  # Assuming load_dataset can handle zip files
        X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, test_size)

        st.success("Dataset loaded and preprocessed successfully!")
        st.write(f"Training set shape: {X_train.shape}")
        st.write(f"Test set shape: {X_test.shape}")

elif page == "Prediction":
    st.title("Model Training and Evaluation")

    # Model selection (example)
    model_type = st.selectbox("Select a model", ["CNN", "ResNet", "VGG16"])

    # Training parameters (example)
    epochs = st.slider("Number of epochs", min_value=1, max_value=100, value=10)

    # Training button
    if st.button("Train Model"):
        st.write(f"Training {model_type} model...")
        # Add model training and evaluation logic here
        st.success("Model trained and evaluated!")

        # Display results
        st.write("## Evaluation Results")
        st.write(f"Accuracy: {0.95}") # Example accuracy
        st.write(f"Loss: {0.05}") # Example loss
