import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.models
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import zipfile
import shutil  # Add shutil for deleting folders

# Import functions from other files
#from randomsampleselection import randomsampleselection
from Readandprep import load_dataset, prep_dataset
from Modeltraining import trainmodel
# from randomsampleselection import randomsampleselection
# Page selection sidebar
page = st.sidebar.selectbox("Select a page", ["Dataset Selection & Training", "Prediction"])

# Initialize session state for the model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'y_label' not in st.session_state:
    st.session_state.y_label = None
if 'flag' not in st.session_state:
    st.session_state.flag = 0
if 'flag2' not in st.session_state:
    st.session_state.flag2 = 0
if 'maxsamplesize' not in st.session_state:
    st.session_state.maxsamplesize = 0

if page == "Dataset Selection & Training":
    st.title("Dataset Loader, Processor & Model Training")

    # Row 1: Data Source Selection, Test Set Size, Epochs
    col1, col2, col3 = st.columns(3)
    with col1:
        # Dataset Selection
        option = st.selectbox("Select Dataset Source:", ["Default dataset (Alphabet) - pretrained model", "Default dataset (Alphabet) - custom settings ", "Custom Dataset"])
        if option == "Default dataset (Alphabet) - default settings":
            st.session_state.flag2 = 1
        elif option == "Default dataset (Alphabet) - custom settings ":
            st.session_state.flag2 = 1
        else:
            st.session_state.flag2 = 0

        temp_ds = "default_dataset" if option == "Default dataset (Alphabet) - custom settings " or  option == "Default dataset (Alphabet) - pretrained model"  else None
        if option == "Custom Dataset":
            temp_ds = st.file_uploader("Upload Dataset (zip file)", type="zip")
            temp_ds_path = None
            if temp_ds is not None:
                with zipfile.ZipFile(temp_ds, 'r') as zip_ref:
                    zip_ref.extractall("temp_dataset")
                temp_ds_path = "temp_dataset"
                for f in os.listdir(temp_ds_path):
                    folder_path = os.path.join(temp_ds_path, f)
                temp_ds_path = folder_path

            st.session_state.flag = 1
            #st.session_state.flag2 = 0
        else:
            st.session_state.flag = 0
            #st.session_state.flag2 = 1
    with col2:
        # Data Preprocessing Parameters
        test_size = st.number_input("Test Set Size (0.1 - 0.5)", min_value=0.1, max_value=0.5, value=0.15, step=0.05)

    with col3:
        # Model Training Parameters
        if option == "Default dataset (Alphabet) - pretrained model":
            epochs = st.number_input("Number of Training Epochs", min_value=1, value=50,disabled=True)
            st.session_state.flag2 = 1
        else:
            epochs = st.number_input("Number of Training Epochs", min_value=1, value=5)

    # Random Sample Selection (Optional) TOO MANY BUGS
    # if option == "Custom Dataset":
    #     use_random_sample = st.checkbox("Use Random Sample of Images", value=True,disabled=True)
    #     if use_random_sample:
    #         num_samples_per_class = st.number_input("Number of Samples per Class", min_value=1, value=50,disabled=True)

    dataset_path = None  # Initialize dataset_path
    if option == "Default dataset (Alphabet) - pretrained model" or option == "Default dataset (Alphabet) - custom settings ":
        dataset_path = "default_dataset"
    elif option == "Custom Dataset" and temp_ds is not None:
        dataset_path = temp_ds_path

    # Load, Preprocess, and Train
    if option == "Default dataset (Alphabet) - pretrained model":
        if st.button("Load, Preprocess & Evaluate Model"):
            if dataset_path is not None:
                with st.spinner("Loading and Preprocessing Dataset..."):
                    X, y, y_label,maxsamplesize = load_dataset(dataset_path)

                    X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, 0.2)

                st.success("Dataset Loaded and Preprocessed!")
                st.write(f"Training Set Shape: {X_train.shape}")
                st.write(f"Test Set Shape: {X_test.shape}")
                st.session_state.y_label = y_label

                with st.spinner(f"Loading Model..."):
                    st.session_state.model = tensorflow.keras.models.load_model("trainedmodel test3_1.h5")
                    st.session_state.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
                st.success("Model Loaded!")

                # Evaluate the model on the test set
                loss, accuracy = st.session_state.model.evaluate(X_test, y_test_ohe, verbose=0)
                st.write("## Evaluation on Test Set:")
                st.write(f"Loss: {loss:.4f}")
                st.write(f"Accuracy: {accuracy:.4f}")

                st.write("## Dataset preview:")
                unique_labels = np.unique(y)
                for label in unique_labels:
                    # Find indices of images belonging to the current label
                    indices = np.where(y == label)[0]
                    selected_indices = np.random.choice(indices, size=min(10, len(indices)), replace=False)
                    fig, axs = plt.subplots(1, 10, figsize=(10, 1))
                    for i, idx in enumerate(selected_indices):
                        axs[i].imshow(X[idx], cmap='gray')
                        axs[i].axis('off')

                    # Display the figure using Streamlit
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to prevent display issues in Streamlit

                # Delete the "temp_dataset" folder after displaying the preview
                if os.path.exists("temp_dataset"):
                    shutil.rmtree("temp_dataset")
        st.session_state.flag2 = 1

    elif option == "Default dataset (Alphabet) - custom settings " or option == "Custom Dataset":
        if st.button("Load, Preprocess & Train Model",disabled=False if option == "Default dataset (Alphabet) - custom settings " or (option == "Custom Dataset" and temp_ds is not None) else True):
            if dataset_path is not None:
                with st.spinner("Loading and Preprocessing Dataset..."):
                    X, y, y_label,maxsamplesize = load_dataset(dataset_path)
                    #if option == "Custom Dataset" and use_random_sample:
                        #randomsampleselection(dataset_path, 'temp_dataset_random', num_samples_per_class)
                        #X, y, y_label, maxsamplesize = load_dataset('temp_dataset_random')
                    X_train, X_test, y_train_ohe, y_test_ohe = prep_dataset(X, y, test_size)
                    #if st.session_state.flag == 1:
                        #if maxsamplesize < num_samples_per_class:
                            #st.warning(f"The dataset exists a folder which contain less than {num_samples_per_class} sample. Using {maxsamplesize}  samples or less per class instead.")
                            #st.stop()

                st.success("Dataset Loaded and Preprocessed!")
                st.write(f"Training Set Shape: {X_train.shape}")
                st.write(f"Test Set Shape: {X_test.shape}")
                st.session_state.y_label = y_label

                with st.spinner(f"Training Model..."):
                    st.session_state.model = trainmodel(X_train, y_train_ohe, epochs)
                    history = st.session_state.model.fit(X_train, y_train_ohe, epochs = epochs, verbose=1)

                st.success("Model Trained!")

                # Evaluate the model on the test set
                loss, accuracy = st.session_state.model.evaluate(X_test, y_test_ohe, verbose=0)
                st.write("## Evaluation on Test Set:")
                st.write(f"Loss: {loss:.4f}")
                st.write(f"Accuracy: {accuracy:.4f}")
                #test 1
                # st.write('length of unique y labels:',len(np.un(st.session_state.# y_label)))
                # st.write('y_train_ohe.shape:',y_train_ohe.shape)
                # st.write('length of unique y labels:',len(np.un(y_label)))


                fig = px.line(history.history, y=['accuracy', 'loss'], labels={'value': 'Metrics', 'index': 'Epoch'})
                fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Value')
                st.plotly_chart(fig)
                plt.close()
                st.write("## Dataset preview:")
                unique_labels = np.unique(y)
                for label in unique_labels:
                    # Find indices of images belonging to the current label
                    indices = np.where(y == label)[0]
                    selected_indices = np.random.choice(indices, size=min(10, len(indices)), replace=False)
                    fig, axs = plt.subplots(1, 10, figsize=(10, 1))
                    for i, idx in enumerate(selected_indices):
                        axs[i].imshow(X[idx], cmap='gray')
                        axs[i].axis('off')

                    # Display the figure using Streamlit
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to prevent display issues in Streamlit

                # Delete the "temp_dataset" folder after displaying the preview
                if os.path.exists("temp_dataset"):
                    shutil.rmtree("temp_dataset")

elif page == "Prediction":
    st.title("Make a Prediction")
    if st.session_state.model is not None:
        # Container for canvas and prediction

        # Columns for canvas and image upload
        col1, col2 = st.columns([2,1])

        #Brushstrokes size slider in sidebar
        stroke_width = st.sidebar.slider("Brushstrokes Size", 1, 30, 10)

        with col1:
            # Streamlit canvas setup
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
        with col2:
            uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], key="uploader")

        # Predict Button
        if st.button("Predict"):

            if canvas_result.image_data is not None:
                img = canvas_result.image_data
                if st.session_state.flag2 == 1:
                    img = cv2.resize(img, (32, 32))
                # Preprocess the input image
                img = cv2.resize(img, (64, 64))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = img_gray.astype(np.float32) / 255.0
                img_gray = img_gray.reshape(1, 64, 64, 1)  # Reshape for the model

                # Make prediction using the loaded model
                prediction = st.session_state.model.predict(img_gray)
                predicted_class = np.argmax(prediction)

                labels = np.unique(st.session_state.y_label)
                st.write(predicted_class)
                predicted_label = labels[predicted_class]
                st.write(st.session_state.flag2)
                # st.write("Length of labels:", len(labels))
                # st.write("Shape of prediction:", np.shape(prediction))
                # st.write("Length of prediction[0]:", len(prediction[0]))


                st.image(img_gray.reshape(64, 64), caption='Processed Input Image', use_column_width=False, clamp=True)
                st.write(f"Predicted Letter: {predicted_label}  {prediction[0][predicted_class]*100:.2f}%")

                # Get the top 5 predictions
                top5_indices = np.argsort(prediction[0])[::-1][:5]
                for i in top5_indices:
                    st.write(f"{labels[i]}: {prediction[0][i] * 100:.2f}%")




            if uploaded_image is not None:
                # Read and preprocess the uploaded image
                img = Image.open(uploaded_image).convert('RGB')  # Convert to RGB if necessary
                img = np.array(img)
                img = cv2.resize(img, (64, 64))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = img_gray.astype(np.float32) / 255.0
                img_gray = img_gray.reshape(1, 64, 64, 1)


                # Create inverted image
                img_gray_inverted = 1 - img_gray

                # Make prediction for normal image
                prediction = st.session_state.model.predict(img_gray)
                predicted_class = np.argmax(prediction)
                labels = np.unique(st.session_state.y_label)
                st.write(predicted_class)
                predicted_label = labels[predicted_class]

                col3, col4 = st.columns(2)

                # Display normal image
                with col3:
                    st.image(img_gray.reshape(64, 64), caption='Processed Input Image', use_column_width=False, clamp=True)
                    st.write(f"Predicted Letter (Normal): {predicted_label}  {prediction[0][predicted_class] * 100:.2f}%")

                    # Get the top 5 predictions for normal image
                    top5_indices = np.argsort(prediction[0])[::-1][:5]
                    for i in top5_indices:
                        st.write(f"{labels[i]}: {prediction[0][i] * 100:.2f}%")

                # Make prediction for inverted image
                prediction_inverted = st.session_state.model.predict(img_gray_inverted)
                predicted_class_inverted = np.argmax(prediction_inverted)
                predicted_label_inverted = labels[predicted_class_inverted]

                # Display inverted image
                with col4:
                    st.image(img_gray_inverted.reshape(64, 64), caption='Processed Inverted Image', use_column_width=False, clamp=True)
                    st.write(f"Predicted Letter (Inverted): {predicted_label_inverted}  {prediction_inverted[0][predicted_class_inverted] * 100:.2f}%")

                    # Get the top 5 predictions for inverted image
                    top5_indices_inverted = np.argsort(prediction_inverted[0])[::-1][:5]
                    for i in top5_indices_inverted:
                        st.write(f"{labels[i]}: {prediction_inverted[0][i] * 100:.2f}%")
            if 'canvas' in st.session_state:
                st.session_state.canvas.clear()

    else:
        st.write("Please train the model on the 'Dataset Selection & Training' page first.")
