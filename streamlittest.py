import streamlit as st
from streamlit_drawable_canvas import st_canvas

def main():
    st.title("Handwritten Digit Classification Web App")
    activities = ["Program", "Credits"]
    choices = st.sidebar.selectbox("Select Option", activities)

    if choices == "Program":
        st.subheader("Draw a digit below and click 'Predict Now'")

        # User input for dataset path and test size
        ds_path = st.text_input("Enter the path to the sample dataset directory:")
        testsize = st.slider("Enter the test size (between 0 and 1):", 0.0, 1.0, 0.2)

        if st.button("Load and Prepare Dataset"):
            st.success("Dataset loaded and prepared successfully! (Placeholder)")

            # User input for number of epochs
            epochs = st.number_input("Enter the number of epochs:", min_value=1, value=10, step=1)

            if st.button("Train Model"):
                st.success("Model trained successfully! (Placeholder)")

                canvas_result = st_canvas(stroke_width=15,
                                          stroke_color='rgb(255, 255, 255)',
                                          background_color='rgb(0, 0, 0)',
                                          height=150,
                                          width=150,
                                          key="canvas")

                if canvas_result.image_data is not None:
                    img = canvas_result.image_data
                    st.image(img)

                    if st.button("Predict Now"):
                        st.success("Prediction done! (Placeholder)")

    elif choices == 'Credits':
        st.write("Placeholder")

if __name__ == "__main__":
    main()
