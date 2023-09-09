# imports
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# load pre-trained CNN model
model = load_model("../server/model/model.h5")


# preprocess image before inference
def preprocess_image(image):
    # resize image to fit input size of model
    image = image.resize((256, 256))
    return image


# set up Streamlit app
def main():
    st.title("TumorScan")
    st.markdown("Upload an MRI image for tumor classification.")

    # create file uploader widget
    uploaded_image = st.file_uploader(
        "Choose an MRI image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        # display uploaded image
        st.image(uploaded_image, caption="Uploaded MRI Image", use_column_width=True)

        # preprocess image for prediction
        processed_image = preprocess_image(Image.open(uploaded_image))

        # get prediction
        prediction = model.predict(processed_image)

        # display classification results
        if prediction[0][0] > 0.5:
            st.error("Tumor Detected")
        else:
            st.success("No Tumor Detected")


if __name__ == "__main__":
    main()
