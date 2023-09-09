# imports
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# load pre-trained CNN model
model = load_model("../server/model/model.h5")


# preprocess image before inference
def preprocess_image(image_data):
    # Resize image to fit the input size of the model
    img = image.load_img(image_data, target_size=(256, 256))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to create a batch of one image
    img_batch = np.expand_dims(img_array, axis=0)

    # Normalize pixel values to [0, 1]
    img_batch /= 255.0

    return img_batch


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
        processed_image = preprocess_image(uploaded_image)

        print("Processed image is:")
        print(processed_image)

        # get prediction
        prediction = model.predict(processed_image)

        # define tumor category labels
        categories = [
            "Category 1 Tumor",
            "Category 2 Tumor",
            "Category 3 Tumor",
            "No Tumor",
        ]

        print(f"The predicted category is {np.argmax(prediction)}")

        # get predicted category
        predicted_category = categories[np.argmax(prediction)]

        # display classification results
        st.success(f"Predicted Tumor Category: {predicted_category}")


if __name__ == "__main__":
    main()
