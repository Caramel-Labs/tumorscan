# imports
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# load pre-trained CNN model
model = load_model("./model/model.h5")


# preprocess image before inference
def preprocess_image(image_data):
    # resize image to fit input size of model
    img = image.load_img(image_data, target_size=(256, 256))

    # convert image to numpy array
    img_array = image.img_to_array(img)

    # expand dimensions to create batch of one image (as model has been trained on batches)
    img_batch = np.expand_dims(img_array, axis=0)

    # normalize pixel values to [0, 1]
    img_batch /= 255.0

    return img_batch


# set up Streamlit app
def main():
    st.title("TumorScan 🧠")
    st.markdown(
        "TumorScan helps you identify and classify brain tumors from MRI scans."
    )

    st.image("icon.png")

    # create file uploader widget
    uploaded_image = st.file_uploader(
        "Upload an MRI scan to get started.", type=["jpg", "jpeg", "png"]
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
            "Glioma",
            "Meningioma",
            "Pituitary Tumor",
            "No Tumor",
        ]

        category_descriptions = [
            "Gliomas are a type of brain tumor that originates in the glial cells, which support and protect nerve cells in the brain. They can vary in aggressiveness and symptoms depending on their location and grade. Treatment typically involves a combination of surgery, radiation therapy, and chemotherapy, tailored to the specific glioma type and stage.",
            "Meningiomas are typically slow-growing tumors that originate from the meninges, the protective membranes surrounding the brain and spinal cord. They are the most common type of primary brain tumor and are often benign, though they can become malignant in some cases. Symptoms may vary depending on their size and location, but treatment options, including surgery and radiation therapy, are available for managing meningiomas.",
            "Pituitary tumors are growths that develop in the pituitary gland, a crucial hormonal control center in the brain. They can disrupt hormone regulation, leading to a wide range of symptoms. Treatment options often include surgery, medication, or radiation therapy.",
        ]

        # get predicted category
        predicted_category = categories[np.argmax(prediction)]

        # display classification results
        if predicted_category == "No Tumor":
            st.success("No tumors detected in MRI scan.")
        else:
            st.warning(
                f"This MRI scan depicts a {predicted_category}. {category_descriptions[np.argmax(prediction)]}"
            )


if __name__ == "__main__":
    main()
