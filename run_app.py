import streamlit as st
from PIL import Image
from utils.inference import Vehicle, LicencePlate

# Create an instance of the ImageDecoder class
vehicle = Vehicle()
licence = LicencePlate()

CLASS_NAMES = ["Ford", "Mercedes", "BMW", "Ferrari"]

uploaded_file = st.file_uploader(
    label="Drop an image here or click to upload", 
    type=["png", "jpg", "jpeg"],
)

image = None  

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption='Uploaded Image', use_column_width=True)

    if original_image is not None:
        # Detect objects and get original image with bounding boxes
        cropped_images, annotated_image = vehicle.detect_vehicle(original_image)

        if cropped_images:
            for cropped_image in cropped_images:
                class_name, confidence = vehicle.classify_vehicle(cropped_image, CLASS_NAMES)
                st.write(f"Class: {class_name}, Confidence: {confidence:.2f}%")
        
        #Detect licence plates and run OCR on them
        licence_number, annotated_image = licence.detect_licence(annotated_image)
        st.write(f"Detect licence plates: {licence_number}")

        # Display the original image with bounding boxes
        st.image(annotated_image, caption="Detected cars and licence plates", use_column_width=True)
    else:
        st.write("No objects detected.")