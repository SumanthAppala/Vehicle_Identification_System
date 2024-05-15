import streamlit as st
from PIL import Image
from utils.inference import Vehicle, LicencePlate

# Create instances of the required classes
vehicle = Vehicle()
licence = LicencePlate()

#********* This list can be modified as required (No training required) *************
CAR_BRANDS = ["Ford", "Mercedes", "BMW", "Ferrari"]

uploaded_file = st.file_uploader(
    label="Drop an image here or click to upload", 
    type=["png", "jpg", "jpeg"],
)

# Placeholder for the images
col1, col2 = st.columns(2)
original_placeholder = col2.empty()
annotated_placeholder = col1.empty()

image = None  

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    # Display the original image
    original_placeholder.image(original_image, caption='Uploaded Image', use_column_width=True)

    if original_image is not None:
        # Detect vehicles in the original image
        cropped_images, processed_image1 = vehicle.detect_vehicle(original_image)

        # Identify brand of each vehicle detected
        if cropped_images:
            for cropped_image in cropped_images:
                class_name, confidence = vehicle.classify_vehicle(cropped_image, CAR_BRANDS)
                
        # Detect licence plates and run OCR on them
        licence_number, processed_image2 = licence.detect_licence(processed_image1)

        # Display the annotated image
        annotated_placeholder.image(processed_image2, caption="Annotated Image", use_column_width=True)
        
        # Write results
        st.write(f"Brand: {class_name}, Confidence: {confidence:.2f}%")
        if licence_number is not None:
            st.write(f"Detected licence plate number: {licence_number}")
        else:
            st.write("No licence plate numbers could be extracted. Image unclear.")
    else:
        st.write("No objects detected.")

    
        