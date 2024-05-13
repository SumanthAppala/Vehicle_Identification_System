# Vehicle_Identification_System

# Overview:
This project delves into using Object Detection, Image Classification and OCR to build a vehicle identification system. 

The goal is to detect and track cars from an image using object detection. Then run image classification on the detected cars to identify their brands (Mercedes, Ferrari, Ford, etc). Next, we once again run object detection on the image to detect licence plates. Finally we run OCR on the detected licence plates to record the licence plte numbers of each car. 

# Frameworks used:
# YOLOv8: 
- Is an object detection framework, which can be trained on various datasets to detect and track specefic objects in an image/video. 
- Used to perform object detection to detect and localise cars, as well as licence plates, using custom trained weights
# CLIP: 
- Is a Vision Transformer that finds similarities between image text pairs. 
- Used to perform image classification on the detected cards, to identify their brands.
# Pytesseract: 
- Is an open-source tool used to perform OCR on images and extract text from them. 
- Used to extract licence plate numbers from the detected licence plate.

![alt text](image.png)

# Getting Started:

Install pytesseract and the required python libraries from the requirements.txt file.





