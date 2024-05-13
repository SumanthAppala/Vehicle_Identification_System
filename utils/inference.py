from PIL import Image, ImageDraw
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def select_pytorch_device():
    """
    Selects the most appropriate device for PyTorch operations.
    Prioritizes CUDA, then MPS, and finally falls back to CPU.

    Returns:
        str: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"



class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    
class Vehicle(metaclass=SingletonMeta):
    def __init__(self):
        # Initialize the models and processors
        self.yolo_model = YOLO(r"model_weights/vehicle_weights.pt").to(self.device)
        self.clip_model = CLIPModel.from_pretrained(r"openai/clip-vit-base-patch16").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(r"openai/clip-vit-base-patch16")
        self.device = select_pytorch_device()

    def detect_vehicle(self, image: Image):
        # YOLO detection logic
        
        results = self.yolo_model.predict(image, 
                                             save=False,
                                             conf=0.5,
                                             classes=[3],
                                             device=self.device)
        
        cropped_images = []
        draw = ImageDraw.Draw(image) # Prepare to draw on the original image
        
        for r in results:
            # Check if there are any detections
            if hasattr(r.boxes, 'xyxy') and len(r.boxes.xyxy) > 0 and (int(r.boxes.cls[0])) == 3:
                for coordinates in r.boxes.xyxy:
                    x1, y1, x2, y2 = coordinates                 
                    cropped_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
                    cropped_images.append(cropped_image)
                    #draw.rectangle([x1, y1, x2, y2], outline="red", width = 4)

                annotated_image = results[0].plot()
        return cropped_images, annotated_image
    
    def classify_vehicle(self, image: Image, class_names):
        # CLIP classification logic
        text_inputs = [f"a photo of {c}" for c in class_names]
        inputs = self.clip_processor(text_inputs, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = ((100 * image_features @ text_features.T).softmax(dim=-1)) * 100
        values, indices = similarity[0].topk(1)

        return class_names[indices[0].item()], values[0].item()
    
class LicencePlate:
    def __init__(self):
        # Initialize the models and processors
        self.device = select_pytorch_device()
        # Load the YOLO model 
        self.model = YOLO(r"model_weights/licence_weights.pt").to(self.device)
    
    def detect_licence(self, image: Image):
        results = self.model.predict(
            task="detect",
            source=image,
            save=False,  # Do not save predictions as we're going to stream them
            conf=0.3,
            device=self.device)
        
        for detections in results:
            for coordinates in detections.boxes.xyxy:
                x1, y1, x2, y2 = coordinates
                cropped_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
                text = pytesseract.image_to_string(cropped_image)
        
        annotated_image = results[0].plot()
        return text, annotated_image
