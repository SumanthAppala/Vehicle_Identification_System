from ultralytics import YOLO
import wandb
import torch

# Function to select the most appropriate device for PyTorch operations
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

def main():
    # Initialize the Weights & Biases project
    wandb.init(project='Vehicle_Identification', name="run1")
    device = select_pytorch_device()

    # Load the base YOLO model
    model = YOLO('yolov8m.pt')

    # Train the model
    results = model.train(save=True,
                          save_period=10,
                          data=r"data\Car_Dataset\data.yaml",
                          batch=16,
                          epochs=50,
                          plots=True,
                          project=r"Results",
                          name="run1",
                          device=device)

    # #### for validating the model on test dataset ####
    # model=YOLO("best.pt")
    # results = model.val(task="segment",
    #                     data='data.yaml',
    #                     save_json=True,
    #                     split="test",
    #                     plots=True,
    #                     save_txt=True)

    # #### for running inference on images (test dataset) #############

    # model=YOLO("best.pt")
    # results = model.predict(task="segment",
    #                         source="./images/test",
    #                         save=True,
    #                         save_txt=True)

    # ### for resuming training #######

    # wandb.init(project='Demo1', id='032ukvhi', resume=True)

    # model = YOLO('path/to/latest/epoch/file')
    # results = model.train(resume=True,
    #                     task="segment",
    #                     save=True,
    #                     save_period = 10,
    #                     data=r"data.yaml",
    #                     batch=16,
    #                     plots=True)
    
# This ensures the main() function is called only when the script is executed directly,
# not when imported as a module.
if __name__ == '__main__':
    main()

