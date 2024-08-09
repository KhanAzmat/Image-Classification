from ultralytics import YOLO
import pandas as pd
import matplotlib as plt
from PIL import Image

if '__name__' == '__main__':

    # Fine-tune pretrained YOLO model
    root_dir = "./nycu-2023-deep-learning-final-project-1/achieve/"

    yolo_model = YOLO("yolov8n-cls.pt")

    yolo_results = yolo_model.train(data = root_dir, 
                                    epochs = 30,
                                    imgsz = 256 
                                    # similar with ResNet model -> but 256 (YOLO need multiple 32)
                                )
    
    yolo_result_csv = "./runs/classify/train6/results.csv"

    pd.read_csv(yolo_result_csv) #we get an accuracy of 95.6%

    # plot the loss
    yolo_result_img = "./runs/classify/train6/results.png"
    yolo_result_img = Image.open(yolo_result_img)

    plt.figure(figsize = (15, 15))
    plt.imshow(yolo_result_img)
    plt.axis("off");
