# if the training fails because of memory limits in trainlabelstudio.py, this script can be used to only train the model and not move the images to train/val folders:

import os
import shutil
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

# Configuration
EXPORT_PATH = "/home/sindre.molvarsmyr/GitHub/exported_data/yolo"  # Directory where the YOLO data is saved, have to be full path
labelimgfolder = EXPORT_PATH
config_path = os.path.join(labelimgfolder, 'config.yaml')  # Path to save the configuration file



# Train the model
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)
results = model.train(data=config_path, epochs=1000, imgsz=416, batch=1, save=True)
