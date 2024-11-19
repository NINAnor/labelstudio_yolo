import os
import shutil
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

# Configuration
EXPORT_PATH = "/home/sindre.molvarsmyr/GitHub/exported_data/yolo"  # Directory where the YOLO data is saved, have to be full path
labelimgfolder = EXPORT_PATH
config_path = os.path.join(labelimgfolder, 'config.yaml') # Path to save the configuration file
with open(os.path.join(labelimgfolder, 'classes.txt'), 'r') as f:
    class_names = [line.strip() for line in f]
print(f"Classes: {class_names}")

# Check and create train/val directories
train_img_dir = os.path.join(labelimgfolder, 'images/train')
val_img_dir = os.path.join(labelimgfolder, 'images/val')
train_label_dir = os.path.join(labelimgfolder, 'labels/train')
val_label_dir = os.path.join(labelimgfolder, 'labels/val')

for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Check if splitting is needed
if not os.listdir(train_img_dir) and not os.listdir(val_img_dir):  # Assuming if one is empty, both are
    # List all images and labels
    images_path = os.path.join(labelimgfolder, 'images')
    labels_path = os.path.join(labelimgfolder, 'labels')
    images = [img for img in os.listdir(images_path) if img.endswith(('.jpg', '.png'))]
    labels = [label for label in os.listdir(labels_path) if label.endswith('.txt')]
    
    # Split data
    img_train, img_val, _, _ = train_test_split(images, images, test_size=0.2, random_state=42)  # Change test_size as needed
    
    # Move files
    for img_file in img_train:
        shutil.move(os.path.join(images_path, img_file), train_img_dir)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if label_file in labels:
            shutil.move(os.path.join(labels_path, label_file), train_label_dir)

    for img_file in img_val:
        shutil.move(os.path.join(images_path, img_file), val_img_dir)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if label_file in labels:
            shutil.move(os.path.join(labels_path, label_file), val_label_dir)

# Create config yaml
config = {
    'train': train_img_dir,
    'val': val_img_dir,
    'nc': len(class_names),
    'names': class_names
}

with open(config_path, 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

print(f"Configuration file created at {config_path}")

# train the model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=config_path, epochs=100, imgsz=640, batch = 1, save=True)