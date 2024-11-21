import os
import shutil
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

# Configuration
EXPORT_PATH = "/home/sindre.molvarsmyr/GitHub/exported_data/yolo"  # Directory where the YOLO data is saved, have to be full path
labelimgfolder = EXPORT_PATH
config_path = os.path.join(labelimgfolder, 'config.yaml')  # Path to save the configuration file

# Read classes from classes.txt
with open(os.path.join(labelimgfolder, 'classes.txt'), 'r') as f:
    class_names = [line.strip() for line in f]
print(f"Original Classes: {class_names}")

# Merge classes if needed
DoClassMerge = True # Set to True if you want to merge classes
if DoClassMerge:
    # Merge mapping
    # Specify which classes to merge, e.g., combine 'Common murre adult' and 'Common murre chick' into 'Common murre'
    merge_mapping = {
        "Beak with fish": "Beak with fish",
        "Common murre adult": "Common murre",
        "Common murre chick": "Common murre chick",
        "Common murre egg": "Common murre egg",
        "Common murre juvenile": "Common murre",
        "European shag": "European shag"
    }

    # Create a new class list based on the mapping
    new_class_names = list(set(merge_mapping.values()))
    print(f"Merged Classes: {new_class_names}")

    # Backup original labels
    labels_path = os.path.join(labelimgfolder, 'labels')
    backup_labels_path = os.path.join(labelimgfolder, 'labels_original')
    if not os.path.exists(backup_labels_path):
        shutil.copytree(labels_path, backup_labels_path)
        print(f"Backup of original labels created at {backup_labels_path}")

    # Update label files based on the merge mapping
    # By using backup_labels_path, we can avoid overwriting the original labels, and can retrain the model with the original labels if needed
    for label_file in os.listdir(backup_labels_path):
        if label_file.endswith('.txt'):
            label_file_path = os.path.join(labels_path, label_file)
            with open(label_file_path, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                class_id, *coords = line.strip().split()
                class_id = int(class_id)
                original_class = class_names[class_id]
                new_class = merge_mapping[original_class]
                new_class_id = new_class_names.index(new_class)
                updated_lines.append(f"{new_class_id} {' '.join(coords)}\n")
            
            # Overwrite the label file with updated class IDs
            with open(label_file_path, 'w') as f:
                f.writelines(updated_lines)

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
    images = [img for img in os.listdir(images_path) if img.endswith(('.jpg', '.png'))]
    
    # Split data
    img_train, img_val = train_test_split(images, test_size=0.2, random_state=42)  # Change test_size as needed
    
    # Move files
    for img_file in img_train:
        shutil.move(os.path.join(images_path, img_file), train_img_dir)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.move(os.path.join(labels_path, label_file), train_label_dir)

    for img_file in img_val:
        shutil.move(os.path.join(images_path, img_file), val_img_dir)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.move(os.path.join(labels_path, label_file), val_label_dir)

# Create config yaml
config = {
    'train': train_img_dir,
    'val': val_img_dir,
    'nc': len(new_class_names),
    'names': new_class_names
}

with open(config_path, 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

print(f"Configuration file created at {config_path}")

# Train the model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
results = model.train(data=config_path, epochs=100, imgsz=640, batch=1, save=True)
