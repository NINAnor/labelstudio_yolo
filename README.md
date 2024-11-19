
# README for LabelStudio YOLO Export and Training Workflow

## Overview

This repository contains two Python scripts for managing the export and training of YOLO datasets using data from labelstudio.nina.no:

1. `exportlabelstudio.py`: Automates the export of labeled data from LabelStudio, downloads associated images, and extracts YOLO-formatted data for use in training.
2. `trainlabelstudio.py`: Prepares the YOLO dataset for training by splitting it into training and validation sets, creating a configuration file, and initiating training using the YOLOv8 framework.

---

## Prerequisites

### Dependencies
Ensure the following Python libraries are installed:
- `requests`
- `sklearn`
- `pyyaml`
- `ultralytics`
- `zipfile`

Install them using pip:
```bash
pip install requests scikit-learn pyyaml ultralytics
```

### Configuration
Both scripts use an API token for LabelStudio authentication. Ensure you have the following:
- **LabelStudio API Token**: Replace the placeholder `API_TOKEN` in the `authenticate.py` file with your actual token.
- **Project ID**: Replace the `PROJECT_ID` placeholder with your LabelStudio project ID.

Create a directory structure for storing the exported and processed data:
```bash
mkdir -p exported_data/yolo/images/train exported_data/yolo/images/val
mkdir -p exported_data/yolo/labels/train exported_data/yolo/labels/val
```

---

## Script Descriptions

### 1. `exportlabelstudio.py`
#### Purpose:
- Export labeled data from LabelStudio in YOLO format.
- Download images corresponding to the annotations.
- Unzip and extract the YOLO data.

#### Usage:
1. Configure the script with:
   - `BASE_URL`: The base URL of your LabelStudio instance.
   - `PROJECT_ID`: The ID of the LabelStudio project to export.
   - `EXPORT_FORMAT`: Export format (default is `YOLO`).
   - `EXPORT_PATH`: Directory to save the exported data.

2. Run the script:
   ```bash
   python exportlabelstudio.py
   ```

3. Results:
   - Labeled data is saved in `exported_data/yolo`.
   - Associated images are downloaded and saved in `exported_data/yolo/images`.

---

### 2. `trainlabelstudio.py`
#### Purpose:
- Split the YOLO dataset into training and validation sets.
- Generate a `config.yaml` file for YOLO training.
- Train a YOLOv8 model using the prepared dataset.

#### Usage:
1. Ensure the exported data from `exportlabelstudio.py` is in `exported_data/yolo`.
2. Run the script:
   ```bash
   python trainlabelstudio.py
   ```
3. Results:
   - Data is split into `train` and `val` folders under `exported_data/yolo/images` and `exported_data/yolo/labels`.
   - A `config.yaml` file is created for YOLO training.
   - Training starts using YOLOv8 with the defined parameters.

#### Training Configuration:
- Model: `yolov8n.pt` (a pretrained YOLOv8 model).
- Training parameters:
  - Epochs: 100
  - Image size: 640
  - Batch size: 1 (adjust as needed).

---

## Notes

1. **API Rate Limiting**: The `exportlabelstudio.py` script includes a delay (`time.sleep(2)`) between image download requests to avoid overloading the server.
2. **Customization**: Modify `EXPORT_FORMAT`, `PROJECT_ID`, and other configuration variables in the scripts to suit your project.
3. **Error Handling**: If the export fails, the script prints the error response from the LabelStudio server for debugging.

---

## Folder Structure
After running both scripts, the folder structure will look like this:
```
exported_data/
└── yolo/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ├── config.yaml
    ├── classes.txt
```

---

## Troubleshooting
- **Export Fails**: Check the `API_TOKEN`, `PROJECT_ID`, and server URL (`BASE_URL`).
- **Training Issues**: Ensure `ultralytics` is installed and YOLOv8 is correctly configured.

For further assistance, refer to [LabelStudio's API Documentation](https://labelstud.io/) and [Ultralytics YOLO](https://ultralytics.com/).
