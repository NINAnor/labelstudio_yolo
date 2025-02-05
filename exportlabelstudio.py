import requests
import os
import time
from authenticate import API_TOKEN

# Configuration
BASE_URL = "https://labelstudio.nina.no/api"
PROJECT_ID = 16  # Replace with your project ID
EXPORT_FORMAT = "YOLO"  # YOLO format
EXPORT_PATH = "exported_data/yolo"  # Directory to save the YOLO data

# Ensure the export directory exists
if not os.path.exists(EXPORT_PATH):
    os.makedirs(EXPORT_PATH)

# Endpoint for exporting project data
export_url = f"{BASE_URL}/projects/{PROJECT_ID}/export?exportType={EXPORT_FORMAT}"

# Headers for authentication
headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

# Make the API request to export data in YOLO format
print("Exporting data in YOLO format...")
response = requests.get(export_url, headers=headers)

if response.status_code == 200:
    # Save the exported YOLO data as a zip file
    yolo_zip_file = os.path.join(EXPORT_PATH, f"project_{PROJECT_ID}_yolo.zip")
    with open(yolo_zip_file, "wb") as file:
        file.write(response.content)
    print(f"YOLO data exported successfully to {yolo_zip_file}")

    # Optional: Unzip the YOLO data
    import zipfile
    with zipfile.ZipFile(yolo_zip_file, 'r') as zip_ref:
        zip_ref.extractall(EXPORT_PATH)
    print(f"YOLO data extracted to {EXPORT_PATH}")
else:
    print(f"Failed to export data in YOLO format. Status code: {response.status_code}")
    print("Response:", response.text)



# after downloading the data, we have to also download images as they don't come with the data on our server...
# URL of the file to download
url = f'https://labelstudio.nina.no/data/upload/{PROJECT_ID}/'

# Headers with Authorization token
headers = {
    "Authorization": f"Token {API_TOKEN}"
}

files = os.listdir(os.path.join(EXPORT_PATH, 'labels'))
files = [file.replace('.txt', '.jpg') for file in files]
#files = [file.replace('.txt', '.mp4') for file in files]

for file in files:
    # Sending a GET request to download the file
    response = requests.get(url + file, headers=headers, allow_redirects=True)

    # Checking if the request was successful
    if response.status_code == 200:
        # Saving the file locally
        with open(os.path.join(EXPORT_PATH, 'images', file), 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as '{file}'")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    
    # to avoid breaking the server we have to wait a bit between the requests
    # without this the server will go down until it restarts
    time.sleep(1) # 1 seconds delay between requests is probably overkill, but better safe than sorry