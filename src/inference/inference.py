import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
import onnx
import onnxruntime
import numpy as np
import datetime
import requests
import threading
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ----------- Configuration Options ---------------
# Set threshold for capturing defective bottles
CONFIDENCE_THRESHOLD = 0.95  # Only save images with confidence above this value

# Directory to save high-confidence defective bottle images
SAVE_DIR = os.path.join(os.path.dirname(__file__), "high_confidence_defects")
os.makedirs(SAVE_DIR, exist_ok=True)

# OneDrive configuration
ONEDRIVE_FOLDER = "DefectiveBottles"  # Folder in OneDrive to store images
PERSONAL_ACCESS_TOKEN = ""  # Replace with your token here, otherswise it will not upload to OneDrive

# ----------- Load the ONNX model ---------------
onnx_model_path = r"C:\Projects\tests\water-bottle-visual-inspection-based-on-transfer-learning-main\models\best_model\best model\best_model.onnx"

# Optional ONNX check
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

# ----------- Create ONNX Runtime Inference Session ---------------
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

# Set provider (CPU only since GPU not available)
providers = ['CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(onnx_model_path, options, providers=providers)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# ----------- Define the image preprocessing transforms ---------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ['Defective', 'Propre']  # Change as per your dataset

# ----------- OneDrive Upload Function ---------------
def upload_to_onedrive(file_path, file_name):
    """
    Upload a file to OneDrive using Microsoft Graph API
    """
    try:
        # Read the image file
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        # Set headers with access token
        headers = {
            'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'image/jpeg'
        }
        
        # Upload URL (to a specific folder)
        upload_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{ONEDRIVE_FOLDER}/{file_name}:/content"
        
        # Make the PUT request to upload
        response = requests.put(upload_url, headers=headers, data=file_data)
        
        if response.status_code in (200, 201):
            print(f"✓ Successfully uploaded {file_name} to OneDrive/{ONEDRIVE_FOLDER}")
            # Optionally, you can get the file metadata from the response
            file_metadata = json.loads(response.text)
            return True
        else:
            print(f"✗ Upload failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"✗ Error uploading {file_name}: {str(e)}")
        return False

# ----------- Create OneDrive Folder (if it doesn't exist) ---------------
def ensure_onedrive_folder_exists():
    """Ensure the target folder exists in OneDrive"""
    try:
        headers = {
            'Authorization': f'Bearer {PERSONAL_ACCESS_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        # Check if folder exists
        folder_check_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{ONEDRIVE_FOLDER}"
        response = requests.get(folder_check_url, headers=headers)
        
        if response.status_code == 404:
            # Folder doesn't exist, create it
            create_folder_url = "https://graph.microsoft.com/v1.0/me/drive/root/children"
            folder_data = {
                "name": ONEDRIVE_FOLDER,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "rename"
            }
            response = requests.post(create_folder_url, headers=headers, json=folder_data)
            
            if response.status_code in (201, 200):
                print(f"Created folder '{ONEDRIVE_FOLDER}' in OneDrive")
                return True
            else:
                print(f"Failed to create OneDrive folder: {response.status_code}")
                print(response.text)
                return False
        return True
    except Exception as e:
        print(f"Error checking/creating OneDrive folder: {str(e)}")
        return False

# ----------- Input Source ---------------
input_source = 'webcam'  # or 'folder'

if input_source == 'webcam':
    # Try to create the OneDrive folder if it doesn't exist
    if PERSONAL_ACCESS_TOKEN != "":
        ensure_onedrive_folder_exists()
    else:
        print("WARNING: You need to set your OneDrive personal access token to enable uploads")
        
    cap = cv2.VideoCapture(0)
    
    # For tracking captured defective images
    captured_defects = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = preprocess(frame_rgb).unsqueeze(0)

        # ONNX runtime works with NumPy arrays, no need to move tensor to CUDA
        input_data = frame_tensor.numpy()

        # ----------- Inference ---------------
        output = ort_session.run([output_name], {input_name: input_data})[0]

        # Get the predicted class and accuracy
        pred_idx = np.argmax(output)
        pred_class = class_names[pred_idx]
        confidence = float(np.max(output))
        
        # ----------- Capture high-confidence defective bottles ---------------
        if pred_class == 'Defective' and confidence >= CONFIDENCE_THRESHOLD:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"defect_{confidence:.2f}_{timestamp}.jpg"
            file_path = os.path.join(SAVE_DIR, file_name)
            
            # Save the image locally
            cv2.imwrite(file_path, frame)
            print(f"Saved high-confidence defect ({confidence:.2f}) to {file_path}")
            
            # Upload to OneDrive in a separate thread (if token is set)
            if PERSONAL_ACCESS_TOKEN != "":
                upload_thread = threading.Thread(
                    target=upload_to_onedrive,
                    args=(file_path, file_name)
                )
                upload_thread.daemon = True
                upload_thread.start()
                captured_defects[file_name] = "Uploading"
            
        # ----------- Display the results ---------------
        display_frame = frame.copy()
        
        # Different color based on class
        if pred_class == 'Defective':
            color = (0, 0, 255)  # Red for defective
            if confidence >= CONFIDENCE_THRESHOLD:
                # Add special marker for high-confidence defects
                cv2.putText(display_frame, "HIGH CONFIDENCE DEFECT!", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            color = (0, 255, 0)  # Green for proper
            
        # Display prediction and confidence
        cv2.putText(display_frame, f"{pred_class} ({confidence:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Water Bottle Inspection", display_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
elif input_source == 'folder':
    # Code for folder-based processing could be added here
    pass
