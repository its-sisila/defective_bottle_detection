import os
import torch
from torchvision import models, transforms
from PIL import Image
import onnx
import onnxruntime
import torchvision.transforms as transforms

# Load the ONNX model
model_path = r"C:\Projects\tests\water-bottle-visual-inspection-based-on-transfer-learning-main\models\best_model\best model\best_model.onnx"
model = onnx.load(model_path)

# Create an ONNX inference session
pytorch_model = onnxruntime.InferenceSession(model_path)
input_name = pytorch_model.get_inputs()[0].name
output_name = pytorch_model.get_outputs()[0].name

# Define the image preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the list of class names
class_names = ['D', 'P']

# Test function to evaluate images from a folder
def test_folder(folder_path, expected_class):
    total = 0
    correct = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Open the image
            img_path = os.path.join(folder_path, filename)
            photo = Image.open(img_path)
            
            # Preprocess the image and perform inference
            with torch.no_grad():
                photo_tensor = preprocess(photo).unsqueeze(0)
                output = pytorch_model.run([output_name], {input_name: photo_tensor.numpy()})[0]
            
            # Get the index of the highest probability for the output
            pred = output.argmax()
            
            # Look up the corresponding class name
            class_name = class_names[pred]
            accuracy = max(output[0])
            
            # Count correct predictions
            total += 1
            if class_name == expected_class:
                correct += 1
                
            # Print the result
            print(f"{filename}: Predicted {class_name}, Confidence {accuracy:.4f}")
    
    # Return accuracy statistics
    return correct, total

# Define the paths for both classes
propre_path = r"C:\Projects\tests\water-bottle-visual-inspection-based-on-transfer-learning-main\data\dataset\dataset\Test\Propre"
defective_path = r"C:\Projects\tests\water-bottle-visual-inspection-based-on-transfer-learning-main\data\dataset\dataset\Test\Defective"

# Test on Propre (proper) bottles
print("\n===== Testing Proper Bottles =====")
correct_proper, total_proper = test_folder(propre_path, 'P')
print(f"Proper bottles accuracy: {correct_proper}/{total_proper} = {correct_proper/total_proper*100:.2f}%")

# Test on Defective bottles
print("\n===== Testing Defective Bottles =====")
correct_defective, total_defective = test_folder(defective_path, 'D')
print(f"Defective bottles accuracy: {correct_defective}/{total_defective} = {correct_defective/total_defective*100:.2f}%")

# Overall accuracy
total_overall = total_proper + total_defective
correct_overall = correct_proper + correct_defective
print("\n===== Overall Results =====")
print(f"Overall accuracy: {correct_overall}/{total_overall} = {correct_overall/total_overall*100:.2f}%")