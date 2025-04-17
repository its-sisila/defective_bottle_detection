
import os
import torch
from torchvision import models, transforms
from PIL import Image
import onnx
import onnxruntime
import torchvision.transforms as transforms
accP=0
accD=0
# Load the ONNX model
model = onnx.load(
    r"C:\Projects\tests\water-bottle-visual-inspection-based-on-transfer-learning-main\models\best_model\best model\best_model.onnx")

# Create a PyTorch model from the ONNX model
pytorch_model = onnxruntime.InferenceSession(
    r"C:\Projects\tests\water-bottle-visual-inspection-based-on-transfer-learning-main\models\best_model\best model\best_model.onnx")
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

# Loop through the images in the folder
folder_path = r"C:\Projects\tests\water-bottle-visual-inspection-based-on-transfer-learning-main\data\dataset\dataset\Test\Defective"
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image
        img_path = os.path.join(folder_path, filename)
        photo = Image.open(img_path)

        # Preprocess the image and perform inference
        with torch.no_grad():
            photo_tensor = preprocess(photo).unsqueeze(0)
            output = pytorch_model.run([], {input_name: photo_tensor.numpy()})[0]

        # Get the index of the highest probability for the output
        pred = output.argmax()

        # Look up the corresponding class name
        class_name = class_names[pred]
        if class_name=='D':
            accD=accD+1
        else:
            accP=accP+1
        # Print the result
        print(filename, class_name, 'acc=', max(output[0]))
print( 'accd=', accD)
print( 'accp=', accP)