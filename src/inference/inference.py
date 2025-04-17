import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
import onnx
import onnxruntime
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

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

# ----------- Input Source ---------------
input_source = 'webcam'  # or 'folder'

if input_source == 'webcam':
    cap = cv2.VideoCapture(0)

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
        accuracy = np.max(output)

        print(f"Prediction: {pred_class}, Confidence: {accuracy:.2f}")

        # ----------- Display the results ---------------
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f"{pred_class} ({accuracy:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Water Bottle Inspection", frame_bgr)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
