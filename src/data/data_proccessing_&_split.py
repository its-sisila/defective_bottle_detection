import os
import cv2
import numpy as np
import random

# Set the directory paths
dir_path = r'E:\DATA\My Data\aug PROPRE'  # Adjust as needed
train_path = r'E:\DATA\mido\Train\Propre'  # Adjust as needed
valid_path = r'E:\DATA\mido\Val\Propre'  # Adjust as needed
test_path = r'E:\DATA\mido\Test\Propre'  # Adjust as needed

# Set the target size for resizing the images
target_size = (224, 224)

# Set the ratios for splitting the data
train_ratio = 0.75
valid_ratio = 0.15

# Create empty lists to hold the image filenames and labels
image_filenames = []
labels = []

# Loop through the images in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_filenames.append(os.path.join(dir_path, filename))
        labels.append("propre")

# Shuffle the image filenames and labels together
combined = list(zip(image_filenames, labels))
random.shuffle(combined)
image_filenames, labels = zip(*combined)

# Compute the number of images for each split
num_train = int(len(image_filenames) * train_ratio)
num_valid = int(len(image_filenames) * valid_ratio)
num_test = len(image_filenames) - num_train - num_valid

# Create the output directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Load, resize, normalize, and save images to respective directories
for i, (filename, label) in enumerate(zip(image_filenames, labels)):
    img = cv2.imread(filename)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.

    if i < num_train:
        cv2.imwrite(os.path.join(train_path, f'{label}_{i}.jpg'), img*255)
    elif i < num_train + num_valid:
        cv2.imwrite(os.path.join(valid_path, f'{label}_{i}.jpg'), img*255)
    else:
        cv2.imwrite(os.path.join(test_path, f'{label}_{i}.jpg'), img*255)

    if i % 100 == 0:
        print(f'{i} images processed...')
