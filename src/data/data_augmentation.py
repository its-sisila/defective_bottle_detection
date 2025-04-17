import cv2
import os
import numpy as np
import albumentations as A
from torchvision.transforms import ColorJitter
from PIL import Image

# define paths
original_dir = r"C:\Users\Sisila\Downloads\Compressed\conv"
augmented_dir = r"C:\Users\Sisila\Downloads\Compressed\aug_no cap"

# define augmentation functions
def luminosity(image):
    # randomly generate a value between 0.5 and 1.5
    alpha = np.random.uniform(0.6, 0.8)
    # apply the value to the image
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def color_filter(image):
    brightness = np.random.uniform(0.6, 0.8)
    contrast = np.random.uniform(0.4, 1.2)
    saturation = np.random.uniform(0.4, 1.1)
    hue = abs(np.random.uniform(-0.3, 0.3))
    transform = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    image = Image.fromarray(image)
    return np.array(transform(image))

def rotate(image):
    angle = np.random.randint(-80, 80)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))

def stretch(image):
    scale = np.random.uniform(3, 3.1)
    height, width = image.shape[:2]
    new_height = int(scale * height)
    return cv2.resize(image, (width, new_height))

def contrast(image):
    alpha = np.random.uniform(0.2, 0.21)
    beta = np.random.randint(-90, 90)

    # add random cropping to the contrast function
    crop_height = np.random.randint(50, 70)
    crop_width = np.random.randint(75, 85)
    aug = A.Compose([
        A.Crop(x_min=crop_width, y_min=crop_height, x_max=image.shape[1], y_max=image.shape[0])
    ])
    image = aug(image=image)['image']

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def blur(image):
    kernel_size = np.random.choice([7, 9, 11])

    # add elastic transformation to the blurring function
    elastic = A.Compose([A.ElasticTransform(p=1, alpha=40, sigma=120 * 0.05, alpha_affine=120 * 0.03)])
    image = elastic(image=image)['image']

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def erase(image):
    # generate a random mask
    mask = np.random.choice([0.9, 0.95], size=image.shape[:2], p=[0.9, 0.1])

    # convert mask to uint8
    mask = mask.astype(np.uint8)

    # apply the mask to the image
    return cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask)

def noise(image):
    mean = -0.5
    var = np.random.uniform(50, 55)
    sigma = var ** 1
    noise = np.random.normal(mean, sigma, image.shape)
    return cv2.add(image, noise.astype(np.uint8))

# add cutout function
def cutout(image):
    height, width = image.shape[:2]
    mask = np.ones((height, width), np.uint8)
    num_rectangles = np.random.randint(11, 12) # choose a random number of rectangles to cut out
    for i in range(num_rectangles):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        w = np.random.randint(50, 60)
        h = np.random.randint(50, 60)
        x1 = np.clip(x - w//2, 0, width)
        y1 = np.clip(y - h//2, 0, height)
        x2 = np.clip(x + w//2, 0, width)
        y2 = np.clip(y + h//2, 0, height)
        mask[y1:y2, x1:x2] = 0
    return cv2.bitwise_and(image, image, mask=mask)


def zoom(image):
    height, width = image.shape[:2]

    # generate a random scale factor between 0.5 and 1.5
    scale_factor = np.random.uniform(0.5, 1.5)

    # ensure that the scale factor is greater than 1
    if scale_factor <= 1:
        scale_factor = 1.1

    # calculate the new height and width of the zoomed image
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)

    # check that new_width is greater than 0
    if new_width <= 0:
        new_width = 1

    # create a new image with the zoomed dimensions
    zoomed = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # calculate the offset for the zoomed image
    x_offset = np.random.randint(0, new_width - width)
    y_offset = np.random.randint(0, new_height - height)

    # copy the original image to the zoomed image with the offset
    zoomed[y_offset:y_offset+height, x_offset:x_offset+width] = image

    # resize the zoomed image to the original dimensions
    zoomed = cv2.resize(zoomed, (width, height))

    return zoomed





# define list of augmentation functions
augmentations = [rotate, stretch, luminosity, color_filter, blur,noise,zoom]

# create directory for augmented images
os.makedirs(augmented_dir, exist_ok=True)

# create list of valid file extensions
valid_extensions = ['.jpg', '.jpeg', '.png']

# generate augmented images
for filename in os.listdir(original_dir):
    # check if the file has a valid extension
    if os.path.splitext(filename)[1] in valid_extensions:
        # load image
        img = cv2.imread(os.path.join(original_dir, filename))

        # generate and save augmented images
        for i, func in enumerate(augmentations):
            augmented_img = func(img.copy())
            cv2.imwrite(os.path.join(augmented_dir, f"{os.path.splitext(filename)[0]}_aug_{i}.jpeg"), augmented_img)
