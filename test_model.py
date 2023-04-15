"""
    This is a dummy program for test_model.py

    It includes necessary functionalities for user, for example: 
        - Applied argument to allow users to set up some hyper parameters
        - Received a folder of image input
        - Applied trained model to perform the prediciton
        - Save the prediction result to a text file

    To run the test_model.py, you can run the program using the following command:

        python test_model.py --model_path /path/to/pretrained/model.pth 
                                --image_folder_path /path/to/folder/with/images 
                                --output_file_path /path/to/output.txt


"""


import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Define the command line arguments
parser = argparse.ArgumentParser(description='Testing a pre-trained PyTorch model on a folder of images')
parser.add_argument('--model_path', type=str, help='path to the pre-trained PyTorch model')
parser.add_argument('--image_folder_path', type=str, help='path to the folder with images to be tested')
parser.add_argument('--output_file_path', type=str, help='path to the output text file')
args = parser.parse_args()

# Load the pre-trained model
model = torch.load(args.model_path)

# Define the image preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(args.target_size),
    transforms.CenterCrop(args.target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get a list of all the images in the folder
filelist = os.listdir(args.image_folder_path)
image_paths = [os.path.join(args.image_folder_path, file) for file in filelist]

# Create an empty list to store the results
results = []

# Loop through each image in the folder and apply the model
for image_path in image_paths:
    # Load the image and preprocess it
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    
    # Make a prediction using the model
    with torch.no_grad(): prediction = model(image)
    
    # Add the result to the list of results
    results.append(prediction.item())
    
# Save the results to a text file
with open(args.output_file_path, 'w') as f:
    for result in results:
        f.write(str(result) + '\n')



