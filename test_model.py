import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import argparse

class_mapping = {
    'Golf': 0,
    'bmw serie 1': 1,
    'chevrolet spark': 2,
    'chevroulet aveo': 3,
    'clio': 4,
    'duster': 5,
    'hyundai i10': 6,
    'hyundai tucson': 7,
    'logan': 8,
    'megane': 9,
    'mercedes class a': 10,
    'nemo citroen': 11,
    'octavia': 12,
    'picanto': 13,
    'polo': 14,
    'sandero': 15,
    'seat ibiza': 16,
    'symbol': 17,
    'toyota corolla': 18,
    'volkswagen tiguan': 19
}

reverse_class_mapping = {v: k for k, v in class_mapping.items()}

# CarTestDataset class
class CarTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.class_names = os.listdir(root_dir)

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            images = os.listdir(class_dir)
            for img in images:
                self.image_paths.append(os.path.join(class_dir, img))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, img_name

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def save_results(saved_result_path, image_names, predicted_labels, reverse_class_mapping):
    with open(saved_result_path, 'w') as f:
        for img_name, label in zip(image_names, predicted_labels):
            class_name = reverse_class_mapping[label]
            f.write(f'{img_name}: {class_name}\n')

reverse_class_mapping = {v: k for k, v in class_mapping.items()}

# CarDataset class
class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.class_names = os.listdir(root_dir)

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            images = os.listdir(class_dir)
            for img in images:
                self.image_paths.append(os.path.join(class_dir, img))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, img_name

def main(test_data_path, trained_model_path, saved_result_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 20)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)

    # Create a test dataset and data loader
    test_dataset = CarTestDataset(test_data_path, transform=data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model.eval()
    image_names = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, img_name in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            image_names.append(img_name[0])
            predicted_labels.append(predicted.item())

    save_results(saved_result_path, image_names, predicted_labels, reverse_class_mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data_path", help="Path to the test data folder")
    parser.add_argument("trained_model_path", help="Path to the trained model file")
    parser.add_argument("saved_result_path", help="Path to save the classification results")
    args = parser.parse_args()

    main(args.test_data_path, args.trained_model_path, args.saved_result_path)
