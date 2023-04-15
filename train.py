# 1. Import the required libraries
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Add argparse to get the data path from the command-line argument
parser = argparse.ArgumentParser(description='Car Classification Training Script')
parser.add_argument('--data_path', type=str, help='Path to the dataset directory', required=True)
args = parser.parse_args()

data_path = args.data_path

# Create a mapping between class names and integer labels
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

# 2. Prepare the dataset and create data loaders
class CarDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(txt_file, sep=":", names=["class", "count"])
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for _, row in self.labels_frame.iterrows():
            class_name = str(row['class']).strip()  # Strip leading and trailing whitespaces
            class_dir = os.path.join(root_dir, class_name)
            images = os.listdir(class_dir)[:row['count']]
            for img in images:
                self.image_paths.append(os.path.join(class_dir, img))
                self.labels.append(class_mapping[class_name])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Modify the dataset initialization to use the data_path variable:
train_dataset = CarDataset(txt_file=os.path.join(data_path, 'train.txt'), root_dir=os.path.join(data_path, 'train'), transform=data_transforms['train'])
val_dataset = CarDataset(txt_file=os.path.join(data_path, 'val.txt'), root_dir=os.path.join(data_path, 'val'), transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

# 3. Define the neural network architecture
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 20)  # 20 classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    # 5. Train the model
    num_epochs = 100
    model_save_dir = 'models'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model_save_path = os.path.join(model_save_dir, f'resnet50_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at {model_save_path}')

        model.eval()
        correct = 0
        total = 0

        # 6. Evaluate the model
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Accuracy on the validation set (Epoch {epoch+1}): {100 * correct / total}%')
