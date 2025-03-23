import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm.auto import tqdm  # Import tqdm for progress bars
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

def load_dataset(test_path="./sampled_test", batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for ResNet
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.133, 0.133, 0.133], std=[0.292, 0.292, 0.292])  # Normalize for 3 channels
    ])
    test_dataset = ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset.classes

def modify_resnet_for_mnist():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Modify output layer for 10 classes
    return model

def evaluate_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    print("Confusion matrix saved as 'confusion_matrix.png'")
    print("Classification report saved as 'classification_report.json'")

def main():
    test_loader, class_names = load_dataset()
    model = modify_resnet_for_mnist()  # Initialize model once
    state_dict = torch.load("resnet50_mnist.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state_dict)  # Load weights into the same instance
    evaluate_model(model, test_loader, class_names)

if __name__ == "__main__":
    main()
