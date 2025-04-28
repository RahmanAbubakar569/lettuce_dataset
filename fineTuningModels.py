import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import drive
from collections import Counter

# Mount Google Drive
def mount_drive():
    drive.mount('/content/drive')
    print("Google Drive Mounted")

mount_drive()

# Set common device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define class names
class_names = ['Bacterial', 'Fungal', 'Healthy']

# Define transformations
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Get data loaders with WeightedRandomSampler for class imbalance
def get_dataloaders(data_dir, batch_size=32):
    # Load dataset
    dataset = {phase: datasets.ImageFolder(os.path.join(data_dir, phase), transform=transform[phase]) for phase in ['train', 'valid', 'test']}
    
    # Class distribution
    class_counts = [len([x for x in dataset['train'].imgs if x[1] == i]) for i in range(len(class_names))]
    class_weights = 1. / np.array(class_counts)
    sample_weights = [class_weights[label] for _, label in dataset['train'].imgs]
    
    # Weighted sampler for training
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoader setup
    dataloaders = {
        phase: DataLoader(
            dataset[phase],
            batch_size=batch_size,
            shuffle=False if phase == 'train' else True,  # No shuffle for training due to WeightedRandomSampler
            sampler=train_sampler if phase == 'train' else None
        )
        for phase in ['train', 'valid', 'test']
    }

    return dataloaders

# Load model function by name
def get_model(model_name):
    if model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 3)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_large(pretrained=True)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, 3)
    elif model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 3)
    else:
        raise ValueError("Model not supported.")
    return model.to(device)

# Train the model with class-weighted loss
def train_model(model, dataloaders, model_name, num_epochs=10, lr=0.001):
    class_counts = [len([x for x in dataloaders['train'].dataset.imgs if x[1] == i]) for i in range(len(class_names))]
    class_weights = torch.tensor([1.0 / count for count in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Applying class weights to loss function

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            correct, total = 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct.double() / total
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'/content/best_{model_name}.pth')
                print("Best model saved!")

        scheduler.step()

    return model

# Evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1, classification_report(all_labels, all_preds, target_names=class_names)

# Run training & testing for all models
def run_all_models(data_dir):
    models_to_test = ['efficientnet', 'mobilenet', 'shufflenet', 'resnet50', 'vit']
    results = {}

    for model_name in models_to_test:
        print(f"\nTraining and Evaluating {model_name.upper()}")
        dataloaders = get_dataloaders(data_dir)
        model = get_model(model_name)
        model = train_model(model, dataloaders, model_name)
        acc, prec, rec, f1, report = evaluate_model(model, dataloaders['test'])

        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'report': report
        }

    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"\nModel: {name.upper()}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print("\nClassification Report:\n", metrics['report'])

# Start the full pipeline
data_path = '/content/drive/My Drive/lettuce_disease_images_2/lettuce'
run_all_models(data_path)