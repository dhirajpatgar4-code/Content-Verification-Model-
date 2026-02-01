#!/usr/bin/env python3
"""
Complete image model training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import json
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_image_data(dataset_path="dataset/Education_PhysicalEdu_Dataset/"):
    """Load images from your dataset"""
    print("📷 Loading image data...")
    
    # Map folder names to categories
    folder_to_category = {
        "Classroom_Learning": "education",
        "Lab_Activities": "education", 
        "Online_Learning": "education",
        "Student_Behavior": "education",
        "PE_Warmup": "sports",
        "PE_Exercises": "sports",
        "Yoga_Meditation": "health",
        "Sports_Football": "sports",
        "Sports_Basketball": "sports",
        "Sports_Cricket": "sports",
        "Sports_Athletics": "sports",
        "Fitness_Workouts": "health",
        "Stretching": "health",
        "Playground_Scenes": "sports",
        "Gym_Scenes": "sports"
    }
    
    # All categories
    categories = [
        "food", "tech", "education", "health", "finance", "fashion",
        "electronics", "automotive", "real_estate", "entertainment",
        "travel", "beauty", "home", "sports",
        "weapons", "drugs", "adult_content", "gambling", "unknown"
    ]
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    image_paths = []
    labels = []
    
    # Walk through dataset directory
    if os.path.exists(dataset_path):
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            
            if os.path.isdir(folder_path) and folder in folder_to_category:
                category = folder_to_category[folder]
                label = category_to_idx[category]
                
                # Get all image files
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(folder_path, filename)
                        image_paths.append(image_path)
                        labels.append(label)
    
    # If no images found in dataset, generate synthetic paths for testing
    if len(image_paths) == 0:
        print("⚠️ No images found in dataset. Using synthetic data for testing.")
        # Create dummy data for testing
        for category in ["education", "sports", "health"]:
            for i in range(100):
                image_paths.append(f"dummy_{category}_{i}.jpg")
                labels.append(category_to_idx[category])
    
    print(f"✅ Loaded {len(image_paths)} images")
    return image_paths, labels, categories, category_to_idx, folder_to_category

def train_image_model():
    """Train the image classification model"""
    print("🚀 Starting image model training...")
    
    # Load data
    image_paths, labels, categories, category_to_idx, folder_to_category = load_image_data()
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(categories)
    model = ImageClassifier(num_classes).to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    epochs = 10
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, train_transform, categories, category_to_idx, folder_to_category)
        
        scheduler.step()
    
    print(f"\n✅ Training complete! Best accuracy: {best_accuracy:.2f}%")

def save_model(model, transform, categories, category_to_idx, folder_to_category, save_path="ml_models/image_model/"):
    """Save the trained model"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    
    # Save transform
    with open(os.path.join(save_path, "transform.pkl"), "wb") as f:
        pickle.dump(transform, f)
    
    # Save category mappings
    with open(os.path.join(save_path, "categories.pkl"), "wb") as f:
        pickle.dump({
            'categories': categories,
            'category_to_idx': category_to_idx,
            'idx_to_category': {idx: cat for cat, idx in category_to_idx.items()},
            'folder_to_category': folder_to_category
        }, f)
    
    # Save config
    config = {
        'model_type': 'resnet18',
        'num_classes': len(categories),
        'input_size': [224, 224]
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f)
    
    print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    train_image_model()