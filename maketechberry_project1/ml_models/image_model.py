import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    """Dataset for image classification"""
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
    """CNN-based image classifier"""
    def __init__(self, num_classes, pretrained=True):
        super(ImageClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final layer
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

class ImageModelTrainer:
    """Train and manage image classification model"""
    
    def __init__(self, dataset_path="dataset/Education_PhysicalEdu_Dataset/"):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Category mapping from folder names
        self.folder_to_category = {
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
        
        # All categories (19 total)
        self.categories = [
            "food", "tech", "education", "health", "finance", "fashion",
            "electronics", "automotive", "real_estate", "entertainment",
            "travel", "beauty", "home", "sports",
            "weapons", "drugs", "adult_content", "gambling", "unknown"
        ]
        
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for idx, cat in enumerate(self.categories)}
        
        # Initialize model
        self.num_classes = len(self.categories)
        self.model = ImageClassifier(self.num_classes).to(self.device)
    
    def load_image_data(self):
        """Load images from your dataset"""
        print("📷 Loading image data from your dataset...")
        
        image_paths = []
        labels = []
        
        # Walk through your dataset directory
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)
            
            if os.path.isdir(folder_path) and folder in self.folder_to_category:
                category = self.folder_to_category[folder]
                label = self.category_to_idx[category]
                
                # Get all image files
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(folder_path, filename)
                        image_paths.append(image_path)
                        labels.append(label)
        
        print(f"✅ Loaded {len(image_paths)} images from your dataset")
        return image_paths, labels
    
    def train(self, epochs=20, batch_size=32, learning_rate=0.001):
        """Train the image classification model"""
        print("🚀 Starting image model training...")
        
        # Load data
        image_paths, labels = self.load_image_data()
        
        if len(image_paths) == 0:
            print("❌ No images found in dataset!")
            return
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = ImageDataset(train_paths, train_labels, self.transform)
        val_dataset = ImageDataset(val_paths, val_labels, self.transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Training loop
        best_accuracy = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Training phase
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model()
            
            scheduler.step()
        
        print(f"✅ Training completed! Best accuracy: {best_accuracy:.2f}%")
    
    def save_model(self, save_path="ml_models/image_model/"):
        """Save the trained model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pth"))
        
        # Save category mappings
        with open(os.path.join(save_path, "categories.pkl"), "wb") as f:
            pickle.dump({
                'categories': self.categories,
                'category_to_idx': self.category_to_idx,
                'idx_to_category': self.idx_to_category,
                'folder_to_category': self.folder_to_category
            }, f)
        
        # Save transform
        with open(os.path.join(save_path, "transform.pkl"), "wb") as f:
            pickle.dump(self.transform, f)
        
        print(f"💾 Image model saved to {save_path}")
    
    def load_model(self, model_path="ml_models/image_model/"):
        """Load trained model"""
        # Load category mappings
        with open(os.path.join(model_path, "categories.pkl"), "rb") as f:
            mappings = pickle.load(f)
            self.categories = mappings['categories']
            self.category_to_idx = mappings['category_to_idx']
            self.idx_to_category = mappings['idx_to_category']
            self.folder_to_category = mappings['folder_to_category']
        
        # Load model
        self.model.load_state_dict(torch.load(
            os.path.join(model_path, "model.pth"),
            map_location=self.device
        ))
        self.model.eval()
        
        # Load transform
        with open(os.path.join(model_path, "transform.pkl"), "rb") as f:
            self.transform = pickle.load(f)
        
        print("✅ Image model loaded successfully")
    
    def predict(self, image_path, top_k=3):
        """Predict category for given image"""
        self.model.eval()
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get top k predictions
            top_conf, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to readable format
        category = self.idx_to_category[predicted_idx.item()]
        
        top_categories = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            top_categories.append({
                'category': self.idx_to_category[idx],
                'confidence': top_conf[0][i].item()
            })
        
        # Check if restricted
        is_restricted = category in ["weapons", "drugs", "adult_content", "gambling"]
        
        return {
            'category': category,
            'confidence': confidence.item(),
            'is_restricted': is_restricted,
            'top_categories': top_categories,
            'model_used': 'resnet_image_classifier'
        }