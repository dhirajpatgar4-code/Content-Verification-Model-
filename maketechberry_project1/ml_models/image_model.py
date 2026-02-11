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
import matplotlib.pyplot as plt

class EnhancedImageDataset(Dataset):
    """Enhanced dataset with data augmentation"""
    def __init__(self, image_paths, labels, transform=None, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        
        # Base transform (always applied)
        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transform
        self.aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]) if augment else self.base_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.augment:
                image = self.aug_transform(image)
            else:
                image = self.base_transform(image)
            
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            # Return a placeholder if image loading fails
            print(f"Error loading image {image_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            label = self.labels[idx]
            return placeholder, label

class LightweightImageClassifier(nn.Module):
    """More appropriate model for smaller dataset"""
    def __init__(self, num_classes, pretrained=True):
        super(LightweightImageClassifier, self).__init__()
        
        # Use ResNet18 but freeze early layers
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class SimpleCNN(nn.Module):
    """Simple CNN for smaller datasets"""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EnhancedImageModelTrainer:
    """Enhanced trainer for image classification"""
    
    def __init__(self, dataset_path=None):
        # Set default path relative to project root
        if dataset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            dataset_path = os.path.join(project_root, "dataset")
        
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Dataset path: {self.dataset_path}")
        
        # Define categories based on YOUR actual data
        # Only include categories you actually have in your dataset
        self.categories = [
            "education",  # From your dataset
            "sports",     # From your dataset  
            "health"      # From your dataset
        ]
        
        # Map ACTUAL folder names to categories (based on your real dataset structure)
        self.folder_to_category = {
            "all_subject_based_learning": "education",      # ✅ Has 96 images
            "all_indoor_outdoor_sports": "sports",           # ✅ Has 96 images
            "yoga_meditation": "health",                      # Health category
            # Legacy folder mappings (kept for compatibility)
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
        
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for idx, cat in enumerate(self.categories)}
        
        # Initialize model - using simpler model
        self.num_classes = len(self.categories)
        self.model = SimpleCNN(self.num_classes).to(self.device)
        print(f"Model initialized with {self.num_classes} classes: {self.categories}")
    
    def analyze_dataset(self):
        """Analyze dataset distribution"""
        print("\n📊 Analyzing dataset...")
        
        category_counts = {cat: 0 for cat in self.categories}
        
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)
            
            if os.path.isdir(folder_path) and folder in self.folder_to_category:
                category = self.folder_to_category[folder]
                image_count = len([f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                category_counts[category] += image_count
                print(f"  {folder}: {image_count} images -> {category}")
        
        print(f"\n📈 Total images by category:")
        for cat, count in category_counts.items():
            print(f"  {cat}: {count} images")
        
        return category_counts
    
    def load_image_data(self):
        """Load images from your dataset"""
        print("\n📷 Loading image data...")
        
        image_paths = []
        labels = []
        
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)
            
            if os.path.isdir(folder_path) and folder in self.folder_to_category:
                category = self.folder_to_category[folder]
                label = self.category_to_idx[category]
                
                # Get all image files
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for filename in image_files:
                    image_path = os.path.join(folder_path, filename)
                    image_paths.append(image_path)
                    labels.append(label)
        
        print(f"✅ Loaded {len(image_paths)} images")
        print(f"📊 Distribution: {np.bincount(labels)}")
        
        return image_paths, labels
    
    def visualize_samples(self, num_samples=9):
        """Visualize sample images"""
        image_paths, labels = self.load_image_data()
        
        plt.figure(figsize=(12, 12))
        for i in range(min(num_samples, len(image_paths))):
            plt.subplot(3, 3, i + 1)
            try:
                img = Image.open(image_paths[i])
                plt.imshow(img)
                plt.title(f"Label: {self.idx_to_category[labels[i]]}")
                plt.axis('off')
            except:
                plt.title("Error loading image")
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def train(self, epochs=30, batch_size=16, learning_rate=0.001):
        """Train the image classification model"""
        print("\n🚀 Starting image model training...")
        
        # Load data
        image_paths, labels = self.load_image_data()
        
        if len(image_paths) == 0:
            print("❌ No images found in dataset!")
            return
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        
        # Create datasets
        train_dataset = EnhancedImageDataset(train_paths, train_labels, augment=True)
        val_dataset = EnhancedImageDataset(val_paths, val_labels, augment=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              num_workers=2, pin_memory=True)
        
        # Training setup
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Calculate class weights for imbalanced dataset
        class_weights = self.calculate_class_weights(train_labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Training loop
        best_accuracy = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Training phase
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in train_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * (predicted == labels).sum().item() / labels.size(0)
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            with torch.no_grad():
                for images, labels in val_bar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_bar.set_postfix({
                        'loss': loss.item(),
                        'acc': 100 * (predicted == labels).sum().item() / labels.size(0)
                    })
            
            # Calculate metrics
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss/len(train_loader))
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(val_loss/len(val_loader))
            history['val_acc'].append(val_accuracy)
            
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, Accuracy: {train_accuracy:.2f}%")
            print(f"  Val   - Loss: {history['val_loss'][-1]:.4f}, Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model()
                print(f"  💾 Saved new best model with accuracy: {val_accuracy:.2f}%")
            
            scheduler.step()
        
        print(f"\n✅ Training completed! Best accuracy: {best_accuracy:.2f}%")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, save_path="ml_models/enhanced_image_model/"):
        """Save the trained model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'categories': self.categories,
            'category_to_idx': self.category_to_idx,
            'idx_to_category': self.idx_to_category,
            'folder_to_category': self.folder_to_category
        }, os.path.join(save_path, "model.pth"))
        
        print(f"💾 Model saved to {save_path}")
    
    def load_model(self, model_path="ml_models/enhanced_image_model/"):
        """Load trained model"""
        checkpoint = torch.load(os.path.join(model_path, "model.pth"), 
                              map_location=self.device)
        
        # Load mappings
        self.categories = checkpoint['categories']
        self.category_to_idx = checkpoint['category_to_idx']
        self.idx_to_category = checkpoint['idx_to_category']
        self.folder_to_category = checkpoint['folder_to_category']
        
        # Reinitialize model with correct number of classes
        self.num_classes = len(self.categories)
        self.model = SimpleCNN(self.num_classes).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✅ Image model loaded successfully")
        print(f"📊 Loaded {self.num_classes} categories: {self.categories}")
    
    def predict(self, image_path, top_k=3):
        """Predict category for given image"""
        self.model.eval()
        
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'is_restricted': False,
                'error': 'Image file not found',
                'top_categories': []
            }
        
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            
            # Apply same transform as validation
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Get top k predictions
                top_conf, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
            
            # Convert to readable format
            category = self.idx_to_category[predicted_idx.item()]
            
            top_categories = []
            for i in range(min(top_k, self.num_classes)):
                idx = top_indices[0][i].item()
                top_categories.append({
                    'category': self.idx_to_category[idx],
                    'confidence': top_conf[0][i].item()
                })
            
            # Check if restricted (though your dataset doesn't have restricted categories)
            is_restricted = False
            
            return {
                'category': category,
                'confidence': confidence.item(),
                'is_restricted': is_restricted,
                'top_categories': top_categories,
                'model_used': 'enhanced_image_classifier'
            }
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'is_restricted': False,
                'error': str(e),
                'top_categories': []
            }
    
    def test_model(self, test_dir=None):
        """Test model on sample images"""
        if test_dir is None:
            test_dir = self.dataset_path
        
        print("\n🧪 Testing model on sample images...")
        
        test_images = []
        for folder in os.listdir(test_dir):
            folder_path = os.path.join(test_dir, folder)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    test_images.append(os.path.join(folder_path, image_files[0]))
        
        for img_path in test_images[:5]:  # Test on first 5 images
            result = self.predict(img_path)
            folder_name = os.path.basename(os.path.dirname(img_path))
            expected_category = self.folder_to_category.get(folder_name, 'unknown')
            
            print(f"\n📸 Image: {os.path.basename(img_path)}")
            print(f"  Expected: {expected_category}")
            print(f"  Predicted: {result['category']} ({result['confidence']:.2%})")
            print(f"  Top predictions:")
            for pred in result['top_categories']:
                print(f"    - {pred['category']}: {pred['confidence']:.2%}")

# Usage example
if __name__ == "__main__":
    # Initialize trainer
    trainer = EnhancedImageModelTrainer("dataset/Education_PhysicalEdu_Dataset/")
    
    # Analyze dataset
    trainer.analyze_dataset()
    
    # Train model
    trainer.train(epochs=30, batch_size=16, learning_rate=0.001)
    
    # Test model
    trainer.test_model()
    
    # Save final model
    trainer.save_model()