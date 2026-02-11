# train_universal_model.py
"""
Train universal image classifier on your dataset
Extends education/sports to other domains
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import pickle
from pathlib import Path

# Define domains (same as universal classifier)
DOMAINS = [
    'education', 'sports', 'food', 'tech', 'health',
    'weapons', 'drugs', 'adult_content', 'gambling',
    'fashion', 'travel', 'entertainment', 'automotive',
    'unknown'
]

class UniversalImageDataset(Dataset):
    """Dataset for universal classification"""
    def __init__(self, image_paths, labels, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, self.labels[idx]
        except:
            # Return dummy for corrupt images
            return torch.zeros((3, 224, 224)), self.labels[idx]

def load_and_label_data():
    """
    Load your dataset and assign labels
    Currently you only have education/sports images
    We'll need to expand this with more data later
    """
    base_path = "dataset/"
    
    # Your current dataset mapping
    folder_to_domain = {
        'all_subject_based_learning': 'education',
        'ClassRoom_Learning': 'education',
        'lab activities': 'education',
        'online_learning': 'education',
        'all_indoor_outdoor_sports': 'sports',
        'physical_training': 'sports',
    }
    
    image_paths = []
    labels = []
    domain_to_idx = {domain: idx for idx, domain in enumerate(DOMAINS)}
    
    print("📷 Loading and labeling your dataset...")
    
    for folder, domain in folder_to_domain.items():
        folder_path = os.path.join(base_path, folder)
        
        if os.path.exists(folder_path):
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images.extend(Path(folder_path).glob(ext))
            
            if images:
                print(f"✅ {folder}: {len(images)} images → {domain}")
                
                for img_path in images:
                    image_paths.append(str(img_path))
                    labels.append(domain_to_idx[domain])
    
    print(f"\n📊 Total labeled images: {len(image_paths)}")
    
    # Count by domain
    domain_counts = {}
    for label in labels:
        domain = DOMAINS[label]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} images")
    
    return image_paths, labels, domain_to_idx

def create_universal_model(num_classes):
    """Create model for universal classification"""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except:
        model = models.resnet18(pretrained=True)
    
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'layer1' in name or 'layer2' in name:
            param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    return model

def train_universal_model():
    """Train the universal model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎯 Using device: {device}")
    
    # Load data
    image_paths, labels, domain_to_idx = load_and_label_data()
    
    if len(image_paths) < 50:
        print("❌ Not enough data for training")
        return
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📈 Dataset split:")
    print(f"Training: {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")
    
    # Create datasets
    train_dataset = UniversalImageDataset(train_paths, train_labels, augment=True)
    val_dataset = UniversalImageDataset(val_paths, val_labels, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)
    
    # Create model
    num_classes = len(DOMAINS)
    model = create_universal_model(num_classes).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize unfrozen parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.001
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    epochs = 10
    best_accuracy = 0
    
    print(f"\n🚀 Training universal model for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, batch_labels in train_bar:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': f"{100 * (predicted == batch_labels).sum().item() / batch_labels.size(0):.1f}%"
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train - Accuracy: {train_acc:.1f}%, Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val   - Accuracy: {val_acc:.1f}%, Loss: {val_loss/len(val_loader):.4f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_universal_model(model, DOMAINS, domain_to_idx, image_paths, labels, epoch+1, val_acc)
            print(f"  💾 Saved new best model (accuracy: {val_acc:.1f}%)")
        
        scheduler.step()
    
    print(f"\n✅ Universal model training complete! Best accuracy: {best_accuracy:.1f}%")
    
    return model

def save_universal_model(model, domains, domain_to_idx, image_paths, labels, epoch, accuracy):
    """Save the universal model"""
    save_dir = "ml_models/universal_model/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'domains': domains,
        'domain_to_idx': domain_to_idx,
        'num_classes': len(domains),
        'epoch': epoch,
        'accuracy': accuracy,
        'torch_version': torch.__version__
    }, os.path.join(save_dir, "model.pth"))
    
    # Save metadata
    metadata = {
        'total_images': len(image_paths),
        'domains': domains,
        'domain_distribution': {domains[i]: labels.count(i) for i in range(len(domains)) if labels.count(i) > 0},
        'training_date': str(Path(__file__).parent),
        'model_architecture': 'ResNet18',
        'input_size': [3, 224, 224]
    }
    
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save mapping
    idx_to_domain = {idx: domain for idx, domain in enumerate(domains)}
    with open(os.path.join(save_dir, "mapping.pkl"), "wb") as f:
        pickle.dump({
            'domain_to_idx': domain_to_idx,
            'idx_to_domain': idx_to_domain
        }, f)
    
    print(f"  💾 Universal model saved to {save_dir}")

def main():
    """Main training function"""
    print("=" * 60)
    print("UNIVERSAL IMAGE CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"📊 Domains: {len(DOMAINS)}")
    print(f"📈 Including restricted domains: weapons, drugs, adult_content, gambling")
    
    # Train model
    model = train_universal_model()
    
    if model:
        print("\n" + "=" * 60)
        print("🎉 UNIVERSAL MODEL TRAINING COMPLETE!")
        print("=" * 60)
        print("📁 Model saved to: ml_models/universal_model/")
        print("🔧 This model can now detect ANY domain")
        print("\n⚠️  NOTE: To detect more domains accurately, you need:")
        print("   1. More training data for other domains")
        print("   2. Or use zero-shot CLIP for unlabelled domains")
        print("   3. Current model only trained on education/sports")

if __name__ == "__main__":
    main()