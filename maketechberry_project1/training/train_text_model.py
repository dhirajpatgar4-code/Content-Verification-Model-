#!/usr/bin/env python3
"""
Complete text model training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import json
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def generate_training_data():
    """Generate training data based on your dataset structure"""
    print("📊 Generating training data...")
    
    # All categories (19 total)
    categories = [
        "food", "tech", "education", "health", "finance", "fashion",
        "electronics", "automotive", "real_estate", "entertainment",
        "travel", "beauty", "home", "sports",
        "weapons", "drugs", "adult_content", "gambling", "unknown"
    ]
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    texts = []
    labels = []
    
    # Generate education-related samples
    education_keywords = [
        "classroom", "learning", "student", "teacher", "school", 
        "education", "study", "course", "lesson", "academic",
        "university", "college", "training", "workshop", "seminar"
    ]
    
    sports_keywords = [
        "sports", "football", "basketball", "cricket", "athletics",
        "fitness", "exercise", "training", "workout", "gym",
        "yoga", "meditation", "stretching", "warmup", "physical"
    ]
    
    health_keywords = [
        "health", "wellness", "fitness", "exercise", "yoga",
        "meditation", "nutrition", "diet", "wellbeing", "therapy"
    ]
    
    # Generate samples for each category
    samples_per_category = 500
    
    # Education samples
    for _ in range(samples_per_category):
        for keyword in education_keywords:
            texts.append(f"{keyword} activities and learning materials for students")
            labels.append(category_to_idx["education"])
    
    # Sports samples
    for _ in range(samples_per_category):
        for keyword in sports_keywords:
            texts.append(f"{keyword} training and physical education exercises")
            labels.append(category_to_idx["sports"])
    
    # Health samples
    for _ in range(samples_per_category):
        for keyword in health_keywords:
            texts.append(f"{keyword} and wellness programs for better health")
            labels.append(category_to_idx["health"])
    
    # Restricted content samples
    restricted_samples = {
        "weapons": ["guns", "firearms", "ammunition", "weapons", "rifles"],
        "drugs": ["drugs", "cocaine", "heroin", "marijuana", "prescription"],
        "adult_content": ["pornography", "adult", "explicit", "nsfw", "xxx"],
        "gambling": ["casino", "gambling", "betting", "poker", "lottery"]
    }
    
    for category, keywords in restricted_samples.items():
        for _ in range(100):
            for keyword in keywords:
                texts.append(f"buy {keyword} online cheap")
                labels.append(category_to_idx[category])
    
    print(f"✅ Generated {len(texts)} training samples")
    return texts, labels, categories, category_to_idx

def train_text_model():
    """Train the text classification model"""
    print("🚀 Starting text model training...")
    
    # Generate training data
    texts, labels, categories, category_to_idx = generate_training_data()
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = len(categories)
    model = TextClassifier(num_classes).to(device)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 5
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, tokenizer, categories, category_to_idx)
    
    print(f"\n✅ Training complete! Best accuracy: {best_accuracy:.2f}%")

def save_model(model, tokenizer, categories, category_to_idx, save_path="ml_models/text_model/"):
    """Save the trained model"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save category mappings
    with open(os.path.join(save_path, "categories.pkl"), "wb") as f:
        pickle.dump({
            'categories': categories,
            'category_to_idx': category_to_idx,
            'idx_to_category': {idx: cat for cat, idx in category_to_idx.items()}
        }, f)
    
    # Save config
    config = {
        'model_type': 'bert',
        'num_classes': len(categories),
        'max_length': 128
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f)
    
    print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    train_text_model()