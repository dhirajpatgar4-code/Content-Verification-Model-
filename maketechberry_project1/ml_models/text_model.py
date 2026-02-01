import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    """Dataset for text classification"""
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
    """BERT-based text classifier"""
    def __init__(self, num_classes, pretrained_model='bert-base-uncased'):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TextModelTrainer:
    """Train and manage text classification model"""
    
    def __init__(self, dataset_path="dataset/text_data/"):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 19  # 14 business + 5 restricted
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TextClassifier(self.num_classes).to(self.device)
        
        # Category mapping
        self.categories = [
            # Business domains (14)
            "food", "tech", "education", "health", "finance", "fashion",
            "electronics", "automotive", "real_estate", "entertainment",
            "travel", "beauty", "home", "sports",
            # Restricted categories (5)
            "weapons", "drugs", "adult_content", "gambling", "unknown"
        ]
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for idx, cat in enumerate(self.categories)}
    
    def load_training_data(self):
        """Load and prepare training data from your dataset"""
        print("📊 Loading training data...")
        
        data = []
        labels = []
        
        # Load from your Education_PhysicalEdu_Dataset
        dataset_root = "dataset/Education_PhysicalEdu_Dataset/"
        
        # Map folder names to categories
        folder_to_category = {
            "Classroom_Learning": "education",
            "Lab_Activities": "education",
            "Online_Learning": "education",
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
            "Gym_Scenes": "sports",
            "Student_Behavior": "education"
        }
        
        # Generate synthetic text data based on your dataset structure
        for folder, category in folder_to_category.items():
            folder_path = os.path.join(dataset_root, folder)
            if os.path.exists(folder_path):
                # Generate text samples for each category
                samples = self._generate_text_samples(category, folder)
                data.extend(samples)
                labels.extend([self.category_to_idx[category]] * len(samples))
        
        # Add restricted content samples
        restricted_samples = self._generate_restricted_samples()
        data.extend(restricted_samples['texts'])
        labels.extend(restricted_samples['labels'])
        
        print(f"✅ Loaded {len(data)} training samples")
        return data, labels
    
    def _generate_text_samples(self, category, folder_name):
        """Generate text samples based on category"""
        samples = []
        
        if category == "education":
            samples = [
                f"Classroom activities in {folder_name}",
                f"Learning materials for {folder_name}",
                f"Educational content about {folder_name}",
                f"Teaching resources for {folder_name}",
                f"Student activities in {folder_name}"
            ]
        elif category == "sports":
            samples = [
                f"Sports training for {folder_name}",
                f"Physical education {folder_name}",
                f"Athletic activities in {folder_name}",
                f"Sports equipment for {folder_name}",
                f"Training exercises for {folder_name}"
            ]
        elif category == "health":
            samples = [
                f"Health and wellness {folder_name}",
                f"Fitness activities for {folder_name}",
                f"Wellness programs in {folder_name}",
                f"Health training {folder_name}",
                f"Physical wellness {folder_name}"
            ]
        
        # Add more variations
        variations = []
        for sample in samples:
            variations.append(sample)
            variations.append(sample.lower())
            variations.append(sample.upper())
            variations.append(f"New: {sample}")
            variations.append(f"Best {sample}")
        
        return variations
    
    def _generate_restricted_samples(self):
        """Generate restricted content samples"""
        restricted_categories = ["weapons", "drugs", "adult_content", "gambling"]
        samples = []
        labels = []
        
        weapons_samples = [
            "Buy guns and firearms online",
            "Automatic rifles for sale",
            "Weapons and ammunition available",
            "Military grade weapons",
            "Concealed firearms"
        ]
        
        drugs_samples = [
            "Illegal drugs for sale",
            "Buy cocaine online",
            "Marijuana delivery",
            "Prescription drugs without prescription",
            "Narcotics available"
        ]
        
        adult_samples = [
            "Adult content and pornography",
            "Explicit material available",
            "Adult videos and photos",
            "NSFW content",
            "Adult entertainment"
        ]
        
        gambling_samples = [
            "Online casino gambling",
            "Sports betting and gambling",
            "Poker and casino games",
            "Betting sites and apps",
            "Lottery and gambling"
        ]
        
        # Add all restricted samples
        for cat, cat_samples in [
            ("weapons", weapons_samples),
            ("drugs", drugs_samples),
            ("adult_content", adult_samples),
            ("gambling", gambling_samples)
        ]:
            samples.extend(cat_samples)
            labels.extend([self.category_to_idx[cat]] * len(cat_samples))
        
        return {"texts": samples, "labels": labels}
    
    def train(self, epochs=10, batch_size=16, learning_rate=2e-5):
        """Train the text classification model"""
        print("🚀 Starting text model training...")
        
        # Load data
        texts, labels = self.load_training_data()
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training setup
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}")
            print(f"Val Accuracy: {accuracy:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model()
        
        print(f"✅ Training completed! Best accuracy: {best_accuracy:.2f}%")
    
    def save_model(self, save_path="ml_models/text_model/"):
        """Save the trained model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pth"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save category mappings
        with open(os.path.join(save_path, "categories.pkl"), "wb") as f:
            pickle.dump({
                'categories': self.categories,
                'category_to_idx': self.category_to_idx,
                'idx_to_category': self.idx_to_category
            }, f)
        
        print(f"💾 Model saved to {save_path}")
    
    def load_model(self, model_path="ml_models/text_model/"):
        """Load trained model"""
        # Load category mappings
        with open(os.path.join(model_path, "categories.pkl"), "rb") as f:
            mappings = pickle.load(f)
            self.categories = mappings['categories']
            self.category_to_idx = mappings['category_to_idx']
            self.idx_to_category = mappings['idx_to_category']
        
        # Load model
        self.model.load_state_dict(torch.load(
            os.path.join(model_path, "model.pth"),
            map_location=self.device
        ))
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        print("✅ Text model loaded successfully")
    
    def predict(self, text, top_k=3):
        """Predict category for given text"""
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
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
            'model_used': 'bert_text_classifier'
        }