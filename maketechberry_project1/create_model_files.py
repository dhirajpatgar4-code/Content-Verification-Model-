#!/usr/bin/env python3
"""
Script to create initial model files for testing
"""

import torch
import pickle
import json
import os

def create_text_model_files():
    """Create initial text model files"""
    print("📝 Creating text model files...")
    
    save_path = "ml_models/text_model/"
    os.makedirs(save_path, exist_ok=True)
    
    # Create dummy model weights (in production, these would come from training)
    categories = [
        "food", "tech", "education", "health", "finance", "fashion",
        "electronics", "automotive", "real_estate", "entertainment",
        "travel", "beauty", "home", "sports",
        "weapons", "drugs", "adult_content", "gambling", "unknown"
    ]
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
    
    # Save category mappings
    with open(os.path.join(save_path, "categories.pkl"), "wb") as f:
        pickle.dump({
            'categories': categories,
            'category_to_idx': category_to_idx,
            'idx_to_category': idx_to_category
        }, f)
    
    # Create dummy model file (empty for now, will be trained)
    # In production, you would save actual trained weights
    dummy_model = {"initialized": True, "requires_training": True}
    torch.save(dummy_model, os.path.join(save_path, "model.pth"))
    
    # Save config
    config = {
        'model_type': 'bert',
        'num_classes': len(categories),
        'max_length': 128,
        'status': 'requires_training'
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Text model files created in {save_path}")

def create_image_model_files():
    """Create initial image model files"""
    print("🖼️ Creating image model files...")
    
    save_path = "ml_models/image_model/"
    os.makedirs(save_path, exist_ok=True)
    
    # Categories
    categories = [
        "food", "tech", "education", "health", "finance", "fashion",
        "electronics", "automotive", "real_estate", "entertainment",
        "travel", "beauty", "home", "sports",
        "weapons", "drugs", "adult_content", "gambling", "unknown"
    ]
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
    
    # Folder to category mapping
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
    
    # Save category mappings
    with open(os.path.join(save_path, "categories.pkl"), "wb") as f:
        pickle.dump({
            'categories': categories,
            'category_to_idx': category_to_idx,
            'idx_to_category': idx_to_category,
            'folder_to_category': folder_to_category
        }, f)
    
    # Create dummy model file
    dummy_model = {"initialized": True, "requires_training": True}
    torch.save(dummy_model, os.path.join(save_path, "model.pth"))
    
    # Save dummy transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with open(os.path.join(save_path, "transform.pkl"), "wb") as f:
        pickle.dump(transform, f)
    
    # Save config
    config = {
        'model_type': 'resnet18',
        'num_classes': len(categories),
        'input_size': [224, 224],
        'status': 'requires_training'
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Image model files created in {save_path}")

def create_multimodal_files():
    """Create multimodal fusion files"""
    print("🔄 Creating multimodal fusion files...")
    
    save_path = "ml_models/multimodal/"
    os.makedirs(save_path, exist_ok=True)
    
    # Fusion rules
    fusion_rules = {
        'text_weight': 0.7,
        'image_weight': 0.3,
        'confidence_threshold': 0.6,
        'consensus_boost': 1.1,
        'disagreement_penalty': 0.9,
        'max_attempts': 3,
        'fallback_category': 'unknown'
    }
    
    # Save fusion rules
    with open(os.path.join(save_path, "fusion_rules.pkl"), "wb") as f:
        pickle.dump(fusion_rules, f)
    
    # Save config
    config = {
        'model_type': 'rule_based_fusion',
        'text_model': 'bert',
        'image_model': 'resnet18',
        'version': '1.0',
        'status': 'ready'
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Multimodal fusion files created in {save_path}")

def create_vectorizer_file():
    """Create text vectorizer file (for compatibility)"""
    print("🔤 Creating text vectorizer file...")
    
    save_path = "ml_models/text_model/"
    os.makedirs(save_path, exist_ok=True)
    
    # Create dummy vectorizer
    vectorizer_info = {
        'type': 'bert_tokenizer',
        'vocab_size': 30522,
        'max_length': 128,
        'requires_training': False
    }
    
    with open(os.path.join(save_path, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer_info, f)
    
    print(f"✅ Vectorizer file created in {save_path}")

def main():
    """Create all model files"""
    print("🚀 Creating all model files...")
    print("="*60)
    
    create_text_model_files()
    print("-"*60)
    
    create_image_model_files()
    print("-"*60)
    
    create_multimodal_files()
    print("-"*60)
    
    create_vectorizer_file()
    print("="*60)
    print("🎉 All model files created successfully!")
    print("\nNote: These are placeholder files.")
    print("Run training scripts to create actual trained models:")
    print("1. python training/train_text_model.py")
    print("2. python training/train_image_model.py")
    print("3. python training/train_multimodal.py")

if __name__ == "__main__":
    main()