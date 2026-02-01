#!/usr/bin/env python3
"""
Data preprocessing utilities
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import pickle

def prepare_text_data(texts: List[str], labels: List[str], output_path: str):
    """Prepare text data for training"""
    print(f"📝 Preparing text data: {len(texts)} samples")
    
    # Create dataframe
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Save label mapping
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    with open(output_path.replace('.csv', '_labels.json'), 'w') as f:
        json.dump(label_map, f)
    
    print(f"✅ Text data saved to {output_path}")
    return df, label_map

def prepare_image_data(image_dir: str, output_path: str):
    """Prepare image data metadata"""
    print(f"🖼️ Preparing image data from {image_dir}")
    
    data = []
    
    # Walk through directory
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                category = os.path.basename(root)
                
                # Get image info
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        
                    data.append({
                        'image_path': image_path,
                        'category': category,
                        'filename': file,
                        'width': width,
                        'height': height
                    })
                except:
                    print(f"⚠️ Could not process {image_path}")
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Save category mapping
    categories = sorted(df['category'].unique())
    category_map = {cat: idx for idx, cat in enumerate(categories)}
    
    with open(output_path.replace('.csv', '_categories.json'), 'w') as f:
        json.dump(category_map, f)
    
    print(f"✅ Image metadata saved to {output_path}")
    print(f"📊 Found {len(df)} images in {len(categories)} categories")
    return df, category_map

def create_train_val_split(data: pd.DataFrame, val_ratio: float = 0.2, output_dir: str = "dataset/"):
    """Create train/validation split"""
    print("✂️ Creating train/validation split")
    
    # Shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    
    print(f"✅ Train split: {len(train_data)} samples -> {train_path}")
    print(f"✅ Val split: {len(val_data)} samples -> {val_path}")
    
    return train_data, val_data

def create_category_json(categories: List[str], output_path: str):
    """Create category mapping JSON file"""
    category_info = {}
    
    for category in categories:
        category_info[category] = {
            'id': len(category_info),
            'description': category.replace('_', ' ').title(),
            'type': 'business' if category not in ['weapons', 'drugs', 'adult_content', 'gambling'] else 'restricted'
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(category_info, f, indent=2)
    
    print(f"✅ Category info saved to {output_path}")
    return category_info

if __name__ == "__main__":
    # Example usage
    print("🔄 Running data preprocessing...")
    
    # Create sample categories JSON
    categories = [
        "food", "tech", "education", "health", "finance", "fashion",
        "electronics", "automotive", "real_estate", "entertainment",
        "travel", "beauty", "home", "sports",
        "weapons", "drugs", "adult_content", "gambling", "unknown"
    ]
    
    create_category_json(categories, "dataset/metadata/categories.json")
    print("✅ Data preprocessing complete!")