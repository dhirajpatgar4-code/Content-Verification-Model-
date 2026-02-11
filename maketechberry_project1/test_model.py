# test_model.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if we can import torch
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed!")
    print("Run: pip install torch torchvision")
    sys.exit(1)

# Check dataset
print("\n🔍 Checking dataset...")
import glob
image_files = glob.glob("dataset/**/*.jpg", recursive=True) + \
              glob.glob("dataset/**/*.png", recursive=True) + \
              glob.glob("dataset/**/*.jpeg", recursive=True)

print(f"📸 Found {len(image_files)} image files")

# Show some samples
print("\n📂 Sample files:")
for img in image_files[:5]:
    print(f"  - {img}")

# Check folder structure
print("\n📁 Folder structure:")
for folder in os.listdir("dataset"):
    folder_path = os.path.join("dataset", folder)
    if os.path.isdir(folder_path):
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.png")) + \
                 glob.glob(os.path.join(folder_path, "*.jpeg"))
        print(f"  📁 {folder}: {len(images)} images")