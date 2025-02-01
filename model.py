import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
source_dir = r"D:\objdetection_inceptionmodel\Image"
train_dir = r"D:\objdetection_inceptionmodel\data\train"
test_dir = r"D:\objdetection_inceptionmodel\data\test"

# Ensure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split images into train and test
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if os.path.isdir(class_path):
        # Get all images in the class folder
        images = os.listdir(class_path)
        
        # Split into train and test sets
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
        
        # Create class subdirectories in train and test directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Move images to train directory
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        
        # Move images to test directory
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

print("Dataset split completed!")
# # Paths for train and test directories
# train_dir = r"D:\objdetection_inceptionmodel\data\train"
# test_dir = r"D:\objdetection_inceptionmodel\data\test"


