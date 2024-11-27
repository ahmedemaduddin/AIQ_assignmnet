import json
import os
import random
from shutil import copy2

def convert_coco_to_yolo(coco_file, output_dir, split_ratio=0.7):
    """
    Converts COCO annotations to YOLO format and splits into train and validation sets.

    Parameters:
        coco_file (str): Path to the COCO annotation JSON file.
        output_dir (str): Directory to save YOLO annotations.
        split_ratio (float): Percentage of images for training (default: 0.7).
    """
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)

    # Extract images and annotations
    images = {image['id']: image for image in coco_data['images']}
    annotations = coco_data['annotations']
    categories = {category['id']: category['name'] for category in coco_data['categories']}
    
    # Prepare a category mapping (category id to index)
    category_mapping = {cat_id: idx for idx, cat_id in enumerate(categories.keys())}

    # Shuffle and split images into train and validation
    image_ids = list(images.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * split_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    # Process annotations
    for annotation in annotations:
        image_id = annotation['image_id']
        image_info = images[image_id]
        img_width, img_height = image_info['width'], image_info['height']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # COCO format: [x_min, y_min, width, height]
        
        # Convert bbox to YOLO format: [x_center, y_center, width, height] normalized
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        width /= img_width
        height /= img_height

        # Map category id to a zero-based index for YOLO
        category_idx = category_mapping[category_id]

        # Generate YOLO annotation line
        yolo_annotation = f"{category_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        # Determine whether the image belongs to train or val
        subset = 'train' if image_id in train_ids else 'val'
        subset_dir = train_dir if subset == 'train' else val_dir

        # Write annotation file
        yolo_file = os.path.join(subset_dir, 'labels', f"{os.path.splitext(image_info['file_name'])[0]}.txt")
        with open(yolo_file, 'a') as yf:
            yf.write(yolo_annotation)

    # Copy image files to train and val directories
    for image in images.values():
        subset = 'train' if image['id'] in train_ids else 'val'
        subset_dir = train_dir if subset == 'train' else val_dir
        src_image_path = os.path.join(os.path.dirname(coco_file), image['file_name'])
        dst_image_path = os.path.join(subset_dir, 'images', image['file_name'])
        copy2(src_image_path, dst_image_path)

    print(f"Dataset split completed. YOLO annotations saved to {output_dir}")

# Example usage
coco_file = "/home/ahmed/AIQ/Challenge1/coin-dataset/_annotations.coco.json"
output_dir = "/home/ahmed/AIQ/Challenge1/yolo_format"
convert_coco_to_yolo(coco_file, output_dir, split_ratio=0.7)
