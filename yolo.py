import os
import json
import random
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import requests
from ultralytics import YOLO

# Configuration
API_URL = "https://aedes.almanza.cc/api/annotations/export-data?format=yolo"
API_KEY = "secret ;)"
DATASET_DIR = Path("dataset")
ORIGINAL_IMAGES_DIR = DATASET_DIR / "original_images"
ORIGINAL_ANNOTATIONS_DIR = DATASET_DIR / "original_labels"
TILED_IMAGES_DIR = DATASET_DIR / "images"
TILED_ANNOTATIONS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = Path("model_output")
EPOCHS = 50
TILE_SIZE = 250
OVERLAP_RATIO = 0.2
BATCH_SIZE = 16
PATIENCE = 50
INITIAL_LR = 0.01
WARMUP_EPOCHS = 5

# Create directories
for directory in [ORIGINAL_IMAGES_DIR, ORIGINAL_ANNOTATIONS_DIR, TILED_IMAGES_DIR, TILED_ANNOTATIONS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def fetch_data():
    response = requests.get(API_URL, headers={"Authorization": f"Bearer {API_KEY}"})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, Response content: {response.content}")
        return None

data = fetch_data()
if not data:
    raise Exception("Failed to fetch data")

raw_data_path = DATASET_DIR / "data.json"
with open(raw_data_path, 'w') as f:
    json.dump(data, f)

def yolo_to_absolute(yolo_ann: str, img_width: int, img_height: int):
    parts = yolo_ann.split()
    if len(parts) != 5:
        raise ValueError(f"Annotation line '{yolo_ann}' does not have 5 parts after splitting. Parts found: {parts}")

    try:
        # Strip each part individually before float conversion
        class_id_str = parts[0].strip()
        center_x_norm_str = parts[1].strip()
        center_y_norm_str = parts[2].strip()
        width_norm_str = parts[3].strip()
        height_norm_str = parts[4].strip()

        # Convert to float first, then class_id to int
        class_id = int(float(class_id_str))
        center_x_norm = float(center_x_norm_str)
        center_y_norm = float(center_y_norm_str)
        width_norm = float(width_norm_str)
        height_norm = float(height_norm_str)

    except ValueError as e:
        # This will be caught by the caller if conversion fails
        raise ValueError(f"Error converting parts of annotation line '{yolo_ann}' to numbers. Original parts: {parts}. Stripped parts: {list(map(str.strip,parts))}. Error: {e}")

    center_x = center_x_norm * img_width
    center_y = center_y_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height
    xmin = center_x - width / 2
    ymin = center_y - height / 2
    xmax = center_x + width / 2
    ymax = center_y + height / 2
    return int(class_id), xmin, ymin, xmax, ymax

def absolute_to_yolo(class_id, box, tile_width, tile_height):
    xmin, ymin, xmax, ymax = box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    
    center_x_norm = center_x / tile_width
    center_y_norm = center_y / tile_height
    width_norm = width / tile_width
    height_norm = height / tile_height
    
    # Return the string without any newline characters
    return f"{int(class_id)} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def create_tiled_dataset(raw_data, tile_size=TILE_SIZE, overlap_ratio=OVERLAP_RATIO, min_visibility_ratio=0.25):
    print("\n--- Starting Dataset Tiling Process ---")
    tiled_image_paths = []
    processed_image_ids = set()
    total_tiles_created = 0
    total_original_annotations_processed = 0
    total_tile_annotations_created = 0

    # First, download all images and save original annotations
    original_image_info = {}
    print("Step 1: Downloading original images and collecting annotation data...")
    for image_id, img_data in tqdm(raw_data.items(), desc="Downloading and pre-processing images"):
        image_url = img_data["image_url"]
        original_image_path = ORIGINAL_IMAGES_DIR / f"{image_id}.jpg"
        original_annotation_path = ORIGINAL_ANNOTATIONS_DIR / f"{image_id}.txt"

        if not original_image_path.exists():
            try:
                img_content = requests.get(image_url).content
                with open(original_image_path, 'wb') as f:
                    f.write(img_content)
            except Exception as e:
                print(f"Failed downloading {image_id}: {e}")
                continue

        original_img = cv2.imread(str(original_image_path))
        if original_img is None:
            print(f"Failed to read original image {original_image_path}")
            continue

        img_height, img_width = original_img.shape[:2]
        current_annotations_raw = img_data.get("annotations", [])
        original_image_info[image_id] = {
            "path": original_image_path,
            "width": img_width,
            "height": img_height,
            "annotations_raw": current_annotations_raw
        }
        total_original_annotations_processed += len(current_annotations_raw)
        
        # Save original annotations - Split by actual newlines and write each annotation on its own line
        with open(original_annotation_path, 'w') as f:
            for ann_line in current_annotations_raw:
                # Split by literal '\n' and write each part on a new line
                parts = ann_line.replace('\\n', '\n').strip().split('\n')
                for part in parts:
                    if part.strip():  # Only write non-empty lines
                        # Remove any trailing \n and write our own
                        f.write(f"{part.strip()}\n")

    print(f"Step 1 finished. Downloaded/verified {len(original_image_info)} images. Processed {total_original_annotations_processed} original annotation lines.")

    # Now, tile images and adjust annotations
    print("\nStep 2: Creating tiles and adjusting annotations...")
    for image_id, info in tqdm(original_image_info.items(), desc="Creating tiles"):
        original_img = cv2.imread(str(info["path"]))
        img_height, img_width = info["height"], info["width"]
        tiles_from_this_image = 0
        
        step_size = int(tile_size * (1 - overlap_ratio))
        if step_size == 0: step_size = tile_size

        for y in range(0, img_height, step_size):
            for x in range(0, img_width, step_size):
                y_end = min(y + tile_size, img_height)
                x_end = min(x + tile_size, img_width)
                
                tile_img = original_img[y:y_end, x:x_end]
                
                current_tile_height, current_tile_width = tile_img.shape[:2]
                if current_tile_height == 0 or current_tile_width == 0:
                    continue

                if current_tile_height < tile_size or current_tile_width < tile_size:
                    pad_h = tile_size - current_tile_height
                    pad_w = tile_size - current_tile_width
                    tile_img = cv2.copyMakeBorder(tile_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                
                tile_actual_height, tile_actual_width = tile_img.shape[:2]

                tile_annotations = []
                for ann_line_from_api_potentially_dirty in info["annotations_raw"]:
                    if not isinstance(ann_line_from_api_potentially_dirty, str):
                        print(f"Warning: non-string annotation found: {ann_line_from_api_potentially_dirty} for image {image_id}. Skipping this annotation.")
                        continue

                    # Split by literal '\n' and process each annotation separately
                    cleaned_api_string = ann_line_from_api_potentially_dirty.replace('\\n', '\n')
                    potential_yolo_lines = cleaned_api_string.strip().split('\n')

                    for current_yolo_line in potential_yolo_lines:
                        current_yolo_line = current_yolo_line.strip()
                        if not current_yolo_line:
                            continue
                        
                        try:
                            class_id, orig_xmin, orig_ymin, orig_xmax, orig_ymax = yolo_to_absolute(current_yolo_line, img_width, img_height)
                        except ValueError as e:
                            print(f"Skipping invalid original annotation line: '{current_yolo_line}' for image {image_id}. Details: {e}")
                            continue
                        
                        obj_box_orig = [orig_xmin, orig_ymin, orig_xmax, orig_ymax]
                        tile_box_orig = [x, y, x_end, y_end]

                        inter_xmin = max(obj_box_orig[0], tile_box_orig[0])
                        inter_ymin = max(obj_box_orig[1], tile_box_orig[1])
                        inter_xmax = min(obj_box_orig[2], tile_box_orig[2])
                        inter_ymax = min(obj_box_orig[3], tile_box_orig[3])

                        if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
                            new_xmin = inter_xmin - x
                            new_ymin = inter_ymin - y
                            new_xmax = inter_xmax - x
                            new_ymax = inter_ymax - y
                            
                            obj_orig_width = orig_xmax - orig_xmin
                            obj_orig_height = orig_ymax - orig_ymin
                            obj_orig_area = obj_orig_width * obj_orig_height if obj_orig_width > 0 and obj_orig_height > 0 else 0
                            
                            inter_width = new_xmax - new_xmin
                            inter_height = new_ymax - new_ymin
                            inter_area = inter_width * inter_height if inter_width > 0 and inter_height > 0 else 0

                            if obj_orig_area > 0 and (inter_area / obj_orig_area) >= min_visibility_ratio:
                                new_xmin_clipped = max(0, new_xmin)
                                new_ymin_clipped = max(0, new_ymin)
                                new_xmax_clipped = min(tile_actual_width -1, new_xmax)
                                new_ymax_clipped = min(tile_actual_height -1, new_ymax)

                                if new_xmax_clipped > new_xmin_clipped and new_ymax_clipped > new_ymin_clipped:
                                    tile_annotations.append(absolute_to_yolo(class_id, 
                                                                    [new_xmin_clipped, new_ymin_clipped, new_xmax_clipped, new_ymax_clipped],
                                                                    tile_actual_width, tile_actual_height))
                                    total_tile_annotations_created +=1
                
                if tile_annotations: 
                    tile_image_filename = f"{image_id}_tile_{y}_{x}.jpg"
                    tile_annotation_filename = f"{image_id}_tile_{y}_{x}.txt"
                    
                    tile_image_save_path = TILED_IMAGES_DIR / tile_image_filename
                    tile_annotation_save_path = TILED_ANNOTATIONS_DIR / tile_annotation_filename
                    
                    cv2.imwrite(str(tile_image_save_path), tile_img)
                    # Write each annotation on a new line, ensuring no trailing \n in the annotation itself
                    with open(tile_annotation_save_path, 'w') as f_ann:
                        for ann_str in tile_annotations:
                            f_ann.write(f"{ann_str.strip()}\n")
                    tiled_image_paths.append(tile_image_save_path)
                    total_tiles_created += 1
                    tiles_from_this_image +=1
                    processed_image_ids.add(image_id)
    
    print(f"\nStep 2 finished. Processed {len(processed_image_ids)} original images into {total_tiles_created} tiles with annotations.")
    print(f"Total original annotations processed: {total_original_annotations_processed}")
    print(f"Total annotations created for tiles: {total_tile_annotations_created}")

    if len(raw_data) > len(processed_image_ids):
        print(f"Warning: {len(raw_data) - len(processed_image_ids)} original images did not result in any tiles (e.g., no annotations, download error, or no annotations met visibility criteria in tiles).")
    print("--- Dataset Tiling Process Finished ---")
    return tiled_image_paths

# Create tiled dataset
all_tiled_image_paths = create_tiled_dataset(data)

if not all_tiled_image_paths:
    raise Exception("Failed to create any tiled images. Check logs for errors.")
print(f"Successfully created {len(all_tiled_image_paths)} tiled image paths for dataset.")

# Split dataset into train, val, test
print("\n--- Splitting Tiled Dataset into Train, Val, Test ---")
random.shuffle(all_tiled_image_paths)
total_tiles = len(all_tiled_image_paths)
train_split_idx = int(0.7 * total_tiles)
val_split_idx = int(0.15 * total_tiles)

splits = {
    'train': all_tiled_image_paths[:train_split_idx],
    'val': all_tiled_image_paths[train_split_idx : train_split_idx + val_split_idx],
    'test': all_tiled_image_paths[train_split_idx + val_split_idx:]
}
print(f"Dataset split: Train ({len(splits['train'])} tiles), Val ({len(splits['val'])} tiles), Test ({len(splits['test'])} tiles).")

# Create split directories and move files (now for tiled images and labels)
print("\n--- Organizing Tiled Files into Split Directories ---")
for split_name, paths in splits.items():
    print(f"  Processing split: {split_name}")
    split_image_dir = TILED_IMAGES_DIR / split_name
    split_annot_dir = TILED_ANNOTATIONS_DIR / split_name
    print(f"    Image directory: {split_image_dir}")
    print(f"    Annotation directory: {split_annot_dir}")
    split_image_dir.mkdir(parents=True, exist_ok=True)
    split_annot_dir.mkdir(parents=True, exist_ok=True)
    
    for tiled_img_path in tqdm(paths, desc=f"Organizing {split_name} set"):
        tiled_annot_path = TILED_ANNOTATIONS_DIR / f"{tiled_img_path.stem}.txt"
        
        new_img_path = split_image_dir / tiled_img_path.name
        new_annot_path = split_annot_dir / tiled_annot_path.name

        if tiled_annot_path.exists():
            shutil.move(tiled_img_path, new_img_path)  # Changed from copy to move
            shutil.move(tiled_annot_path, new_annot_path)  # Changed from copy to move
        else:
            print(f"Warning: Annotation file {tiled_annot_path} not found for image {tiled_img_path}. Skipping this file for {split_name} set.")

# Clean up any remaining files in the root of images and labels directories
print("\nCleaning up root directories...")
for file in TILED_IMAGES_DIR.glob("*.jpg"):
    if file.is_file():  # Only remove files, not directories
        file.unlink()
for file in TILED_ANNOTATIONS_DIR.glob("*.txt"):
    if file.is_file():  # Only remove files, not directories
        file.unlink()

print("File organization complete.")

# Create dataset YAML file for YOLO
print("\n--- Creating dataset.yaml ---")
# Paths in YAML should be relative to the YAML file's location (DATASET_DIR)
# Example: If TILED_IMAGES_DIR is "dataset/images", and dataset.yaml is in "dataset/",
# then the path for training images inside YAML should be "images/train".
relative_images_path_for_yaml = TILED_IMAGES_DIR.name # This should be 'images'

dataset_yaml_content = f"""
path: {str(DATASET_DIR.absolute())}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

nc: 1  # number of classes
names: ['bbox']  # class names
"""
dataset_yaml_path = DATASET_DIR / "dataset.yaml"
with open(dataset_yaml_path, 'w') as f:
    f.write(dataset_yaml_content.strip())
print(f"dataset.yaml created at {dataset_yaml_path}")

# Train YOLO model
print("\n--- Training YOLO Model ---")
try:
    # Load a pre-trained model - using small instead of nano for better accuracy
    model = YOLO('yolov8s.pt')

    # Train the model with optimized parameters for small object detection
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=EPOCHS,
        imgsz=TILE_SIZE,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        lr0=INITIAL_LR,
        overlap_mask=True,  # Enable mask overlap
        multi_scale=True,   # Enable multi-scale training
        degrees=0.0,        # No rotation augmentation (can cause issues with small objects)
        scale=0.5,         # Scale augmentation
        mosaic=1.0,        # Enable mosaic augmentation
        mixup=0.3,         # Increased mixup augmentation
        copy_paste=0.3,    # Increased copy-paste augmentation
        box=7.5,           # Box loss gain
        cls=0.5,           # Class loss gain
        dfl=1.5,           # DFL loss gain
        warmup_epochs=WARMUP_EPOCHS,  # Add warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        close_mosaic=10,   # Disable mosaic in last 10 epochs
        label_smoothing=0.1  # Add label smoothing
    )
    print("--- Training Complete ---")

    # The best model is automatically saved as 'best.pt' in the experiment directory (e.g., model_output/train/weights/best.pt)
    # You can also save the last model explicitly if needed
    # last_model_path = OUTPUT_DIR / "last.pt"
    # model.save(str(last_model_path))
    # print(f"Last trained model saved to {last_model_path}")
    
    print(f"Trained model and results saved in: {results.save_dir}")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

except Exception as e:
    print(f"An error occurred during model training: {e}")
    # Consider re-raising the exception if you want the script to stop
    # raise

print("\n--- YOLO Script Finished ---")
