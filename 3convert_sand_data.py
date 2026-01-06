import json
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
from pathlib import Path
from tqdm import tqdm
import cv2
import random
import shutil

def create_coco_with_file_structure(base_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Create COCO format in a new directory while keeping original data intact.
    
    Output Structure:
    sand_coco/
    ├── 2-bm-aps/
    │   ├── images/
    │   │   ├── train/
    │   │   │   ├── image1.tiff
    │   │   │   └── train_annotations.json
    │   │   └── test/
    │   │       ├── image2.tiff
    │   │       └── test_annotations.json
    │   └── masks/
    │       ├── Mask_image1.tiff
    │       └── Mask_image2.tiff
    └── ...
    
    Args:
        base_dir: Path to original sand_data folder
        output_dir: Path to output folder (e.g., sand_coco)
        val_ratio: Ratio for test split (0.2 = 20% test)
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all subdirectories (datasets)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print("=" * 70)
    print(f"Creating COCO Dataset in: {output_dir}")
    print("=" * 70)
    
    for subdir in subdirs:
        print(f"\n{'='*70}")
        print(f"Processing: {subdir.name}")
        print(f"{'='*70}")
        
        images_dir = subdir / "images"
        masks_dir = subdir / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            print(f"  ⚠ Skipping {subdir.name} - missing images or masks folder")
            continue
        
        # Create output subdirectory structure
        output_subdir = output_path / subdir.name
        output_images_dir = output_subdir / "images"
        output_masks_dir = output_subdir / "masks"
        train_dir = output_images_dir / "train"
        test_dir = output_images_dir / "test"
        
        # Create all directories
        train_dir.mkdir(exist_ok=True, parents=True)
        test_dir.mkdir(exist_ok=True, parents=True)
        output_masks_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all image files
        image_files = sorted(list(images_dir.glob("*.tiff")) + list(images_dir.glob("*.tif")))
        
        if len(image_files) == 0:
            print(f"  ⚠ No images found in {images_dir}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Split into train and test
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_ratio))
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        print(f"  Split: {len(train_files)} train, {len(test_files)} test")
        
        # Copy masks to output directory
        print(f"\n  Copying masks...")
        mask_files = list(masks_dir.glob("*.tiff")) + list(masks_dir.glob("*.tif"))
        for mask_file in tqdm(mask_files, desc="    Masks"):
            dest_mask = output_masks_dir / mask_file.name
            if not dest_mask.exists():
                shutil.copy2(mask_file, dest_mask)
        
        # Process train set
        print(f"\n  Processing TRAIN set...")
        train_coco = process_split(
            train_files,
            output_masks_dir,  # Use copied masks
            train_dir,
            split_name="train"
        )
        
        # Save train COCO JSON
        train_json_path = train_dir / "train_annotations.json"
        with open(train_json_path, 'w') as f:
            json.dump(train_coco, f, indent=2)
        print(f"    ✓ Saved: {train_json_path}")
        print(f"      Images: {len(train_coco['images'])}")
        print(f"      Annotations: {len(train_coco['annotations'])}")
        
        # Process test set
        print(f"\n  Processing TEST set...")
        test_coco = process_split(
            test_files,
            output_masks_dir,  # Use copied masks
            test_dir,
            split_name="test"
        )
        
        # Save test COCO JSON
        test_json_path = test_dir / "test_annotations.json"
        with open(test_json_path, 'w') as f:
            json.dump(test_coco, f, indent=2)
        print(f"    ✓ Saved: {test_json_path}")
        print(f"      Images: {len(test_coco['images'])}")
        print(f"      Annotations: {len(test_coco['annotations'])}")
        
        # Print class distribution
        print_class_distribution(train_coco, "TRAIN")
        print_class_distribution(test_coco, "TEST")
    
    print("\n" + "="*70)
    print("✓ ALL DATASETS PROCESSED!")
    print("="*70)


def process_split(image_files, masks_dir, output_dir, split_name):
    """
    Process a train or test split
    
    Args:
        image_files: List of image file paths (from original location)
        masks_dir: Directory containing mask files (in output location)
        output_dir: Directory to copy images to (train/ or test/)
        split_name: "train" or "test"
    
    Returns:
        COCO format dictionary
    """
    
    # COCO format: category IDs start at 1, not 0
    # Standard COCO structure: info, licenses, categories, images, annotations
    coco_output = {
        "info": {
            "description": "NIST Sand Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "NIST",
            "date_created": "2025/01/06"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [
            {"id": 1, "name": "outside_field_of_view", "supercategory": "background"},
            {"id": 2, "name": "air", "supercategory": "object"},
            {"id": 3, "name": "sand", "supercategory": "object"},
            {"id": 4, "name": "capillary_area", "supercategory": "object"},
            {"id": 5, "name": "inclusions", "supercategory": "object"}
        ],
        "images": [],
        "annotations": []
    }
    
    annotation_id = 1
    
    for image_id, image_path in enumerate(tqdm(image_files, desc=f"    {split_name}"), start=1):
        # Load image to get dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            print(f"      ✗ Error loading {image_path.name}: {e}")
            continue
        
        # Copy image to output directory
        dest_path = output_dir / image_path.name
        if not dest_path.exists():
            shutil.copy2(image_path, dest_path)
        
        # Add image info
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "height": height,
            "width": width
        })
        
        # Find corresponding mask
        mask_name1 = f"Mask_{image_path.stem}.tiff"
        mask_name2 = f"Mask_{image_path.stem}.tif"
        
        mask_path = masks_dir / mask_name1
        if not mask_path.exists():
            mask_path = masks_dir / mask_name2
        
        if not mask_path.exists():
            print(f"      ⚠ No mask found for {image_path.name}")
            continue
        
        # Load mask
        try:
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            print(f"      ✗ Error loading mask for {image_path.name}: {e}")
            continue
        
        # Process each class - now including class 0
        # Mask values: 0, 1, 2, 3, 4
        # Map to category IDs: 1, 2, 3, 4, 5
        unique_classes = np.unique(mask)
        
        for mask_value in unique_classes:
            # Map mask value to COCO category ID
            # mask_value 0 -> category_id 1
            # mask_value 1 -> category_id 2
            # etc.
            category_id = int(mask_value) + 1
            
            # Create binary mask for this class
            binary_mask = (mask == mask_value).astype(np.uint8)
            
            # Find separate instances (connected components)
            num_instances, labeled_mask = cv2.connectedComponents(binary_mask)
            
            # Process each instance
            for instance_id in range(1, num_instances):
                instance_mask = (labeled_mask == instance_id).astype(np.uint8)
                
                # Skip tiny instances
                if instance_mask.sum() < 10:
                    continue
                
                # Get bounding box
                rows = np.any(instance_mask, axis=1)
                cols = np.any(instance_mask, axis=0)
                
                if not rows.any() or not cols.any():
                    continue
                
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                # COCO bbox format: [x, y, width, height]
                bbox = [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]
                
                # Convert mask to RLE
                fortran_mask = np.asfortranarray(instance_mask)
                rle = mask_util.encode(fortran_mask)
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # Calculate area
                area = float(instance_mask.sum())
                
                # Add annotation
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # This is now 1-5 instead of 0-4
                    "bbox": bbox,
                    "area": area,
                    "segmentation": rle,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    return coco_output


def print_class_distribution(coco_data, split_name):
    """Print class distribution statistics"""
    
    class_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
    
    print(f"\n    {split_name} Class Distribution:")
    for cat in coco_data['categories']:
        if cat['id'] in class_counts:
            print(f"      {cat['id']}: {cat['name']:<25} - {class_counts[cat['id']]:>5} instances")


def create_summary_report(output_dir):
    """
    Create a summary report of all datasets
    """
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    output_path = Path(output_dir)
    subdirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    total_train_images = 0
    total_test_images = 0
    total_train_annotations = 0
    total_test_annotations = 0
    
    for subdir in subdirs:
        train_json = subdir / "images" / "train" / "train_annotations.json"
        test_json = subdir / "images" / "test" / "test_annotations.json"
        
        if not train_json.exists() or not test_json.exists():
            continue
        
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        
        print(f"\n{subdir.name}:")
        print(f"  Train: {len(train_data['images']):>4} images, {len(train_data['annotations']):>6} annotations")
        print(f"  Test:  {len(test_data['images']):>4} images, {len(test_data['annotations']):>6} annotations")
        
        total_train_images += len(train_data['images'])
        total_test_images += len(test_data['images'])
        total_train_annotations += len(train_data['annotations'])
        total_test_annotations += len(test_data['annotations'])
    
    print(f"\n{'='*70}")
    print(f"TOTAL:")
    print(f"  Train: {total_train_images:>4} images, {total_train_annotations:>6} annotations")
    print(f"  Test:  {total_test_images:>4} images, {total_test_annotations:>6} annotations")
    print(f"  GRAND TOTAL: {total_train_images + total_test_images} images, {total_train_annotations + total_test_annotations} annotations")
    print("="*70)


def create_merged_datasets(output_dir):
    """
    Create merged train/test JSONs across all datasets
    """
    print("\n" + "="*70)
    print("Creating Merged Datasets")
    print("="*70)
    
    output_path = Path(output_dir)
    subdirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    # Merged train
    merged_train = {
        "info": {
            "description": "NIST Sand Dataset - Merged Train Set",
            "version": "1.0",
            "year": 2025,
            "contributor": "NIST",
            "date_created": "2025/01/06"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [
            {"id": 1, "name": "outside_field_of_view", "supercategory": "background"},
            {"id": 2, "name": "air", "supercategory": "object"},
            {"id": 3, "name": "sand", "supercategory": "object"},
            {"id": 4, "name": "capillary_area", "supercategory": "object"},
            {"id": 5, "name": "inclusions", "supercategory": "object"}
        ],
        "images": [],
        "annotations": []
    }
    
    # Merged test
    merged_test = {
        "info": {
            "description": "NIST Sand Dataset - Merged Test Set",
            "version": "1.0",
            "year": 2025,
            "contributor": "NIST",
            "date_created": "2025/01/06"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": merged_train["categories"].copy(),
        "images": [],
        "annotations": []
    }
    
    image_id_offset = 0
    annotation_id_offset = 0
    
    # Merge train sets
    for subdir in sorted(subdirs):
        train_json = subdir / "images" / "train" / "train_annotations.json"
        
        if not train_json.exists():
            continue
        
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        
        # Update image file paths to include dataset name
        for img in train_data['images']:
            img['id'] += image_id_offset
            # Keep relative path from sand_coco root
            img['file_name'] = f"{subdir.name}/images/train/{img['file_name']}"
            merged_train['images'].append(img)
        
        for ann in train_data['annotations']:
            ann['id'] += annotation_id_offset
            ann['image_id'] += image_id_offset
            merged_train['annotations'].append(ann)
        
        if merged_train['images']:
            image_id_offset = max(img['id'] for img in merged_train['images']) + 1
        if merged_train['annotations']:
            annotation_id_offset = max(ann['id'] for ann in merged_train['annotations']) + 1
    
    # Reset offsets for test
    image_id_offset = 0
    annotation_id_offset = 0
    
    # Merge test sets
    for subdir in sorted(subdirs):
        test_json = subdir / "images" / "test" / "test_annotations.json"
        
        if not test_json.exists():
            continue
        
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        
        for img in test_data['images']:
            img['id'] += image_id_offset
            img['file_name'] = f"{subdir.name}/images/test/{img['file_name']}"
            merged_test['images'].append(img)
        
        for ann in test_data['annotations']:
            ann['id'] += annotation_id_offset
            ann['image_id'] += image_id_offset
            merged_test['annotations'].append(ann)
        
        if merged_test['images']:
            image_id_offset = max(img['id'] for img in merged_test['images']) + 1
        if merged_test['annotations']:
            annotation_id_offset = max(ann['id'] for ann in merged_test['annotations']) + 1
    
    # Save merged JSONs in root of sand_coco
    merged_train_path = output_path / "all_train_annotations.json"
    merged_test_path = output_path / "all_test_annotations.json"
    
    with open(merged_train_path, 'w') as f:
        json.dump(merged_train, f, indent=2)
    
    with open(merged_test_path, 'w') as f:
        json.dump(merged_test, f, indent=2)
    
    print(f"  ✓ Merged train: {merged_train_path}")
    print(f"    {len(merged_train['images'])} images, {len(merged_train['annotations'])} annotations")
    print(f"  ✓ Merged test: {merged_test_path}")
    print(f"    {len(merged_test['images'])} images, {len(merged_test['annotations'])} annotations")


if __name__ == "__main__":
    
    # Configuration
    INPUT_DIR = "sand_data"
    OUTPUT_DIR = "sand_coco"
    VAL_RATIO = 0.2  # 20% test split
    RANDOM_SEED = 42
    
    print("\n" + "="*70)
    print("SAND DATA TO COCO CONVERTER")
    print("="*70)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Test split: {VAL_RATIO*100:.0f}%")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*70)
    
    # Step 1: Create COCO format with file structure
    create_coco_with_file_structure(
        base_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
    )
    
    # Step 2: Create summary report
    create_summary_report(OUTPUT_DIR)
    
    # Step 3: Create merged datasets
    create_merged_datasets(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nOutput structure in '{OUTPUT_DIR}':")
    print("sand_coco/")
    print("├── all_train_annotations.json    (merged all datasets)")
    print("├── all_test_annotations.json     (merged all datasets)")
    print("├── 2-bm-aps/")
    print("│   ├── images/")
    print("│   │   ├── train/")
    print("│   │   │   ├── image1.tiff")
    print("│   │   │   └── train_annotations.json")
    print("│   │   └── test/")
    print("│   │       ├── image2.tiff")
    print("│   │       └── test_annotations.json")
    print("│   └── masks/")
    print("│       └── Mask_*.tiff")
    print("├── diad-diomand/")
    print("│   └── ...")
    print("└── ... (other datasets)")
    print(f"\nOriginal data in '{INPUT_DIR}' is UNCHANGED")