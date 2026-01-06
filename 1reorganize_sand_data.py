import os
import shutil
import csv
from pathlib import Path

# Read the CSV file
csv_file = './nist-sand/annotations/annotations_paths.csv'  # Adjust this path if needed
output_base = './sand_data'

# Read and process the CSV
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for idx, row in enumerate(reader):
        filepath = row['filepath'].replace('/global/cfs/cdirs/als', '.')
        maskpath = row['maskpath'].replace('/global/cfs/cdirs/als', '.')
        
        # Extract facility name from the filepath
        # Path pattern: ./nist-sand/{facility}/...
        path_parts = filepath.split('/')
        facility_folder = path_parts[2] if len(path_parts) > 2 else 'unknown'
        
        # Create directory structure if it doesn't exist
        os.makedirs(os.path.join(output_base, facility_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base, facility_folder, 'masks'), exist_ok=True)
        
        # Get the mask filename from the mask path
        mask_filename = os.path.basename(maskpath)
        
        # Create the image filename by removing 'mask_' or 'Mask_' prefix
        if mask_filename.startswith('mask_'):
            image_filename = mask_filename[5:]  # Remove 'mask_'
        elif mask_filename.startswith('Mask_'):
            image_filename = mask_filename[5:]  # Remove 'Mask_'
        else:
            image_filename = mask_filename
        
        # Define destination paths
        dest_image_path = os.path.join(output_base, facility_folder, 'images', image_filename)
        dest_mask_path = os.path.join(output_base, facility_folder, 'masks', mask_filename)
        
        # Copy files if they exist
        if os.path.exists(filepath):
            print(f"Copying image: {filepath} -> {dest_image_path}")
            shutil.copy2(filepath, dest_image_path)
        else:
            print(f"Warning: Image file not found: {filepath}")
        
        if os.path.exists(maskpath):
            print(f"Copying mask: {maskpath} -> {dest_mask_path}")
            shutil.copy2(maskpath, dest_mask_path)
        else:
            print(f"Warning: Mask file not found: {maskpath}")

print("\nReorganization complete!")
print(f"\nNew structure created in: {output_base}")

# Count files in each facility
print("\nFolder structure:")
for facility_folder in sorted(os.listdir(output_base)):
    facility_path = os.path.join(output_base, facility_folder)
    if os.path.isdir(facility_path):
        img_count = len(os.listdir(os.path.join(facility_path, 'images'))) if os.path.exists(os.path.join(facility_path, 'images')) else 0
        mask_count = len(os.listdir(os.path.join(facility_path, 'masks'))) if os.path.exists(os.path.join(facility_path, 'masks')) else 0
        print(f"  {facility_folder}/")
        print(f"    images/ ({img_count} files)")
        print(f"    masks/ ({mask_count} files)")
