from pillow_heif import register_heif_opener
from PIL import Image
import os

# Enable HEIC support in Pillow
register_heif_opener()

def convert_heic_to_jpg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(input_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)

            try:
                with Image.open(heic_path) as img:
                    rgb_img = img.convert("RGB")
                    rgb_img.save(jpg_path, "JPEG")
                print(f"Converted: {filename} → {jpg_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Example usage
input_dir = r"C:\Users\hp\Downloads\AI Photos (1)top"
output_dir = r"C:\Users\hp\Downloads\AI Photos (1)top\results"
convert_heic_to_jpg(input_dir, output_dir)
