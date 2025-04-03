# MIT License
# Copyright geocine
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import logging
import re
import sys
import json
import torch
import argparse
from PIL import Image
import shutil
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load Florence-2 model and processor
def load_model():
    logging.info("Loading Florence-2 model and processor...")
    model = AutoModelForCausalLM.from_pretrained(
        "MiaoshouAI/Florence-2-base-PromptGen-v2.0", 
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "MiaoshouAI/Florence-2-base-PromptGen-v2.0", 
        trust_remote_code=True
    )
    logging.info("Model and processor loaded successfully")
    return model, processor

# Global model and processor to avoid reloading for each image
model, processor = load_model()

def caption_image(image_path, prompt="<DETAILED_CAPTION>"):
    logging.info(f"Captioning image: {image_path}")
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process image and prompt
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
        
        # Decode generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process the generated text
        caption = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        logging.info(f"Caption generated successfully")
        logging.info(f"Caption type: {type(caption)}")
        
        # Handle the case where caption is a dictionary
        if isinstance(caption, dict):
            # Log the keys to help with debugging
            logging.info(f"Caption keys: {caption.keys()}")
            
            # Check for the <DETAILED_CAPTION> key specifically
            if prompt in caption:
                return caption[prompt]
            # Try other common keys
            elif 'caption' in caption:
                return caption['caption']
            elif 'text' in caption:
                return caption['text']
            elif 'description' in caption:
                return caption['description']
            else:
                # If we can't find a specific key, convert the whole dict to a string
                return json.dumps(caption)
        
        return caption
    
    except Exception as e:
        logging.error(f"Error generating caption: {str(e)}")
        return None

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

def resize_image(input_path, output_path, size=(512, 512)):
    with Image.open(input_path) as img:
        img.thumbnail(size)
        img.save(output_path)

def rename_and_process_images(input_folder, output_folder, prefix, metadata_path=None, clear_existing=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create parent directory for metadata file if provided
    if metadata_path and not os.path.exists(os.path.dirname(metadata_path)):
        os.makedirs(os.path.dirname(metadata_path))

    # Define JSONL output path
    jsonl_output_path = metadata_path if metadata_path else os.path.join(output_folder, "metadata.jsonl")

    counter = 1
    jsonl_data = []
    
    # If metadata file exists and we're not clearing, load existing data
    if os.path.exists(jsonl_output_path) and not clear_existing:
        try:
            with open(jsonl_output_path, 'r', encoding='utf-8') as jsonl_file:
                for line in jsonl_file:
                    jsonl_data.append(json.loads(line.strip()))
            logging.info(f"Loaded {len(jsonl_data)} existing entries from metadata file")
        except Exception as e:
            logging.error(f"Error loading existing metadata: {str(e)}")
            jsonl_data = []
    
    # Get set of already processed filenames to avoid duplicates
    processed_filenames = {entry['filename'] for entry in jsonl_data}
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            old_path = os.path.join(input_folder, filename)
            new_filename = f"{prefix}_{counter}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(output_folder, new_filename)
            
            # Check if this image has already been processed
            caption_filename = f"{os.path.splitext(new_filename)[0]}.txt"
            caption_path = os.path.join(output_folder, caption_filename)
            
            # Skip if already processed and not clearing
            if new_filename in processed_filenames and not clear_existing and os.path.exists(caption_path):
                logging.info(f"Skipping {new_filename} - already processed")
                counter += 1
                continue
            
            # Copy image to output folder
            shutil.copy2(old_path, new_path)
            
            # Get original image dimensions
            width, height = get_image_dimensions(new_path)
            
            # Get caption directly from the original image
            caption_result = caption_image(new_path)
            
            # Save caption
            if caption_result:
                # Ensure caption is a string for both the JSONL file and text file
                caption_str = caption_result
                if not isinstance(caption_result, str):
                    caption_str = str(caption_result)
                
                # Create JSONL entry
                jsonl_entry = {
                    "filename": new_filename,
                    "caption_or_tags": caption_str,
                    "width": width,
                    "height": height,
                    "is_tag_based": False,
                    "is_url_based": False,
                    "loss_weight": 1.0
                }
                
                # Remove existing entry if present
                jsonl_data = [entry for entry in jsonl_data if entry['filename'] != new_filename]
                jsonl_data.append(jsonl_entry)
                
                # Also save caption as text file
                with open(caption_path, "w", encoding='utf-8') as caption_file:
                    caption_file.write(caption_str)
                logging.info(f"Caption saved for {new_filename}")
            else:
                logging.error(f"Failed to get caption for {new_filename}")
            
            counter += 1

    # Write JSONL file
    with open(jsonl_output_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + '\n')
    logging.info(f"JSONL file created at {jsonl_output_path}")
    
    logging.info(f"Processed {counter-1} images")

# Usage
if __name__ == "__main__":
    logging.info("Starting image renaming and captioning process")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process and caption images')
    parser.add_argument('--clear', action='store_true', help='Clear existing captions and regenerate all')
    args = parser.parse_args()
    
    clear_existing = args.clear
    if clear_existing:
        logging.info("--clear flag detected. Will regenerate all captions.")
    
    # Hardcoded paths
    input_folder = "images"
    output_folder = "dataset/images"
    metadata_file = "dataset/metadata.jsonl"
    prefix = "training"  # Default prefix for renamed images
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Note: Input folder '{input_folder}' didn't exist and has been created.")
        print(f"Please put your images in the '{input_folder}' directory and run the script again.")
        sys.exit(0)
    
    if not os.listdir(input_folder):
        print(f"Error: Input folder '{input_folder}' is empty. Please add images to process.")
        sys.exit(1)
    
    rename_and_process_images(input_folder, output_folder, prefix, metadata_file, clear_existing)
    logging.info("Process completed")