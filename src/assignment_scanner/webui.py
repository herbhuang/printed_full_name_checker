"""
Web UI for configuring and running the Assignment Scanner.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas
import pytesseract
import io

from .scanner import AssignmentScanner


def create_stacked_preview(pdf_path: str, dpi: int = 300, alpha: float = 0.3) -> Image.Image:
    """
    Create a preview image with all pages stacked with transparency.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for conversion
        alpha: Transparency level for each page (0.0 to 1.0)
        
    Returns:
        PIL Image with all pages stacked
    """
    # Convert all pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        return None
        
    # Use first page as base
    base_image = pages[0].copy()
    
    # Convert to RGBA to support transparency
    base_image = base_image.convert('RGBA')
    
    # Stack subsequent pages with transparency
    for page in pages[1:]:
        # Convert page to RGBA
        overlay = page.convert('RGBA')
        
        # Ensure same size as base image
        if overlay.size != base_image.size:
            overlay = overlay.resize(base_image.size, Image.Resampling.LANCZOS)
        
        # Create transparent version of the overlay
        transparent = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        transparent.paste(overlay, (0, 0))
        
        # Adjust alpha for this layer
        data = transparent.getdata()
        newData = [(r, g, b, int(a * alpha)) for r, g, b, a in data]
        transparent.putdata(newData)
        
        # Composite the images
        base_image = Image.alpha_composite(base_image, transparent)
    
    return base_image


def load_first_page(pdf_path: str, dpi: int = 300, stack_preview: bool = True, alpha: float = 0.3) -> Image.Image:
    """
    Load either first page or stacked preview of all pages.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for conversion
        stack_preview: Whether to create stacked preview
        alpha: Transparency level for stacked pages
    """
    if stack_preview:
        return create_stacked_preview(pdf_path, dpi, alpha)
    else:
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        return images[0]


def save_config(config: Dict, output_path: str):
    """Save region configuration to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path: str) -> Dict:
    """Load region configuration from a JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def process_image_with_ocr(image: Image.Image, region: Tuple[int, int, int, int]) -> Tuple[str, Image.Image]:
    """Process image region with OCR and return text and visualization."""
    # Crop image to region
    region_image = image.crop(region)
    
    # Convert to OpenCV format for preprocessing
    cv_image = cv2.cvtColor(np.array(region_image), cv2.COLOR_RGB2BGR)
    
    # Preprocess image
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Convert back to PIL for OCR
    processed_image = Image.fromarray(denoised)
    
    # Perform OCR with custom configuration
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "'
    text = pytesseract.image_to_string(processed_image, config=custom_config).strip()
    
    # Create visualization
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Draw rectangle around region
    draw.rectangle(region, outline='red', width=2)
    
    # Draw text above the region
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text_position = (region[0], max(0, region[1] - 30))
    draw.text(text_position, text, fill='red', font=font)
    
    return text, vis_image


def stitch_regions_vertically(image: Image.Image, regions: List[Tuple[int, int, int, int]], padding: int = 10) -> Image.Image:
    """
    Crop regions from the image and stitch them vertically into a single image.
    
    Args:
        image: Source image
        regions: List of (x1, y1, x2, y2) coordinates
        padding: Padding between regions in pixels
        
    Returns:
        Combined image with all regions stitched vertically
    """
    if not regions:
        return None
        
    # Crop all regions
    cropped_regions = [image.crop(region) for region in regions]
    
    # Calculate dimensions for the combined image
    total_height = sum(img.height for img in cropped_regions) + padding * (len(regions) - 1)
    max_width = max(img.width for img in cropped_regions)
    
    # Create new image with white background
    combined_image = Image.new('RGB', (max_width, total_height), 'white')
    
    # Paste all regions vertically
    current_y = 0
    for img in cropped_regions:
        # Center the image horizontally if it's narrower than the widest one
        x_offset = (max_width - img.width) // 2
        combined_image.paste(img, (x_offset, current_y))
        current_y += img.height + padding
        
    return combined_image


def save_image_to_jpg(image: Image.Image, quality: int = 95) -> bytes:
    """Convert PIL Image to JPG bytes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=quality)
    return img_byte_arr.getvalue()


def main():
    st.set_page_config(layout="wide")
    st.title("Assignment Scanner Configuration")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'regions_map' not in st.session_state:
        st.session_state.regions_map = {}  # Maps file paths to their regions
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()  # Track which files have been processed
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    # Sidebar for navigation and controls
    st.sidebar.title("Steps")
    step_names = {
        1: "Upload Files",
        2: "Draw Regions",
        3: "Process Files"
    }
    
    for step_num, step_name in step_names.items():
        if st.sidebar.button(
            f"{step_name} {'✓' if st.session_state.step > step_num else ''}",
            disabled=st.session_state.step < step_num
        ):
            st.session_state.step = step_num
            if step_num == 2:  # Reset canvas when returning to drawing step
                st.session_state.canvas_key += 1
    
    # Step 1: File Upload
    if st.session_state.step == 1:
        st.header("Step 1: Upload PDF Files")
        
        # Add transparency control
        alpha = st.slider("Page Transparency", 0.0, 1.0, 0.3, 0.1,
                         help="Adjust transparency level for stacked page preview")
        
        uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
            
        if uploaded_files:
            st.session_state.uploaded_files = []
            st.session_state.regions_map = {}
            st.session_state.processed_files = set()
            st.session_state.processing_complete = False
            
            # Save all uploaded files temporarily with their original names
            os.makedirs("temp_uploads", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("temp_uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.uploaded_files.append(file_path)
            
            st.write(f"Total files uploaded: {len(st.session_state.uploaded_files)}")
            for file_path in st.session_state.uploaded_files:
                st.write(f"- {os.path.basename(file_path)}")
                # Show stacked preview for each file
                try:
                    preview = load_first_page(file_path, stack_preview=True, alpha=alpha)
                    st.image(preview, caption=f"Stacked preview: {os.path.basename(file_path)}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error creating preview for {os.path.basename(file_path)}: {str(e)}")
            
            if st.button("Start Processing Files"):
                st.session_state.step = 2
                st.session_state.current_file = st.session_state.uploaded_files[0]
                st.session_state.current_image = load_first_page(st.session_state.current_file, stack_preview=True, alpha=alpha)
                st.session_state.canvas_key += 1
                st.experimental_rerun()

    # Step 2: Region Selection and Processing
    elif st.session_state.step == 2:
        if not st.session_state.current_file:
            st.error("No file selected for processing")
            return

        current_filename = os.path.basename(st.session_state.current_file)
        st.header(f"Processing: {current_filename}")
        
        # Add transparency control
        alpha = st.slider("Page Transparency", 0.0, 1.0, 0.3, 0.1,
                         help="Adjust transparency level for stacked page preview")
        
        # Show file progress
        total_files = len(st.session_state.uploaded_files)
        processed_files = len(st.session_state.processed_files)
        st.write(f"Processing file {processed_files + 1} of {total_files}")
        
        # Update current image with new alpha value
        st.session_state.current_image = load_first_page(
            st.session_state.current_file,
            stack_preview=True,
            alpha=alpha
        )
        
        # Region drawing interface
        st.write("Draw rectangles around the regions to extract.")
        
        # Calculate scaling factor
        DISPLAY_HEIGHT = 800
        scale_factor = DISPLAY_HEIGHT / st.session_state.current_image.height
        display_width = int(st.session_state.current_image.width * scale_factor)
        
        # Resize image for display
        display_image = st.session_state.current_image.resize(
            (display_width, DISPLAY_HEIGHT), 
            Image.Resampling.LANCZOS
        )

        # Clear regions button for current file
        if st.button("Clear Regions"):
            if st.session_state.current_file in st.session_state.regions_map:
                del st.session_state.regions_map[st.session_state.current_file]
            st.session_state.canvas_key += 1
            st.experimental_rerun()
        
        # Canvas for drawing regions
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e00",
            background_image=display_image,
            drawing_mode="rect",
            key=f"canvas_{st.session_state.canvas_key}",
            height=DISPLAY_HEIGHT,
            width=display_width,
            update_streamlit=True
        )
        
        # Process drawn regions
        if canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
            regions = canvas_result.json_data["objects"]
            # Update regions with proper scaling
            current_regions = [
                (
                    max(0, int(r['left'] / scale_factor)),
                    max(0, int(r['top'] / scale_factor)),
                    min(st.session_state.current_image.width, int((r['left'] + r['width']) / scale_factor)),
                    min(st.session_state.current_image.height, int((r['top'] + r['height']) / scale_factor))
                )
                for r in regions
            ]
            st.session_state.regions_map[st.session_state.current_file] = current_regions
            
            # Show current regions
            st.write("### Current Regions:")
            for i, region in enumerate(current_regions, 1):
                st.write(f"Region {i}: (x1={region[0]}, y1={region[1]}, x2={region[2]}, y2={region[3]})")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process Current File"):
                # Process current file
                scanner = AssignmentScanner()
                result = scanner.process_pdf(
                    st.session_state.current_file,
                    st.session_state.regions_map.get(st.session_state.current_file, [])
                )
                
                if result[0] is not None:
                    # Save result
                    st.session_state.processed_files.add(st.session_state.current_file)
                    
                    # Show preview
                    img, filename = result
                    st.write(f"### Processed: {filename}")
                    
                    # Calculate display size
                    preview_height = 400
                    preview_scale = preview_height / img.height
                    preview_width = int(img.width * preview_scale)
                    
                    # Show preview
                    preview_img = img.resize(
                        (preview_width, preview_height),
                        Image.Resampling.LANCZOS
                    )
                    st.image(preview_img, caption=filename)
                    
                    # Add download button
                    jpg_bytes = scanner.get_image_bytes(img)
                    st.download_button(
                        label=f"Download {filename}",
                        data=jpg_bytes,
                        file_name=filename,
                        mime="image/jpeg"
                    )
        
        with col2:
            # Move to next file button
            remaining_files = [f for f in st.session_state.uploaded_files 
                             if f not in st.session_state.processed_files]
            
            if remaining_files:
                if st.button("Next File"):
                    next_file = remaining_files[0]
                    st.session_state.current_file = next_file
                    st.session_state.current_image = load_first_page(next_file, stack_preview=True, alpha=alpha)
                    st.session_state.canvas_key += 1
                    st.experimental_rerun()
            else:
                st.session_state.processing_complete = True
                st.success("All files processed!")
                if st.button("Process More Files"):
                    # Clean up
                    for file_path in st.session_state.uploaded_files:
                        try:
                            os.remove(file_path)
                        except:
                            pass
                    try:
                        os.rmdir("temp_uploads")
                    except:
                        pass
                    
                    # Reset state
                    st.session_state.step = 1
                    st.session_state.regions_map = {}
                    st.session_state.current_image = None
                    st.session_state.current_file = None
                    st.session_state.canvas_key = 0
                    st.session_state.uploaded_files = []
                    st.session_state.processed_files = set()
                    st.session_state.processing_complete = False
                    st.experimental_rerun()

        # If processing is complete, show download all button
        if st.session_state.processing_complete and len(st.session_state.processed_files) > 1:
            st.write("### Download All Files")
            # Create ZIP of all processed files
            scanner = AssignmentScanner()
            results = scanner.process_multiple_pdfs(
                list(st.session_state.processed_files),
                st.session_state.regions_map
            )
            
            if results:
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for img, filename in results:
                        img_bytes = scanner.get_image_bytes(img)
                        zip_file.writestr(filename, img_bytes)
                
                st.download_button(
                    label="Download All as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="processed_files.zip",
                    mime="application/zip"
                )


if __name__ == "__main__":
    main() 