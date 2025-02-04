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
import zipfile
from io import BytesIO

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
    
    # Initialize session state variables
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'regions_map' not in st.session_state:
        st.session_state.regions_map = {}
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'stacked_previews' not in st.session_state:
        st.session_state.stacked_previews = {}
    if 'preview_generated' not in st.session_state:
        st.session_state.preview_generated = False
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'preview_file_index' not in st.session_state:
        st.session_state.preview_file_index = 0
    if 'first_page_previews' not in st.session_state:
        st.session_state.first_page_previews = {}
    
    # Create three main columns for consistent layout
    step_col, control_col, preview_col = st.columns([0.2, 0.3, 0.5])
    
    # Left Column: Steps and Navigation
    with step_col:
        st.title("Steps")
        step_names = {
            1: "Upload Files",
            2: "Draw Regions",
            3: "Process Files"
        }
        
        for step_num, step_name in step_names.items():
            if st.button(
                f"{step_name} {'✓' if st.session_state.step > step_num else ''}",
                disabled=st.session_state.step < step_num,
                key=f"step_{step_num}"
            ):
                st.session_state.step = step_num
                if step_num == 2:  # Reset canvas when returning to drawing step
                    st.session_state.canvas_key += 1
        
        # Show progress information
        if st.session_state.step > 1 and st.session_state.uploaded_files:
            st.write("### Progress")
            total_files = len(st.session_state.uploaded_files)
            processed_files = len(st.session_state.processed_files)
            st.write(f"Files: {processed_files} / {total_files}")
            
            if st.session_state.current_file:
                st.write("Current file:")
                st.write(os.path.basename(st.session_state.current_file))
    
    # Middle Column: Controls
    with control_col:
        if st.session_state.step == 1:
            st.header("Upload Files")
            uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
            
            if uploaded_files:
                # File handling code remains the same...
                current_filenames = {os.path.basename(f) for f in st.session_state.uploaded_files}
                new_filenames = {f.name for f in uploaded_files}
                
                if current_filenames != new_filenames:
                    # Reset state code remains the same...
                    st.session_state.uploaded_files = []
                    st.session_state.regions_map = {}
                    st.session_state.processed_files = set()
                    st.session_state.processing_complete = False
                    st.session_state.stacked_previews = {}
                    st.session_state.preview_generated = False
                    st.session_state.first_page_previews = {}
                    
                    # Save files code remains the same...
                    os.makedirs("temp_uploads", exist_ok=True)
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("temp_uploads", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        st.session_state.uploaded_files.append(file_path)
                
                st.write(f"Total files: {len(st.session_state.uploaded_files)}")
                for file_path in st.session_state.uploaded_files:
                    st.write(f"- {os.path.basename(file_path)}")
                
                st.write("### Preview Settings")
                alpha = st.slider("Page Transparency", 0.0, 1.0, 0.3, 0.1)
                
                st.write("### Actions")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Stacked Previews", type="primary"):
                        with st.spinner("Generating previews..."):
                            progress = st.progress(0)
                            for idx, file_path in enumerate(st.session_state.uploaded_files):
                                stacked = create_stacked_preview(file_path, alpha=alpha)
                                if stacked is not None:
                                    if stacked.mode == 'RGBA':
                                        white_bg = Image.new('RGB', stacked.size, 'white')
                                        white_bg.paste(stacked, mask=stacked.split()[3])
                                        stacked = white_bg
                                    st.session_state.stacked_previews[file_path] = stacked
                                progress.progress((idx + 1) / len(st.session_state.uploaded_files))
                            st.session_state.preview_generated = True
                            st.rerun()
                
                with col2:
                    if st.button("Start Processing", type="primary"):
                        st.session_state.step = 2
                        st.session_state.current_file = st.session_state.uploaded_files[0]
                        # Use first page preview if no stacked preview available
                        if st.session_state.current_file not in st.session_state.stacked_previews:
                            if st.session_state.current_file not in st.session_state.first_page_previews:
                                preview = load_first_page(st.session_state.current_file, stack_preview=False)
                                st.session_state.first_page_previews[st.session_state.current_file] = preview
                            st.session_state.current_image = st.session_state.first_page_previews[st.session_state.current_file]
                        else:
                            st.session_state.current_image = st.session_state.stacked_previews[st.session_state.current_file]
                        st.session_state.canvas_key += 1
                        st.rerun()

        elif st.session_state.step == 2:
            st.header("Draw Regions")
            if st.session_state.current_file:
                # Use appropriate preview based on what's available
                if st.session_state.current_file in st.session_state.stacked_previews:
                    current_preview = st.session_state.stacked_previews[st.session_state.current_file]
                    preview_type = "stacked"
                else:
                    if st.session_state.current_file not in st.session_state.first_page_previews:
                        preview = load_first_page(st.session_state.current_file, stack_preview=False)
                        st.session_state.first_page_previews[st.session_state.current_file] = preview
                    current_preview = st.session_state.first_page_previews[st.session_state.current_file]
                    preview_type = "first page"
                
                st.write(f"Drawing on {preview_type} preview")
                
                # Calculate scaling for canvas
                DISPLAY_HEIGHT = 600
                scale_factor = DISPLAY_HEIGHT / current_preview.height
                display_width = int(current_preview.width * scale_factor)
                
                # Canvas controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear Regions"):
                        if st.session_state.current_file in st.session_state.regions_map:
                            del st.session_state.regions_map[st.session_state.current_file]
                        st.session_state.canvas_key += 1
                        st.rerun()
                
                with col2:
                    if st.button("Process Current", type="primary"):
                        if st.session_state.current_file in st.session_state.regions_map:
                            scanner = AssignmentScanner()
                            result = scanner.process_pdf(
                                st.session_state.current_file,
                                st.session_state.regions_map.get(st.session_state.current_file, [])
                            )
                            if result[0] is not None:
                                st.session_state.processed_files.add(st.session_state.current_file)
                                st.session_state.current_result = result
                                st.rerun()
                
                # Drawing canvas
                display_image = current_preview.resize(
                    (display_width, DISPLAY_HEIGHT), 
                    Image.Resampling.LANCZOS
                )
                
                # Show existing regions if any
                existing_regions = st.session_state.regions_map.get(st.session_state.current_file, [])
                existing_objects = []
                if existing_regions:
                    for region in existing_regions:
                        # Convert back to display coordinates
                        display_region = {
                            'type': 'rect',
                            'left': region[0] * scale_factor,
                            'top': region[1] * scale_factor,
                            'width': (region[2] - region[0]) * scale_factor,
                            'height': (region[3] - region[1]) * scale_factor,
                            'fill': 'rgba(255, 165, 0, 0.3)',
                            'stroke': '#e00',
                            'strokeWidth': 2
                        }
                        existing_objects.append(display_region)
                
                # Track previous number of objects to detect new region completion
                prev_objects_key = f"prev_objects_{st.session_state.current_file}"
                if prev_objects_key not in st.session_state:
                    st.session_state[prev_objects_key] = len(existing_objects)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#e00",
                    background_image=display_image,
                    drawing_mode="rect",
                    initial_drawing={"version": "4.4.0", "objects": existing_objects} if existing_objects else None,
                    key=f"canvas_{st.session_state.canvas_key}",
                    height=DISPLAY_HEIGHT,
                    width=display_width,
                    update_streamlit=True
                )
                
                # Handle canvas drawing results
                if (canvas_result.json_data is not None and 
                    canvas_result.json_data.get("objects") and 
                    len(canvas_result.json_data["objects"]) > st.session_state[prev_objects_key]):
                    # Only update when a new region is completed
                    regions = canvas_result.json_data["objects"]
                    # Convert drawn regions back to original image coordinates
                    current_regions = [
                        (
                            max(0, int(r['left'] / scale_factor)),
                            max(0, int(r['top'] / scale_factor)),
                            min(current_preview.width, int((r['left'] + r['width']) / scale_factor)),
                            min(current_preview.height, int((r['top'] + r['height']) / scale_factor))
                        )
                        for r in regions
                    ]
                    st.session_state.regions_map[st.session_state.current_file] = current_regions
                    st.session_state[prev_objects_key] = len(regions)
                    st.write(f"Saved {len(current_regions)} regions for {os.path.basename(st.session_state.current_file)}")
                
                # Navigation buttons with file info
                remaining_files = [f for f in st.session_state.uploaded_files 
                                 if f not in st.session_state.processed_files]
                
                # Show navigation controls
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                
                with nav_col1:
                    # Get current file index
                    current_idx = st.session_state.uploaded_files.index(st.session_state.current_file)
                    if current_idx > 0:
                        if st.button("◀ Previous"):
                            prev_file = st.session_state.uploaded_files[current_idx - 1]
                            st.session_state.current_file = prev_file
                            if prev_file in st.session_state.stacked_previews:
                                st.session_state.current_image = st.session_state.stacked_previews[prev_file]
                            else:
                                if prev_file not in st.session_state.first_page_previews:
                                    preview = load_first_page(prev_file, stack_preview=False)
                                    st.session_state.first_page_previews[prev_file] = preview
                                st.session_state.current_image = st.session_state.first_page_previews[prev_file]
                            st.session_state.canvas_key += 1
                            st.rerun()
                
                with nav_col2:
                    st.write(f"File {current_idx + 1} of {len(st.session_state.uploaded_files)}")
                    st.write(os.path.basename(st.session_state.current_file))
                    if st.session_state.current_file in st.session_state.regions_map:
                        st.write(f"Regions: {len(st.session_state.regions_map[st.session_state.current_file])}")
                
                with nav_col3:
                    if remaining_files:
                        if st.button("Next ▶", type="primary"):
                            next_file = remaining_files[0]
                            st.session_state.current_file = next_file
                            if next_file in st.session_state.stacked_previews:
                                st.session_state.current_image = st.session_state.stacked_previews[next_file]
                            else:
                                if next_file not in st.session_state.first_page_previews:
                                    preview = load_first_page(next_file, stack_preview=False)
                                    st.session_state.first_page_previews[next_file] = preview
                                st.session_state.current_image = st.session_state.first_page_previews[next_file]
                            st.session_state.canvas_key += 1
                            st.rerun()
    
    # Right Column: Previews and Results
    with preview_col:
        if st.session_state.step == 1 and st.session_state.uploaded_files:
            st.header("Previews")
            
            # File navigation controls
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("◀", disabled=st.session_state.preview_file_index == 0):
                    st.session_state.preview_file_index = max(0, st.session_state.preview_file_index - 1)
                    st.rerun()
            
            with col2:
                current_file = st.session_state.uploaded_files[st.session_state.preview_file_index]
                st.write(f"File {st.session_state.preview_file_index + 1} of {len(st.session_state.uploaded_files)}")
                st.write(os.path.basename(current_file))
            
            with col3:
                if st.button("▶", disabled=st.session_state.preview_file_index >= len(st.session_state.uploaded_files) - 1):
                    st.session_state.preview_file_index = min(len(st.session_state.uploaded_files) - 1, 
                                                            st.session_state.preview_file_index + 1)
                    st.rerun()
            
            # Show preview based on whether stacked preview is available
            current_file = st.session_state.uploaded_files[st.session_state.preview_file_index]
            
            if current_file in st.session_state.stacked_previews:
                # Show stacked preview
                preview_image = st.session_state.stacked_previews[current_file]
                st.image(preview_image, caption="Stacked Preview", use_column_width=True)
            else:
                # Show or generate first page preview
                if current_file not in st.session_state.first_page_previews:
                    try:
                        preview = load_first_page(current_file, stack_preview=False)
                        st.session_state.first_page_previews[current_file] = preview
                    except Exception as e:
                        st.error(f"Error loading preview: {str(e)}")
                        preview = None
                else:
                    preview = st.session_state.first_page_previews[current_file]
                
                if preview is not None:
                    st.image(preview, caption="First Page Preview", use_column_width=True)
        
        elif st.session_state.step == 2:
            st.header("Results")
            # Show current regions
            if st.session_state.current_file in st.session_state.regions_map:
                current_regions = st.session_state.regions_map[st.session_state.current_file]
                st.write(f"Regions: {len(current_regions)}")
                
                # Show processed result if available
                if st.session_state.current_result is not None:
                    try:
                        img, filename = st.session_state.current_result
                        if img is not None and filename is not None:
                            preview_height = 400
                            preview_scale = preview_height / img.height
                            preview_width = int(img.width * preview_scale)
                            preview_img = img.resize(
                                (preview_width, preview_height),
                                Image.Resampling.LANCZOS
                            )
                            st.image(preview_img, caption=filename)
                            
                            scanner = AssignmentScanner()
                            jpg_bytes = scanner.get_image_bytes(img)
                            st.download_button(
                                label=f"Download {filename}",
                                data=jpg_bytes,
                                file_name=filename,
                                mime="image/jpeg",
                                type="primary"
                            )
                        else:
                            st.error("Failed to process the current file. The result was empty.")
                    except (TypeError, ValueError, AttributeError) as e:
                        st.error(f"Error displaying result: {str(e)}")
                        st.session_state.current_result = None
            
            # Show completion options
            if not remaining_files:
                st.success("All files processed!")
                if len(st.session_state.processed_files) > 1:
                    st.write("### Download All")
                    scanner = AssignmentScanner()
                    results = scanner.process_multiple_pdfs(
                        list(st.session_state.processed_files),
                        st.session_state.regions_map
                    )
                    if results:
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                            for img, filename in results:
                                img_bytes = scanner.get_image_bytes(img)
                                zip_file.writestr(filename, img_bytes)
                        st.download_button(
                            label="Download All as ZIP",
                            data=zip_buffer.getvalue(),
                            file_name="processed_files.zip",
                            mime="application/zip",
                            type="primary"
                        )


if __name__ == "__main__":
    main() 