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
import base64

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


def stitch_regions_vertically(source_image: Optional[Image.Image] = None, 
                            regions: Optional[List[Tuple[int, int, int, int]]] = None,
                            images: Optional[List[Image.Image]] = None,
                            padding: int = 10) -> Image.Image:
    """
    Crop regions from the image and stitch them vertically into a single image.
    Can either take a source image and regions, or a list of pre-cropped images.
    """
    if images is None and source_image is not None and regions is not None:
        images = [source_image.crop(region) for region in regions]
    
    if not images:
        return None
        
    # Calculate dimensions for the combined image
    total_height = sum(img.height for img in images) + padding * (len(images) - 1)
    max_width = max(img.width for img in images)
    
    # Create new image with white background
    combined_image = Image.new('RGB', (max_width, total_height), 'white')
    
    # Paste all regions vertically
    current_y = 0
    for img in images:
        # Center the image horizontally if it's narrower than the widest one
        x_offset = (max_width - img.width) // 2
        combined_image.paste(img, (x_offset, current_y))
        current_y += img.height + padding
        
    return combined_image


def stitch_regions_horizontally(source_image: Optional[Image.Image] = None,
                              regions: Optional[List[Tuple[int, int, int, int]]] = None,
                              images: Optional[List[Image.Image]] = None,
                              padding: int = 10) -> Image.Image:
    """
    Crop regions from the image and stitch them horizontally into a single image.
    Can either take a source image and regions, or a list of pre-cropped images.
    """
    if images is None and source_image is not None and regions is not None:
        images = [source_image.crop(region) for region in regions]
    
    if not images:
        return None
        
    # Calculate dimensions for the combined image
    total_width = sum(img.width for img in images) + padding * (len(images) - 1)
    max_height = max(img.height for img in images)
    
    # Create new image with white background
    combined_image = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste all regions horizontally
    current_x = 0
    for img in images:
        # Center the image vertically if it's shorter than the tallest one
        y_offset = (max_height - img.height) // 2
        combined_image.paste(img, (current_x, y_offset))
        current_x += img.width + padding
        
    return combined_image


def stitch_regions_grid(source_image: Optional[Image.Image] = None,
                       regions: Optional[List[Tuple[int, int, int, int]]] = None,
                       images: Optional[List[Image.Image]] = None,
                       max_cols: int = 2, padding: int = 10,
                       by_column: bool = False) -> Image.Image:
    """
    Crop regions from the image and arrange them in a grid.
    Can either take a source image and regions, or a list of pre-cropped images.
    If by_column is True, fills grid column by column instead of row by row.
    """
    if images is None and source_image is not None and regions is not None:
        images = [source_image.crop(region) for region in regions]
    
    if not images:
        return None
        
    # Calculate number of rows and columns
    n_images = len(images)
    if by_column:
        n_rows = min(max_cols, n_images)
        n_cols = (n_images + n_rows - 1) // n_rows
    else:
        n_cols = min(max_cols, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
    
    # Calculate maximum dimensions for each cell
    max_cell_width = max(img.width for img in images)
    max_cell_height = max(img.height for img in images)
    
    # Calculate total dimensions
    total_width = n_cols * max_cell_width + (n_cols - 1) * padding
    total_height = n_rows * max_cell_height + (n_rows - 1) * padding
    
    # Create new image with white background
    combined_image = Image.new('RGB', (total_width, total_height), 'white')
    
    # Paste all regions in grid
    for idx, img in enumerate(images):
        if by_column:
            # Fill column by column
            col = idx // n_rows
            row = idx % n_rows
        else:
            # Fill row by row
            row = idx // n_cols
            col = idx % n_cols
        
        # Calculate position
        x = col * (max_cell_width + padding)
        y = row * (max_cell_height + padding)
        
        # Center the image in its cell
        x_offset = (max_cell_width - img.width) // 2
        y_offset = (max_cell_height - img.height) // 2
        
        combined_image.paste(img, (x + x_offset, y + y_offset))
    
    return combined_image


def save_image_to_jpg(image: Image.Image, quality: int = 100) -> bytes:
    """Convert PIL Image to JPG bytes with maximum quality."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Use PNG for lossless quality
    return img_byte_arr.getvalue()


def apply_regions_to_page(page: Image.Image, regions: List[Tuple[int, int, int, int]], scale_to_height: Optional[int] = None) -> Image.Image:
    """
    Draw regions on a page and optionally scale it.
    """
    # Create a copy to draw on
    page_with_regions = page.copy()
    draw = ImageDraw.Draw(page_with_regions)
    
    # Draw each region
    for i, region in enumerate(regions):
        draw.rectangle(region, outline='red', width=2)
        # Draw region number
        draw.text((region[0], region[1] - 20), f"Region {i+1}", fill='red')
    
    # Scale if requested
    if scale_to_height and scale_to_height != page_with_regions.height:
        scale_factor = scale_to_height / page_with_regions.height
        new_width = int(page_with_regions.width * scale_factor)
        page_with_regions = page_with_regions.resize(
            (new_width, scale_to_height),
            Image.Resampling.LANCZOS
        )
    
    return page_with_regions


def get_region_key(file_path: str, page_num: int, region_num: int) -> str:
    """Generate a unique key for each region."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return f"{base_name}_page{page_num}_region{region_num}"


def add_border(img: Image.Image, border_width: int = 2, border_color: str = 'red') -> Image.Image:
    """Add a colored border to an image."""
    bordered_img = ImageDraw.Draw(img.copy())
    bordered_img.rectangle([(0, 0), (img.width-1, img.height-1)], 
                         outline=border_color, width=border_width)
    return img


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
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "Separate"
    if 'cropped_regions_cache' not in st.session_state:
        st.session_state.cropped_regions_cache = {}
    
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
                # Add redraw state to session state if not exists
                if 'redrawing' not in st.session_state:
                    st.session_state.redrawing = False
                    st.session_state.redraw_page = None
                    st.session_state.redraw_region = None

                if st.session_state.redrawing:
                    # Show redraw interface in the middle column
                    st.write("---")
                    st.write(f"### Redrawing Region {st.session_state.redraw_region} on Page {st.session_state.redraw_page}")
                    
                    # Get the specific page image
                    pages = convert_from_path(st.session_state.current_file)
                    page_img = pages[st.session_state.redraw_page - 1].convert('RGB')
                    
                    # Calculate scaling for canvas
                    DISPLAY_HEIGHT = 600
                    scale_factor = DISPLAY_HEIGHT / page_img.height
                    display_width = int(page_img.width * scale_factor)
                    
                    # Resize image for display
                    display_image = page_img.resize(
                        (display_width, DISPLAY_HEIGHT),
                        Image.Resampling.LANCZOS
                    )
                    
                    # Create columns for buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("Cancel Redraw"):
                            st.session_state.redrawing = False
                            st.rerun()
                    
                    # Drawing canvas for redraw
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="#e00",
                        background_image=display_image,
                        drawing_mode="rect",
                        key=f"redraw_canvas_{st.session_state.canvas_key}",
                        height=DISPLAY_HEIGHT,
                        width=display_width,
                        update_streamlit=True
                    )
                    
                    # Handle redraw result
                    if (canvas_result.json_data is not None and 
                        canvas_result.json_data.get("objects")):
                        regions = canvas_result.json_data["objects"]
                        if regions:  # If a new region is drawn
                            # Convert coordinates back to original scale
                            new_region = regions[-1]  # Get the last drawn region
                            original_coords = (
                                max(0, int(new_region['left'] / scale_factor)),
                                max(0, int(new_region['top'] / scale_factor)),
                                min(page_img.width, int((new_region['left'] + new_region['width']) / scale_factor)),
                                min(page_img.height, int((new_region['top'] + new_region['height']) / scale_factor))
                            )
                            
                            # Update the region in the cache
                            region_img = page_img.crop(original_coords)
                            region_key = get_region_key(
                                st.session_state.current_file,
                                st.session_state.redraw_page,
                                st.session_state.redraw_region
                            )
                            st.session_state.cropped_regions_cache[region_key] = region_img
                            
                            # Exit redraw mode
                            st.session_state.redrawing = False
                            st.rerun()
                else:
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
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Clear Regions"):
                            if st.session_state.current_file in st.session_state.regions_map:
                                del st.session_state.regions_map[st.session_state.current_file]
                            st.session_state.canvas_key += 1
                            st.rerun()
                    
                    with col3:
                        if st.button("Preview Current", type="primary"):
                            if st.session_state.current_file in st.session_state.regions_map:
                                # Process and cache all regions first
                                pages = convert_from_path(st.session_state.current_file)
                                current_regions = st.session_state.regions_map[st.session_state.current_file]
                                
                                for i, page in enumerate(pages):
                                    page_rgb = page.convert('RGB')
                                    scale_x = page_rgb.width / current_preview.width
                                    scale_y = page_rgb.height / current_preview.height
                                    scaled_regions = [
                                        (
                                            int(region[0] * scale_x),
                                            int(region[1] * scale_y),
                                            int(region[2] * scale_x),
                                            int(region[3] * scale_y)
                                        )
                                        for region in current_regions
                                    ]
                                    
                                    # Cache each cropped region
                                    for j, region in enumerate(scaled_regions):
                                        region_img = page_rgb.crop(region)
                                        region_key = get_region_key(st.session_state.current_file, i+1, j+1)
                                        st.session_state.cropped_regions_cache[region_key] = region_img
                                st.rerun()
                    
                    # Drawing canvas
                    display_image = current_preview.resize(
                        (display_width, DISPLAY_HEIGHT), 
                        Image.Resampling.LANCZOS
                    )
                    
                    # Handle editing mode
                    if hasattr(st.session_state, 'editing_region'):
                        st.write(f"Editing Region {st.session_state.editing_region + 1}")
                        if st.button("Cancel Edit"):
                            del st.session_state.editing_region
                            del st.session_state.editing_page
                            st.session_state.canvas_key += 1
                            st.rerun()
                    
                    # Show existing regions if any
                    existing_regions = st.session_state.regions_map.get(st.session_state.current_file, [])
                    existing_objects = []
                    if existing_regions:
                        for i, region in enumerate(existing_regions):
                            # If in editing mode, only show the region being edited
                            if hasattr(st.session_state, 'editing_region') and i != st.session_state.editing_region:
                                continue
                                
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
                        canvas_result.json_data.get("objects")):
                        regions = canvas_result.json_data["objects"]
                        current_regions = [
                            (
                                max(0, int(r['left'] / scale_factor)),
                                max(0, int(r['top'] / scale_factor)),
                                min(current_preview.width, int((r['left'] + r['width']) / scale_factor)),
                                min(current_preview.height, int((r['top'] + r['height']) / scale_factor))
                            )
                            for r in regions
                        ]
                        
                        if hasattr(st.session_state, 'editing_region'):
                            # Update only the edited region
                            existing_regions = st.session_state.regions_map.get(st.session_state.current_file, [])
                            if current_regions:  # If there's a new region drawn
                                existing_regions[st.session_state.editing_region] = current_regions[0]
                                st.session_state.regions_map[st.session_state.current_file] = existing_regions
                                # Clear editing mode
                                del st.session_state.editing_region
                                del st.session_state.editing_page
                                st.session_state.canvas_key += 1
                                st.rerun()
                        else:
                            # Normal mode - handle all regions
                            st.session_state.regions_map[st.session_state.current_file] = current_regions
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
            if st.session_state.current_file in st.session_state.regions_map:
                current_regions = st.session_state.regions_map[st.session_state.current_file]
                st.write(f"Regions: {len(current_regions)}")
                
                # Add tabs for different views
                tab1, tab2 = st.tabs(["Region Preview", "All Cropped Regions"])
                
                with tab1:
                    # Get the current preview image
                    if st.session_state.current_file in st.session_state.stacked_previews:
                        current_preview = st.session_state.stacked_previews[st.session_state.current_file]
                    else:
                        if st.session_state.current_file not in st.session_state.first_page_previews:
                            preview = load_first_page(st.session_state.current_file, stack_preview=False)
                            st.session_state.first_page_previews[st.session_state.current_file] = preview
                        current_preview = st.session_state.first_page_previews[st.session_state.current_file]
                    
                    # Show regions
                    if current_regions:
                        for i, region in enumerate(current_regions):
                            region_img = current_preview.crop(region)
                            st.image(region_img, caption=f"Region {i+1}", use_column_width=True)
                
                with tab2:
                    st.write("### Cropped Regions")
                    
                    if st.session_state.current_file in st.session_state.regions_map:
                        current_regions = st.session_state.regions_map[st.session_state.current_file]
                        num_pages = len(convert_from_path(st.session_state.current_file))
                        
                        # Create a list of all regions from all pages
                        all_regions = []
                        for page_num in range(num_pages):
                            for region_num in range(len(current_regions)):
                                region_key = get_region_key(st.session_state.current_file, page_num+1, region_num+1)
                                if region_key in st.session_state.cropped_regions_cache:
                                    all_regions.append((
                                        page_num + 1,
                                        region_num + 1,
                                        st.session_state.cropped_regions_cache[region_key]
                                    ))
                        
                        # Initialize excluded regions if not exists
                        if 'excluded_regions' not in st.session_state:
                            st.session_state.excluded_regions = set()
                        if 'temp_selections' not in st.session_state:
                            st.session_state.temp_selections = {}
                        
                        # Create table data
                        table_data = []
                        for page_num, region_num, img in all_regions:
                            region_key = f"{page_num}_{region_num}"
                            if region_key not in st.session_state.temp_selections:
                                st.session_state.temp_selections[region_key] = region_key not in st.session_state.excluded_regions
                                
                            # Add red border to the image
                            bordered_img = add_border(img)
                            
                            # Create table row
                            table_data.append({
                                "Preview": bordered_img,
                                "Page": page_num,
                                "Region": region_num,
                                "Edit": region_key,  # We'll use this to create edit buttons
                                "Select": region_key  # We'll use this for checkboxes
                            })
                        
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(table_data)
                        
                        # Display each row with proper formatting
                        for _, row in df.iterrows():
                            cols = st.columns([0.7, 0.06, 0.06, 0.08, 0.1])
                            
                            with cols[0]:
                                st.image(row["Preview"], use_column_width=True)
                            with cols[1]:
                                st.write(str(row["Page"]))
                            with cols[2]:
                                st.write(str(row["Region"]))
                            with cols[3]:
                                if st.button("↺", key=f"edit_btn_{row['Edit']}", help="Redraw region"):
                                    st.session_state.redrawing = True
                                    st.session_state.redraw_page = row["Page"]
                                    st.session_state.redraw_region = row["Region"]
                                    st.session_state.canvas_key += 1
                                    st.rerun()
                            with cols[4]:
                                region_key = row["Select"]
                                st.session_state.temp_selections[region_key] = st.checkbox(
                                    f"Select region {row['Region']} from page {row['Page']}",
                                    value=st.session_state.temp_selections[region_key],
                                    key=f"checkbox_{region_key}",
                                    label_visibility="collapsed"
                                )
                        
                        # Add Process Files button at the bottom
                        if st.button("Process Files", type="primary"):
                            # Collect all selected regions
                            selected_regions = []
                            for key, is_selected in st.session_state.temp_selections.items():
                                if is_selected:
                                    page_num, region_num = map(int, key.split('_'))
                                    region_key = get_region_key(st.session_state.current_file, page_num, region_num)
                                    if region_key in st.session_state.cropped_regions_cache:
                                        selected_regions.append({
                                            'page': page_num,
                                            'region': region_num,
                                            'image': st.session_state.cropped_regions_cache[region_key]
                                        })
                            
                            # Store selected regions for processing
                            st.session_state.selected_regions = selected_regions
                            # Move to next step
                            st.session_state.step = 3
                            st.rerun()
            
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

        elif st.session_state.step == 3:
            with control_col:
                st.header("Process Files")
                
                if hasattr(st.session_state, 'selected_regions') and st.session_state.selected_regions:
                    st.write(f"Selected {len(st.session_state.selected_regions)} regions for processing")
                    
                    # OCR Optimization options
                    st.write("### Image Processing Options")
                    use_ocr_optimization = st.checkbox("Optimize images for OCR", value=True, help="Apply preprocessing to improve OCR accuracy")
                    
                    if use_ocr_optimization:
                        st.write("OCR Optimization will:")
                        st.write("- Convert to grayscale")
                        st.write("- Apply adaptive thresholding")
                        st.write("- Remove noise")
                        st.write("- Enhance contrast")
                    
                    # Stitch mode selection
                    st.write("### Output Format")
                    stitch_mode = st.selectbox(
                        "Select output format",
                        ["Individual Images", "Horizontal Stack", "Vertical Stack", "Grid by Row", "Grid by Column"]
                    )
                    
                    if stitch_mode.startswith("Grid"):
                        cols_per_row = st.number_input("Images per row/column", min_value=1, value=3)
                    
                    # Process button
                    if st.button("Generate Output", type="primary"):
                        with st.spinner("Processing images..."):
                            # Process images based on options
                            processed_images = []
                            for region in st.session_state.selected_regions:
                                img = region['image']
                                if use_ocr_optimization:
                                    # Convert to OpenCV format for preprocessing while maintaining quality
                                    cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                    
                                    # Convert to grayscale
                                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                                    
                                    # Apply adaptive thresholding with better parameters for quality
                                    binary = cv2.adaptiveThreshold(
                                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 21, 11  # Adjusted parameters for better quality
                                    )
                                    
                                    # Denoise with parameters optimized for document images
                                    denoised = cv2.fastNlMeansDenoising(binary, None, h=10, templateWindowSize=7, searchWindowSize=21)
                                    
                                    # Convert back to PIL with high quality
                                    processed_img = Image.fromarray(denoised)
                                else:
                                    processed_img = img.copy()  # Make a copy to preserve original
                                
                                processed_images.append({
                                    'page': region['page'],
                                    'region': region['region'],
                                    'image': processed_img
                                })
                            
                            # Store processed images in session state for preview
                            st.session_state.processed_images = processed_images
                            st.session_state.current_stitch_mode = stitch_mode
                            st.session_state.current_cols_per_row = cols_per_row if stitch_mode.startswith("Grid") else None
                            
                            # Create output based on stitch mode
                            if stitch_mode == "Individual Images":
                                # Create ZIP file with individual images
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_STORED) as zip_file:  # No compression for images
                                    for img_data in processed_images:
                                        img_byte_arr = BytesIO()
                                        img_data['image'].save(img_byte_arr, format='PNG')  # Use PNG format
                                        zip_file.writestr(
                                            f"page{img_data['page']}_region{img_data['region']}.png",
                                            img_byte_arr.getvalue()
                                        )
                                
                                # Store ZIP buffer in session state
                                st.session_state.zip_buffer = zip_buffer.getvalue()
                                st.rerun()
                            else:
                                # Extract just the images for stitching
                                images_to_stitch = [img_data['image'] for img_data in processed_images]
                                
                                # Create stitched image based on mode
                                if stitch_mode == "Horizontal Stack":
                                    final_image = stitch_regions_horizontally(None, None, images_to_stitch)
                                elif stitch_mode == "Vertical Stack":
                                    final_image = stitch_regions_vertically(None, None, images_to_stitch)
                                elif stitch_mode == "Grid by Row":
                                    final_image = stitch_regions_grid(None, None, images_to_stitch, 
                                                                   max_cols=cols_per_row, by_column=False)
                                else:  # Grid by Column
                                    final_image = stitch_regions_grid(None, None, images_to_stitch, 
                                                                   max_cols=cols_per_row, by_column=True)
                                
                                # Store final image in session state
                                st.session_state.final_image = final_image
                                st.rerun()
                    
                    # Option to go back
                    if st.button("← Back to Region Selection"):
                        st.session_state.step = 2
                        st.rerun()
                else:
                    st.error("No regions selected. Please go back and select regions to process.")
                    if st.button("← Back to Region Selection"):
                        st.session_state.step = 2
                        st.rerun()
            
            # Preview column now only shows results
            with preview_col:
                st.header("Preview")
                if hasattr(st.session_state, 'processed_images'):
                    if hasattr(st.session_state, 'zip_buffer'):
                        st.download_button(
                            label="Download All Regions (ZIP)",
                            data=st.session_state.zip_buffer,
                            file_name="processed_regions.zip",
                            mime="application/zip"
                        )
                    elif hasattr(st.session_state, 'final_image'):
                        st.image(st.session_state.final_image, caption="Generated Output", use_column_width=True)
                        
                        # Offer download of stitched image
                        img_byte_arr = BytesIO()
                        st.session_state.final_image.save(img_byte_arr, format='PNG')  # Use PNG format
                        st.download_button(
                            label="Download Stitched Image",
                            data=img_byte_arr.getvalue(),
                            file_name="stitched_regions.png",
                            mime="image/png"
                        )


if __name__ == "__main__":
    main() 