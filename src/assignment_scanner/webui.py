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

from src.assignment_scanner.scanner import AssignmentScanner
from src.assignment_scanner.state_manager import StateManager
from src.assignment_scanner.region_processor import RegionProcessor
from src.assignment_scanner.ui_manager import UIManager


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


def create_stacked_preview_from_pages(pages: List[Image.Image], alpha: float = 0.3) -> Optional[Image.Image]:
    """
    Create a preview image with all pages stacked with transparency using cached pages.
    
    Args:
        pages: List of PIL Image objects
        alpha: Transparency level for each page (0.0 to 1.0)
        
    Returns:
        PIL Image with all pages stacked
    """
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


def main():
    st.set_page_config(layout="wide")
    
    # Add CSS for layout
    st.markdown("""
        <style>
            /* Reset main container */
            .main .block-container {
                padding: 2rem 1rem 1rem;
                max-width: 100%;
            }
            
            /* Reset sidebar */
            section[data-testid="stSidebar"] {
                width: 0px;
            }
            
            /* Fix column layout */
            .stMarkdown {
                min-height: auto;
            }
            
            /* Fix left and middle columns */
            [data-testid="column"]:nth-child(1),
            [data-testid="column"]:nth-child(2) {
                position: sticky !important;
                top: 0;
                background: white;
                z-index: 999;
            }
            
            /* Make right column scrollable */
            [data-testid="column"]:nth-child(3) {
                max-height: calc(100vh - 3rem);
                overflow-y: auto;
                overflow-x: hidden;
            }
            
            /* Target region preview elements in right column */
            [data-testid="column"]:nth-child(3) .e1f1d6gn2 {
                gap: 0.1rem !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            
            /* Basic column padding */
            [data-testid="column"] {
                padding: 0 0.5rem !important;
            }
            
            /* Hide default scrollbar */
            [data-testid="column"]:nth-child(3)::-webkit-scrollbar {
                display: none;
            }
            
            /* Make dividers more compact */
            hr {
                margin: 0.1rem 0 !important;
            }
            
            /* Reset headers */
            h1, h2, h3 {
                margin: 0 0 1rem 0 !important;
                padding: 0 !important;
            }
            
            /* Reset buttons */
            .stButton > button {
                margin: 0 0 0.5rem 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize managers
    state_manager = StateManager()
    region_processor = RegionProcessor(state_manager)
    ui_manager = UIManager(state_manager, region_processor)
    
    # Create three main columns for layout
    step_col, control_col, preview_col = st.columns([0.2, 0.3, 0.5])
    
    # Left Column: Steps and Progress
    with step_col:
        st.title("Steps")
        step_names = {
            1: "Upload File",
            2: "Draw Regions",
            3: "Process Regions"
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
        
        # Show file details if a file is loaded
        if st.session_state.current_file:
            st.write("### File Details")
            st.caption(os.path.basename(st.session_state.current_file))
            
            # Show regions info
            regions = state_manager.get_all_regions(st.session_state.current_file)
            pages = state_manager.get_pdf_pages(st.session_state.current_file)
            
            if pages:
                total_pages = len(pages)
                pages_with_regions = len({r.page_idx for r in regions}) if regions else 0
                
                # Progress bar for pages with regions
                st.progress(pages_with_regions / total_pages, 
                          text=f"Pages with regions: {pages_with_regions}/{total_pages}")
                
                # Visual representation of pages with regions
                if regions:
                    pages_status = []
                    for i in range(total_pages):
                        has_regions = any(r.page_idx == i + 1 for r in regions)
                        pages_status.append("✓" if has_regions else "·")
                    
                    # Display in rows of 10
                    for i in range(0, total_pages, 10):
                        st.text("".join(pages_status[i:i+10]))
                    
                    # Show total regions count with a metric
                    st.metric("Total Regions", len(regions))
    
    # Middle Column: Controls
    with control_col:
        if st.session_state.step == 1:
            st.header("Upload File")
            uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
            
            if uploaded_file:
                # Save file
                os.makedirs("temp_uploads", exist_ok=True)
                file_path = os.path.join("temp_uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.session_state.current_file = file_path
                
                # Preview settings
                st.write("### Preview Settings")
                alpha = st.slider("Page Transparency", 0.0, 1.0, 0.3, 0.1)
                
                # Generate preview
                with st.spinner("Generating preview..."):
                    # Cache PDF pages if not already cached
                    cached_pages = state_manager.get_pdf_pages(file_path)
                    if cached_pages is None:
                        pages = convert_from_path(file_path)
                        state_manager.cache_pdf_pages(file_path, pages)
                    else:
                        pages = cached_pages
                    
                    # Set first page as current image if not set
                    if not st.session_state.current_image:
                        st.session_state.current_image = pages[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Generate Stacked Preview", type="primary"):
                            preview = region_processor.create_stacked_preview(pages, alpha=alpha)
                            if preview is not None:
                                if preview.mode == 'RGBA':
                                    white_bg = Image.new('RGB', preview.size, 'white')
                                    white_bg.paste(preview, mask=preview.split()[3])
                                    preview = white_bg
                                st.session_state.current_image = preview
                                st.session_state.canvas_key += 1
                                st.rerun()
                    
                    with col2:
                        if st.button("Continue to Drawing", type="primary"):
                            st.session_state.step = 2
                            st.session_state.canvas_key += 1
                            st.rerun()
        
        elif st.session_state.step == 2:
            st.header("Draw Regions")
            if st.session_state.current_file and st.session_state.current_image:
                if st.session_state.redrawing:
                    # Get the specific page image from cache
                    pages = state_manager.get_pdf_pages(st.session_state.current_file)
                    if pages:
                        page_img = pages[st.session_state.redraw_page - 1].convert('RGB')
                        canvas_result, scale_factor = ui_manager.render_redraw_interface(page_img)
                        
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
                                
                                # Update the region
                                state_manager.update_region(
                                    st.session_state.current_file,
                                    st.session_state.redraw_page,
                                    st.session_state.redraw_region,
                                    original_coords
                                )
                                
                                # Process the updated region
                                region_processor.process_dirty_regions(st.session_state.current_file)
                                
                                # Exit redraw mode
                                state_manager.end_redraw()
                                st.rerun()
                else:
                    # Normal drawing mode
                    canvas_result, scale_factor = ui_manager.render_drawing_interface(st.session_state.current_image)
    
    # Right Column: Preview and Results
    with preview_col:
        if st.session_state.step == 1:
            if st.session_state.current_file and st.session_state.current_image:
                st.header("Preview")
                st.image(st.session_state.current_image, caption="Preview", use_column_width=True)
        
        elif st.session_state.step == 2:
                ui_manager.render_region_previews()


if __name__ == "__main__":
    main() 