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
from src.assignment_scanner.image_processor import ImageProcessor
from src.assignment_scanner.ocr_processor import OCRProcessor


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
    
    # Initialize session state variables
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'ocr_settings' not in st.session_state:
        st.session_state.ocr_settings = {}
    if 'processed_regions' not in st.session_state:
        st.session_state.processed_regions = None
    if 'stitch_mode' not in st.session_state:
        st.session_state.stitch_mode = "individual"

    # Initialize OCR processor with error handling
    if 'ocr_processor' not in st.session_state:
        try:
            st.session_state.ocr_processor = OCRProcessor()
        except Exception as e:
            st.error(f"Error initializing OCR processor: {str(e)}")
            st.error("Some OCR methods may not be available. Please check your GPU and CUDA installation.")
            # Create a basic processor without VLLM models
            st.session_state.ocr_processor = OCRProcessor(skip_vllm=True)

    def handle_start_ocr():
        # Save the current optimization settings to session state for OCR step
        st.session_state.ocr_settings = {
            'use_yuv': use_yuv,
            'use_contrast_enhancement': use_contrast,
            'use_bilateral_filter': use_bilateral,
            'use_clahe': use_clahe,
            'use_adaptive_threshold': use_threshold,
            'use_morphology': use_morphology,
            'use_denoising': use_denoising,
            'contrast_alpha': contrast_alpha,
            'contrast_beta': contrast_beta,
            'bilateral_d': bilateral_d,
            'bilateral_sigma_color': bilateral_sigma,
            'bilateral_sigma_space': bilateral_sigma,
            'clahe_clip_limit': clahe_clip,
            'clahe_grid_size': clahe_grid,
            'threshold_block_size': threshold_block,
            'threshold_c': threshold_c,
            'morph_kernel_size': morph_size,
            'morph_iterations': morph_iter,
            'denoise_strength': denoise_strength,
            'denoise_template_window': denoise_template,
            'denoise_search_window': denoise_search
        }
        
        # Save stitch mode and create processor
        st.session_state.stitch_mode = stitch_mode
        processor = ImageProcessor()
        
        if stitch_mode == "individual":
            # Process individual regions
            processed_regions = []
            for img in region_images:
                if optimize_for_ocr:
                    processed_img = processor.optimize_for_ocr(
                        img,
                        preserve_dpi=True,
                        **st.session_state.ocr_settings
                    )
                else:
                    processed_img = img
                processed_regions.append(processed_img)
            st.session_state.processed_regions = processed_regions
            st.session_state.stitched_image = None
        else:
            # Create and process stitched image
            if stitch_mode == "vertical":
                stitched = stitch_regions_vertically(images=region_images)
            elif stitch_mode == "horizontal":
                stitched = stitch_regions_horizontally(images=region_images)
            else:  # grid
                stitched = stitch_regions_grid(images=region_images, max_cols=max_cols)
            
            if stitched and optimize_for_ocr:
                stitched = processor.optimize_for_ocr(
                    stitched,
                    preserve_dpi=True,
                    **st.session_state.ocr_settings
                )
            st.session_state.stitched_image = stitched
            st.session_state.processed_regions = None
            
        st.session_state.step = 4

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
            3: "Process Regions",
            4: "OCR Text"
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
        
        elif st.session_state.step == 3:
            if st.session_state.current_file:
                st.header("Process Regions")
                
                # Get all regions and validate
                regions = state_manager.get_all_regions(st.session_state.current_file)
                if not regions:
                    st.warning("No regions found. Please go back to step 2 and draw regions.")
                    return
                
                # Get all region images
                region_images = []
                for region in regions:
                    region_img = state_manager.get_region_image(
                        st.session_state.current_file,
                        region.page_idx,
                        region.region_idx
                    )
                    if region_img:
                        region_images.append(region_img)
                
                if not region_images:
                    st.warning("No region images found. Please go back to step 2 and ensure regions are properly drawn.")
                    return

                st.markdown("### Preprocessing Controls")
                
                # Step enablers
                use_yuv = st.checkbox("Use YUV Conversion", value=True)
                use_contrast = st.checkbox("Use Contrast Enhancement", value=True)
                use_bilateral = st.checkbox("Use Bilateral Filter", value=True)
                use_clahe = st.checkbox("Use CLAHE", value=True)
                use_threshold = st.checkbox("Use Adaptive Threshold", value=True)
                use_morphology = st.checkbox("Use Morphology", value=True)
                use_denoising = st.checkbox("Use Denoising", value=True)
                
                # Advanced settings expander
                with st.expander("Advanced Settings"):
                    # Contrast settings
                    st.markdown("##### Contrast Enhancement")
                    contrast_alpha = st.slider("Contrast Factor", 1.0, 3.0, 1.2, 0.1)
                    contrast_beta = st.slider("Brightness Adjustment", -50, 50, 0, 1)
                    
                    # Bilateral filter settings
                    st.markdown("##### Bilateral Filter")
                    bilateral_d = st.slider("Diameter", 1, 15, 1, 2)
                    bilateral_sigma = st.slider("Sigma", 1, 150, 45, 1)
                    
                    # CLAHE settings
                    st.markdown("##### CLAHE")
                    clahe_clip = st.slider("Clip Limit", 1.0, 20.0, 9.0, 0.5)
                    clahe_grid = st.slider("Grid Size", 2, 100, 80, 2)
                    
                    # Threshold settings
                    st.markdown("##### Adaptive Threshold")
                    threshold_block = st.slider("Block Size", 3, 99, 21, 2)
                    threshold_c = st.slider("C Constant", -50, 50, 10, 1)
                    
                    # Morphology settings
                    st.markdown("##### Morphology")
                    morph_size = st.slider("Kernel Size", 1, 5, 2, 1)
                    morph_iter = st.slider("Iterations", 1, 5, 1, 1)
                    
                    # Denoising settings
                    st.markdown("##### Denoising")
                    denoise_strength = st.slider("Strength", 1, 20, 1, 1)
                    denoise_template = st.slider("Template Window", 3, 15, 5, 2)
                    denoise_search = st.slider("Search Window", 5, 30, 15, 2)
                
                # Stitch mode selection
                stitch_mode = st.selectbox(
                    "Select Stitch Mode",
                    ["individual", "vertical", "horizontal", "grid"],
                    index=0,
                    key="stitch_mode_select",
                    on_change=lambda: setattr(st.session_state, 'stitch_mode', st.session_state.stitch_mode_select)
                )
                
                # Grid options if grid mode selected
                if stitch_mode == "grid":
                    max_cols = st.slider("Maximum Columns", 1, 5, 2)
                else:
                    max_cols = 2
                
                # OCR optimization option
                optimize_for_ocr = st.checkbox("Apply Optimization to All Regions", value=True)
                
                # Save and OCR buttons in a row
                button_cols = st.columns(2)
                with button_cols[0]:
                    save_button = st.button("Save Processed Regions", type="primary")
                with button_cols[1]:
                    st.button("Start OCR", type="primary", on_click=handle_start_ocr)

        elif st.session_state.step == 4:
            if not hasattr(st.session_state, 'processed_regions') and not hasattr(st.session_state, 'stitched_image'):
                st.warning("No processed images found. Please go back to step 3 and process regions first.")
                return
            
            # Validate processed regions or stitched image
            if st.session_state.stitch_mode == "individual":
                if not st.session_state.processed_regions:
                    st.warning("No processed regions found. Please go back to step 3 and process your regions.")
                    return
            else:
                if not st.session_state.stitched_image:
                    st.warning("No stitched image found. Please go back to step 3 and process your regions.")
                    return
            
            st.header("OCR Text")
            
            # Show current mode
            st.info(f"Processing Mode: {'Individual Regions' if st.session_state.stitch_mode == 'individual' else 'Stitched Output'}")
            
            # OCR Method Selection
            ocr_method = st.selectbox(
                "Select OCR Method",
                ["Tesseract (Default)", "Florence-2", "Florence-2 with Region", "Qwen2.5-VL", "Qwen2.5-VL with Region"],
                index=0
            )
            
            # Store OCR settings in session state
            if 'ocr_settings' not in st.session_state:
                st.session_state.ocr_settings = {
                    'method': ocr_method,
                    'kwargs': {}
                }
            
            # Language and Settings based on method
            if ocr_method == "Tesseract (Default)":
                languages = ["eng", "eng+fra", "eng+deu", "eng+spa"]
                lang = st.selectbox("Select Language", languages, index=0)
                
                # Tesseract-specific settings
                with st.expander("Advanced Settings"):
                    psm_mode = st.slider("Page Segmentation Mode (PSM)", 0, 13, 6,
                                       help="PSM modes: 6=Uniform block of text, 3=Auto, etc.")
                    oem_mode = st.slider("OCR Engine Mode (OEM)", 0, 3, 3,
                                       help="OEM modes: 3=Default, 1=Neural nets LSTM only")
                    whitelist = st.text_input("Character Whitelist", 
                                            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
                
                st.session_state.ocr_settings['kwargs'] = {
                    'lang': lang,
                    'psm_mode': psm_mode,
                    'oem_mode': oem_mode,
                    'whitelist': whitelist
                }
            
            elif ocr_method.startswith("Florence-2"):
                with st.expander("Model Information"):
                    st.markdown("""
                    Florence-2 is a powerful vision-language model optimized for OCR tasks.
                    - Supports both handwritten and printed text
                    - High accuracy text recognition
                    - Region detection capability
                    - English language support
                    """)
                    
                with st.expander("Advanced Settings"):
                    max_tokens = st.slider("Maximum Output Tokens", 128, 2048, 1024,
                                         help="Maximum number of tokens in the output")
                
                st.session_state.ocr_settings['kwargs'] = {
                    'max_new_tokens': max_tokens,
                    'with_region': "with Region" in ocr_method
                }
            
            else:  # Qwen methods
                with st.expander("Model Information"):
                    st.markdown("""
                    Qwen2.5-VL is a powerful vision-language model optimized for OCR tasks.
                    - Supports both handwritten and printed text
                    - Multilingual support
                    - High accuracy text recognition
                    - Region detection capability with precise coordinates
                    """)
                    
                with st.expander("Advanced Settings"):
                    col1, col2 = st.columns(2)
                    with col1:
                        max_tokens = st.slider("Maximum Output Tokens", 128, 2048, 128,
                                             help="Maximum number of tokens in the output")
                        temperature = st.slider("Temperature", 0.0, 1.0, 0.0,
                                              help="Higher values make the output more random")
                    
                    with col2:
                        use_flash_attention = st.checkbox("Use Flash Attention", value=True,
                                                        help="Enable for better performance on supported GPUs")
                        
                    # Visual token settings
                    st.markdown("##### Visual Token Settings")
                    st.caption("Adjust these settings to balance between performance and memory usage")
                    token_col1, token_col2 = st.columns(2)
                    with token_col1:
                        min_token_multiplier = st.slider("Min Token Multiplier", 1, 10, 4,
                                                       help="Lower value = faster processing")
                        min_pixels = min_token_multiplier * 64 * 64
                    with token_col2:
                        max_token_multiplier = st.slider("Max Token Multiplier", 10, 40, 20,
                                                       help="Lower value = less memory usage")
                        max_pixels = max_token_multiplier * 64 * 64
                    
                    if max_token_multiplier <= min_token_multiplier:
                        st.warning("Max token multiplier should be greater than min token multiplier")
                
                st.session_state.ocr_settings['kwargs'] = {
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    'with_region': "with Region" in ocr_method,
                    'use_flash_attention': use_flash_attention,
                    'min_pixels': min_pixels,
                    'max_pixels': max_pixels
                }
            
            st.session_state.ocr_settings['method'] = ocr_method
            
            # Start OCR button
            if st.button("Start OCR", type="primary"):
                st.session_state.start_ocr = True

    # Right Column: Preview and Results
    with preview_col:
        if st.session_state.step == 1:
            if st.session_state.current_file and st.session_state.current_image:
                st.header("Preview")
                st.image(st.session_state.current_image, caption="Preview", use_column_width=True)
        
        elif st.session_state.step == 2:
            st.header("Region Previews")
            ui_manager.render_region_previews()
            
        elif st.session_state.step == 3:
            st.header("Final Output")
            
            # Create processor instance
            processor = ImageProcessor()
            
            # Show preview of all regions
            if st.session_state.stitch_mode == "individual":
                for i, img in enumerate(region_images):
                    if optimize_for_ocr:
                        img = processor.optimize_for_ocr(
                            img,
                            preserve_dpi=True,
                            **st.session_state.ocr_settings
                        )
                    st.image(img)
            else:
                # Preview stitched result
                if st.session_state.stitch_mode == "vertical":
                    preview = stitch_regions_vertically(images=region_images)
                elif st.session_state.stitch_mode == "horizontal":
                    preview = stitch_regions_horizontally(images=region_images)
                else:  # grid
                    preview = stitch_regions_grid(images=region_images, max_cols=max_cols)
                
                if preview:
                    if optimize_for_ocr:
                        preview = processor.optimize_for_ocr(
                            preview,
                            preserve_dpi=True,
                            **st.session_state.ocr_settings
                        )
                    st.image(preview, caption="Stitched Output", use_column_width=True)
        
        elif st.session_state.step == 4:
            st.header("OCR Results")
            
            # Process OCR if button was clicked
            if hasattr(st.session_state, 'start_ocr') and st.session_state.start_ocr:
                # Initialize progress bar
                progress_bar = st.progress(0, text="Starting OCR...")
                
                try:
                    # Map OCR method to processor method
                    method_map = {
                        "Tesseract (Default)": "tesseract",
                        "Florence-2": "florence",
                        "Florence-2 with Region": "florence_with_region",
                        "Qwen2.5-VL": "qwen",
                        "Qwen2.5-VL with Region": "qwen_with_region"
                    }
                    
                    results = []
                    method = method_map[st.session_state.ocr_settings['method']]
                    kwargs = st.session_state.ocr_settings['kwargs']
                    
                    # Process images based on mode
                    if st.session_state.stitch_mode == "individual":
                        total_images = len(st.session_state.processed_regions)
                        for i, img in enumerate(st.session_state.processed_regions):
                            if img is None:
                                st.warning(f"Skipping region {i+1} - image is None")
                                continue
                            
                            # Update progress
                            progress = (i + 1) / total_images
                            progress_bar.progress(progress, text=f"Processing region {i+1} of {total_images}...")
                            
                            try:
                                # Ensure image is in RGB mode
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Process single image
                                result = st.session_state.ocr_processor.process_batch(
                                    [img],
                                    method=method,
                                    **kwargs
                                )[0]
                                results.append(result)
                            except Exception as e:
                                st.warning(f"Error processing region {i+1}: {str(e)}")
                                results.append(None)
                                continue
                    else:
                        # Update progress for single stitched image
                        progress_bar.progress(0.5, text="Processing stitched image...")
                        
                        # Ensure stitched image is in RGB mode
                        img = st.session_state.stitched_image
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Process stitched image as a single region
                        result = st.session_state.ocr_processor.process_batch(
                            [img],
                            method=method,
                            **kwargs
                        )[0]
                        results = [result]
                    
                    # Complete progress
                    progress_bar.progress(1.0, text="OCR completed!")
                    
                    # Store results in session state
                    st.session_state.ocr_results = results
                    st.session_state.start_ocr = False
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    st.session_state.start_ocr = False
                    return
            
            # Display results if available
            if hasattr(st.session_state, 'ocr_results') and st.session_state.ocr_results:
                results = st.session_state.ocr_results
                
                for i, result in enumerate(results):
                    if not result:
                        st.warning(f"No result for region {i+1}")
                        continue
                    
                    title = f"Region {i+1}" if st.session_state.stitch_mode == "individual" else "Stitched Output"
                    st.markdown(f"#### {title}")
                    
                    try:
                        if st.session_state.stitch_mode == "individual":
                            img = st.session_state.processed_regions[i] if i < len(st.session_state.processed_regions) else None
                        else:
                            img = st.session_state.stitched_image
                        
                        if img is None:
                            st.warning(f"Image not found for {title}")
                            continue
                        
                        img_height = img.height
                        text_height = int(img_height * 0.9)
                        
                        # Create columns for image and text
                        img_col, text_col = st.columns(2)
                        
                        with img_col:
                            st.image(img)
                            if 'regions' in result:  # Qwen format
                                st.markdown("Detected Regions:")
                                for j, region in enumerate(result['regions']):
                                    st.write(f"- Region {j+1}: {region['text']}")
                                    st.write(f"  Box: {region['box']}")
                            elif 'boxes' in result:  # Florence format
                                st.markdown("Detected Regions:")
                                boxes = result['boxes']
                                labels = result.get('labels', [])
                                for j, (box, label) in enumerate(zip(boxes, labels)):
                                    st.write(f"- Region {j+1}: {label}")
                                    st.write(f"  Box: {box}")
                        
                        with text_col:
                            st.text_area("Extracted Text", 
                                       result.get('text', ''),
                                       height=text_height,
                                       key=f"text_area_{i}")
                            
                            # Show raw output for debugging
                            if 'raw_output' in result:
                                with st.expander("Raw Model Output"):
                                    st.code(str(result['raw_output']))
                        
                    except Exception as e:
                        st.error(f"Error displaying result for {title}: {str(e)}")
                        continue
                    
                    st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main() 