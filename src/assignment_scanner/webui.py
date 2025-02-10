"""
Web UI for configuring and running the Assignment Scanner.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib
import sys

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
import traceback

# Force reload all OCR-related modules
print("Reloading OCR modules...")  # Debug print
for module in list(sys.modules.keys()):
    if module.startswith('src.ocr.'):
        print(f"Reloading module: {module}")  # Debug print
        if module in sys.modules:
            importlib.reload(sys.modules[module])

from src.assignment_scanner.scanner import AssignmentScanner
from src.assignment_scanner.state_manager import StateManager
from src.assignment_scanner.region_processor import RegionProcessor
from src.assignment_scanner.ui_manager import UIManager
from src.assignment_scanner.image_processor import ImageProcessor
from src.ocr.processor import OCRProcessor
from src.ocr.base import OCRResult, ModelConfig

print("All modules imported, OCRProcessor:", OCRProcessor)  # Debug print


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


def render_ocr_interface():
    """Render the OCR interface with model selection and configuration."""
    st.header("OCR Configuration")
    
    print("Rendering OCR interface...")  # Debug print
    print(f"OCR processor: {st.session_state.ocr_processor}")  # Debug print
    print(f"dir(OCR processor): {dir(st.session_state.ocr_processor)}")  # Debug print
    
    # Get available models
    try:
        available_models = st.session_state.ocr_processor.get_available_models()
        print(f"Available models: {available_models}")  # Debug print
    except Exception as e:
        print(f"Error getting available models: {str(e)}")  # Debug print
        st.error(f"Error getting available models: {str(e)}")
        available_models = []
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        available_models,
        help="Choose the model to use for OCR"
    )
    
    # Task selection
    task_type = st.selectbox(
        "Select Task",
        ["Text Extraction", "Region Detection"],
        help="Choose the type of OCR task to perform"
    )
    
    # Model configuration
    with st.expander("Model Configuration"):
        # Get model config if available
        model_config = st.session_state.ocr_processor.get_model_config(model_name)
        
        if model_config and model_config.model_type == 'api':
            # API configuration
            api_key = st.text_input(
                "API Key",
                value=model_config.api_key or "",
                type="password",
                help="Enter your API key"
            )
            api_url = st.text_input(
                "API URL (Optional)",
                value=model_config.api_url or "",
                help="Enter custom API endpoint if needed"
            )
            
            # Update API configuration if changed
            if api_key != model_config.api_key or api_url != model_config.api_url:
                try:
                    st.session_state.ocr_processor.add_api_model(
                        name=model_name,
                        model_path=model_config.model_path,
                        api_key=api_key,
                        api_url=api_url or None
                    )
                    st.success("API configuration updated!")
                except Exception as e:
                    st.error(f"Failed to update API configuration: {str(e)}")
        
        # Common model parameters
        max_tokens = st.slider(
            "Maximum Tokens",
            min_value=128,
            max_value=2048,
            value=model_config.max_tokens if model_config else 1024,
            help="Maximum number of tokens in the output"
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=model_config.temperature if model_config else 0.0,
            help="Randomness in model output (0.0 = deterministic)"
        )
    
    # Prompt configuration
    with st.expander("Prompt Configuration", expanded=True):
        # Default prompts based on task
        default_prompt = (
            "Please analyze this image and identify text regions. For each region, "
            "extract only the first word and provide its location in the format "
            "'first_word: (x1, y1, x2, y2)'. Ignore all other words in each region."
        ) if task_type == "Region Detection" else (
            "Please extract only the first word from any text visible in this image. "
            "Ignore all other words and just return the first word you see."
        )
        
        # Store prompt in session state
        if 'custom_prompt' not in st.session_state:
            st.session_state.custom_prompt = default_prompt
        
        # Prompt templates
        prompt_templates = {
            "Default": default_prompt,
            "Extract All Text": "Please extract all text visible in this image.",
            "Extract Numbers": "Please extract only the numbers visible in this image.",
            "Custom": st.session_state.custom_prompt
        }
        
        # Template selection
        selected_template = st.selectbox(
            "Prompt Template",
            list(prompt_templates.keys()),
            index=0
        )
        
        # Custom prompt input
        prompt = st.text_area(
            "Custom Prompt",
            value=prompt_templates[selected_template],
            height=100,
            help="Customize the instruction given to the model"
        )
        
        # Update session state when prompt changes
        if prompt != st.session_state.custom_prompt:
            st.session_state.custom_prompt = prompt
    
    # Return configuration
    return {
        'model_name': model_name,
        'with_region': task_type == "Region Detection",
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature
    }


def render_ocr_results(results: List[OCRResult], images: List[Image.Image]):
    """Render OCR results in a clean format."""
    if not results:
        st.warning("No results to display")
        return
    
    for i, (result, img) in enumerate(zip(results, images)):
        if not result:
            st.warning(f"No result for region {i+1}")
            continue
        
        try:
            # Calculate display heights
            img_height = img.height
            text_height = int(img_height * 0.9)
            
            # Create columns for image and text
            img_col, text_col = st.columns(2)
            
            with img_col:
                # Show image without caption
                st.image(img)
            
            with text_col:
                # Show extracted text with hidden label
                st.text_area(
                    label=f"OCR Result {i+1}",  # Add descriptive label
                    value=result.text,
                    height=text_height,
                    key=f"ocr_result_{i}",
                    label_visibility="hidden"  # Hide the label but keep it for accessibility
                )
                
                # Show error if any
                if result.error:
                    st.error(f"Error: {result.error}")
            
            # Add small spacing between results
            st.markdown("<br>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error displaying result: {str(e)}")


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
        print("Initializing OCR processor in webui.py...")  # Debug print
        try:
            processor = OCRProcessor(skip_local_models=False)
            available_models = processor.get_available_models()
            print(f"OCRProcessor initialized, available models: {available_models}")  # Debug print
            if not available_models:
                raise RuntimeError("No models were successfully initialized")
            st.session_state.ocr_processor = processor
        except Exception as e:
            print(f"Error initializing OCR processor with local models: {str(e)}")  # Debug print
            print("Falling back to API-only mode")  # Debug print
            st.warning("Local OCR models could not be initialized. Falling back to API-only mode.")
            st.session_state.ocr_processor = OCRProcessor(skip_local_models=True)

    def handle_save_for_ocr():
        """Handle saving and preprocessing regions for OCR."""
        try:
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
            
            with st.spinner("Processing images..."):
                if stitch_mode == "individual":
                    # Process individual regions
                    processed_regions = []
                    for img in region_images:
                        if optimize_for_ocr:
                            processed_img = processor.optimize_for_ocr(
                                img,
                                preserve_dpi=True
                            )
                        else:
                            processed_img = img
                        # Ensure image is in RGB mode
                        if processed_img.mode != 'RGB':
                            processed_img = processed_img.convert('RGB')
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
                    
                    if stitched:
                        if optimize_for_ocr:
                            stitched = processor.optimize_for_ocr(
                                stitched,
                                preserve_dpi=True
                            )
                        # Ensure image is in RGB mode
                        if stitched.mode != 'RGB':
                            stitched = stitched.convert('RGB')
                        st.session_state.stitched_image = stitched
                        st.session_state.processed_regions = None
            
            #st.success("Images processed and ready for OCR!")
            st.session_state.step = 4
            
        except Exception as e:
            st.error(f"Error processing images: {str(e)}")
            st.error("Traceback:")
            st.code(traceback.format_exc())

    def handle_save_and_next():
        """Handle saving regions and moving to next step."""
        try:
            # Save any new regions if they exist
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                state_manager.save_canvas_regions(
                    st.session_state.current_file, 
                    canvas_result.json_data, 
                    scale_factor
                )
                region_processor.process_all_regions(st.session_state.current_file)
            
            # Update state to move to next step
            st.session_state.show_regions = True
            st.session_state.step = 3
            
        except Exception as e:
            st.error(f"Error saving regions: {str(e)}")

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
            4: "OCR Text",
            5: "Export Results"
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
                        if st.button("Generate Stacked Preview", type="primary", key="btn_generate_stacked"):
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
                        if st.button("Continue to Drawing", type="primary", key="btn_continue_drawing"):
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
                else:
                    # Normal drawing mode
                    canvas_result, scale_factor = ui_manager.render_drawing_interface(st.session_state.current_image)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        def clear_regions():
                            state_manager.clear_regions(st.session_state.current_file)
                            st.session_state.canvas_key += 1
                        st.button("Clear Regions", on_click=clear_regions, key="btn_clear_regions")
                    
                    with col2:
                        def preview_regions():
                            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                                state_manager.save_canvas_regions(
                                    st.session_state.current_file, 
                                    canvas_result.json_data, 
                                    scale_factor
                                )
                                region_processor.process_all_regions(st.session_state.current_file)
                                st.session_state.show_regions = True
                        st.button("Preview Regions", type="primary", on_click=preview_regions, key="btn_preview_regions")
                    
                    with col3:
                        st.button("Save and Next", type="primary", on_click=handle_save_and_next, key="btn_save_next")
        
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
                    save_button = st.button("Save Processed Regions", type="primary", key="btn_save_processed")
                with button_cols[1]:
                    st.button("Save for OCR", type="primary", on_click=handle_save_for_ocr, key="btn_save_for_ocr")

        elif st.session_state.step == 4:
            st.header("OCR Processing")
            
            # Validate processed regions or stitched image
            if not hasattr(st.session_state, 'processed_regions') and not hasattr(st.session_state, 'stitched_image'):
                st.warning("No processed images found. Please go back to step 3 and process regions first.")
                return
            
            # Get OCR configuration
            ocr_config = render_ocr_interface()
            
            # Create two columns for the buttons
            ocr_col1, ocr_col2 = st.columns(2)
            
            # Start OCR button in first column
            with ocr_col1:
                if st.button("Start OCR", type="primary", key="btn_start_ocr_step4"):
                    st.session_state.start_ocr = True
                    st.session_state.ocr_config = ocr_config
            
            # Save and Next button in second column
            with ocr_col2:
                if st.button("Save and Next →", type="primary", key="btn_save_next_step4"):
                    # Save OCR results if they exist
                    if hasattr(st.session_state, 'ocr_results') and st.session_state.ocr_results:
                        # Save OCR results to regions
                        for idx, result in enumerate(st.session_state.ocr_results):
                            if result and result.text:
                                # For individual mode, map results directly to regions
                                if st.session_state.stitch_mode == "individual":
                                    regions = state_manager.get_all_regions(st.session_state.current_file)
                                    if idx < len(regions):
                                        region = regions[idx]
                                        state_manager.save_ocr_result(
                                            st.session_state.current_file,
                                            region.page_idx,
                                            region.region_idx,
                                            result.text,
                                            result.confidence if hasattr(result, 'confidence') else None
                                        )
                    # Move to step 5
                    st.session_state.step = 5
                    st.rerun()
            
            # Process OCR if button was clicked
            if hasattr(st.session_state, 'start_ocr') and st.session_state.start_ocr:
                try:
                    # Get images to process
                    if st.session_state.stitch_mode == "individual":
                        images = st.session_state.processed_regions
                    else:
                        images = [st.session_state.stitched_image]
                    
                    # Initialize progress in control column
                    progress_container = st.empty()
                    progress_bar = progress_container.progress(0, text="Starting OCR...")
                    
                    # Process images with progress updates
                    results = []
                    total_images = len(images)
                    
                    # Process each image
                    for idx, img in enumerate(images):
                        # Update progress
                        progress = (idx) / total_images
                        progress_bar.progress(progress, text=f"Processing image {idx + 1} of {total_images}...")
                        
                        # Process single image
                        result = st.session_state.ocr_processor.models[ocr_config['model_name']].process_image(
                            image=img,
                            prompt=ocr_config['prompt'],
                            with_region=ocr_config['with_region']
                        )
                        results.append(result)
                    
                    # Complete progress
                    progress_bar.progress(1.0, text="OCR completed!")
                    
                    # Store results
                    st.session_state.ocr_results = results
                    st.session_state.start_ocr = False
                    
                    # Clear progress after completion
                    progress_container.empty()
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    st.session_state.start_ocr = False

    # Right Column: Preview
    with preview_col:
        st.header("Preview")
        
        if st.session_state.step == 1:
            # Show current image if available
            if st.session_state.current_image:
                st.image(st.session_state.current_image, use_column_width=True)
        
        elif st.session_state.step == 2:
            # Show drawing canvas or current image
            if st.session_state.current_image:
                if hasattr(st.session_state, 'redrawing') and st.session_state.redrawing:
                    # In redrawing mode, the canvas is shown in the middle column
                    pass
                else:

                    # Show region previews if available
                    if st.session_state.current_file:
                        # Ensure show_regions is set
                        if 'show_regions' not in st.session_state:
                            st.session_state.show_regions = True
                        ui_manager.render_region_previews()
        
        elif st.session_state.step == 3:
            # Show stitched preview based on mode
            if st.session_state.current_file:
                regions = state_manager.get_all_regions(st.session_state.current_file)
                if regions:
                    region_images = []
                    for region in regions:
                        region_img = state_manager.get_region_image(
                            st.session_state.current_file,
                            region.page_idx,
                            region.region_idx
                        )
                        if region_img:
                            region_images.append(region_img)
                    
                    if region_images:
                        if stitch_mode == "individual":
                            # For individual mode, show first region at original size
                            st.image(region_images[0])
                        else:
                            # Create stitched preview based on mode
                            if stitch_mode == "vertical":
                                preview = stitch_regions_vertically(images=region_images)
                            elif stitch_mode == "horizontal":
                                preview = stitch_regions_horizontally(images=region_images)
                            else:  # grid
                                preview = stitch_regions_grid(images=region_images, max_cols=max_cols)
                            
                            if preview:
                                # Calculate scale to fit column width while preserving aspect ratio
                                container_width = st.get_container_width() if 'get_container_width' in dir(st) else 800
                                if preview.width > container_width:
                                    scale = container_width / preview.width
                                    new_width = container_width
                                    new_height = int(preview.height * scale)
                                    preview = preview.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                st.image(preview, use_column_width=True)
        
        elif st.session_state.step == 4:
            # Show OCR results in preview
            if st.session_state.stitch_mode == "individual":
                images = st.session_state.processed_regions
            else:
                images = [st.session_state.stitched_image]

            # Create a container for results
            results_container = st.container()
            
            with results_container:
                # Display images and results
                for i, img in enumerate(images):
                    # Create columns for image and text
                    img_col, text_col = st.columns(2)
                    with img_col:
                        st.image(img)
                    with text_col:
                        # Show result if available, otherwise show waiting message
                        result_text = "Waiting for OCR..."
                        if hasattr(st.session_state, 'ocr_results') and len(st.session_state.ocr_results) > i:
                            result = st.session_state.ocr_results[i]
                            if result:
                                result_text = result.text
                                if result.error:
                                    st.error(f"Error: {result.error}")
                        elif hasattr(st.session_state, 'start_ocr') and st.session_state.start_ocr:
                            result_text = "Processing..."
                        
                        st.text_area(
                            label=f"OCR Result {i+1}",
                            value=result_text,
                            height=int(img.height * 0.9),
                            key=f"ocr_result_preview_{i}",
                            label_visibility="hidden"
                        )
                    st.markdown("<br>", unsafe_allow_html=True)

        elif st.session_state.step == 5:
            handle_step5_export_results(state_manager)


def handle_step4_ocr_text(state_manager: StateManager, ui_manager: UIManager):
    """Handle OCR text extraction step."""
    if not st.session_state.current_file:
        st.warning("Please upload a PDF file first.")
        st.session_state.step = 1
        st.rerun()
        return
    
    st.markdown("### OCR Text Extraction")
    st.markdown("Review and edit the extracted text from each region.")
    
    regions = state_manager.get_all_regions(st.session_state.current_file)
    if not regions:
        st.warning("No regions defined. Please draw regions first.")
        st.session_state.step = 2
        st.rerun()
        return
    
    # Group regions by page
    regions_by_page = {}
    for region in regions:
        if region.page_idx not in regions_by_page:
            regions_by_page[region.page_idx] = []
        regions_by_page[region.page_idx].append(region)
    
    # Display regions with OCR results and edit capability
    for page_num in sorted(regions_by_page.keys()):
        st.markdown(f"#### Page {page_num}")
        page_regions = regions_by_page[page_num]
        
        for region in sorted(page_regions, key=lambda r: r.region_idx):
            region_img = state_manager.get_region_image(
                st.session_state.current_file,
                region.page_idx,
                region.region_idx
            )
            
            if region_img:
                col1, col2 = st.columns([0.4, 0.6])
                with col1:
                    st.image(region_img, caption=f"Region {region.region_idx}")
                
                with col2:
                    current_text, confidence = state_manager.get_ocr_result(
                        st.session_state.current_file,
                        region.page_idx,
                        region.region_idx
                    )
                    
                    # Allow editing of OCR text
                    new_text = st.text_area(
                        f"Extracted Text (Region {region.region_idx})",
                        value=current_text or "",
                        key=f"ocr_text_p{page_num}_r{region.region_idx}"
                    )
                    
                    # Save edited text
                    if new_text != current_text:
                        state_manager.save_ocr_result(
                            st.session_state.current_file,
                            region.page_idx,
                            region.region_idx,
                            new_text
                        )
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Preview"):
            st.session_state.step = 3
            st.rerun()
    
    with col3:
        if st.button("Save and Export →", type="primary"):
            st.session_state.step = 5
            st.rerun()


def handle_step5_export_results(state_manager: StateManager):
    """Handle results export step."""
    if not st.session_state.current_file:
        st.warning("Please upload a PDF file first.")
        st.session_state.step = 1
        st.rerun()
        return
    
    st.markdown("### Export Results")
    
    # Check if we have OCR results from step 4
    has_step4_results = hasattr(st.session_state, 'ocr_results') and st.session_state.ocr_results
    stitch_mode = getattr(st.session_state, 'stitch_mode', 'individual')
    
    if stitch_mode != 'individual' and has_step4_results:
        # For stitched modes, show the combined OCR result
        st.markdown("#### Combined OCR Result")
        combined_text = st.session_state.ocr_results[0].text if st.session_state.ocr_results else ""
        
        # Allow editing of the combined text
        edited_text = st.text_area(
            "Combined Text Result",
            value=combined_text,
            height=300,
            key="combined_ocr_text"
        )
        
        # Create a single-row dataframe for the combined result
        df = pd.DataFrame([{
            'mode': stitch_mode.capitalize(),
            'text': edited_text,
            'confidence': st.session_state.ocr_results[0].confidence if st.session_state.ocr_results else None
        }])
        
        # Add copy buttons in columns
        copy_col1, copy_col2 = st.columns([0.7, 0.3])
        with copy_col1:
            st.text_area("Copy this text:", value=edited_text, height=100, key="copy_area_stitched")
        with copy_col2:
            if st.button("📋 Copy Text", key="btn_copy_stitched"):
                st.code(edited_text)
                st.toast("Text is ready to copy! Use Ctrl+C or ⌘+C to copy", icon="📋")
        
    else:
        # For individual mode or when using saved region results
        st.markdown("#### Individual Region Results")
        results = state_manager.export_ocr_results(st.session_state.current_file)
        
        if not results['regions']:
            st.warning("No OCR results available. Please process regions first.")
            st.session_state.step = 4
            st.rerun()
            return
        
        # Create DataFrame for individual results
        df = pd.DataFrame(results['regions'])
        
        # Extract only texts and join with newlines
        all_texts = "\n".join(df['text'].dropna().astype(str))
        
        # Add single Copy All button
        if st.button("📋 Copy All Texts", type="primary", key="btn_copy_individual"):
            st.code(all_texts)
            st.toast("All texts copied! Use Ctrl+C or ⌘+C to copy", icon="📋")
    
    # Display results table based on mode
    if stitch_mode != 'individual' and has_step4_results:
        st.dataframe(df[['mode', 'text', 'confidence']])
    else:
        st.markdown("#### Results Preview Table")
        # Create an interactive table with copy functionality
        st.dataframe(
            df[['page', 'region', 'text', 'confidence']],
            column_config={
                "page": st.column_config.NumberColumn("Page", help="Page number"),
                "region": st.column_config.NumberColumn("Region", help="Region number"),
                "text": st.column_config.TextColumn(
                    "Text",
                    help="Click to copy text",
                    width="large",
                ),
                "confidence": st.column_config.NumberColumn(
                    "Confidence",
                    help="OCR confidence score",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Export options
    st.markdown("### Export Options")
    export_format = st.selectbox(
        "Select export format:",
        ["CSV", "JSON", "Excel"],
        key="export_format"
    )
    
    # Add metadata to results
    export_data = {
        'file_name': Path(st.session_state.current_file).name,
        'ocr_mode': stitch_mode,
        'timestamp': pd.Timestamp.now().isoformat(),
        'results': df.to_dict('records')
    }
    
    if st.button("Export Results", type="primary"):
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                file_name=f"{Path(st.session_state.current_file).stem}_results.csv",
                mime="text/csv"
            )
        elif export_format == "JSON":
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                "Download JSON",
                json_str,
                file_name=f"{Path(st.session_state.current_file).stem}_results.json",
                mime="application/json"
            )
        else:  # Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='OCR Results')
                
                # Add metadata sheet
                metadata_df = pd.DataFrame([{
                    'Property': key,
                    'Value': value
                } for key, value in {
                    'File Name': export_data['file_name'],
                    'OCR Mode': export_data['ocr_mode'],
                    'Timestamp': export_data['timestamp']
                }.items()])
                metadata_df.to_excel(writer, index=False, sheet_name='Metadata')
                
            st.download_button(
                "Download Excel",
                output.getvalue(),
                file_name=f"{Path(st.session_state.current_file).stem}_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to OCR"):
            st.session_state.step = 4
            st.rerun()
    
    with col2:
        if st.button("Start New Scan", type="primary"):
            # Reset state
            state_manager.clear_regions(st.session_state.current_file)
            st.session_state.current_file = None
            st.session_state.step = 1
            st.rerun()


if __name__ == "__main__":
    main() 