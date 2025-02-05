from pathlib import Path
from typing import List
from PIL import Image
from .image_processor import ImageProcessor

def process_regions(image_paths: List[Path], 
                   output_dir: Path,
                   optimize_for_ocr: bool = True,
                   stitch_mode: str = None,
                   max_cols: int = 3) -> None:
    """
    Process a list of image regions with optional OCR optimization and stitching.
    
    Args:
        image_paths: List of paths to image regions
        output_dir: Directory to save processed images
        optimize_for_ocr: Whether to apply OCR optimization
        stitch_mode: One of [None, 'horizontal', 'vertical', 'grid_row', 'grid_col']
        max_cols: Maximum columns for grid mode
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process each image
    processed_images = []
    for i, img_path in enumerate(image_paths):
        # Load image
        img = Image.open(img_path)
        
        # Optimize if requested
        if optimize_for_ocr:
            img = processor.optimize_for_ocr(img)
        
        processed_images.append(img)
        
        # Save individual image if not stitching
        if not stitch_mode:
            output_path = output_dir / f"region_{i+1}.png"
            processor.save_high_quality(img, output_path)
    
    # Stitch images if requested
    if stitch_mode and processed_images:
        stitched = processor.stitch_images(
            processed_images,
            mode=stitch_mode,
            max_cols=max_cols
        )
        if stitched:
            output_path = output_dir / f"stitched_{stitch_mode}.png"
            processor.save_high_quality(stitched, output_path)

def main():
    # Example usage
    input_dir = Path("temp_regions")
    output_dir = Path("processed_regions")
    
    # Get all PNG files in input directory
    image_paths = list(input_dir.glob("*.png"))
    
    # Process with OCR optimization and horizontal stitching
    process_regions(
        image_paths,
        output_dir,
        optimize_for_ocr=True,
        stitch_mode='horizontal'
    )

if __name__ == "__main__":
    main() 
