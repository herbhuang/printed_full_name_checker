"""Module for processing regions in PDF documents."""
"""Module for processing regions in PDF documents."""

from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import streamlit as st
from .state_manager import StateManager, RegionInfo

class RegionProcessor:
    """Class to handle processing of regions in PDF pages."""
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
    
    def process_dirty_regions(self, file_path: str):
        """Process only regions that are marked as dirty."""
        dirty_regions = self.state_manager.get_dirty_regions(file_path)
        if not dirty_regions:
            return
        
        pages = self.state_manager.get_pdf_pages(file_path)
        if not pages:
            return
        
        for region in dirty_regions:
            page_img = pages[region.page_idx - 1].convert('RGB')
            region_img = page_img.crop(region.coordinates)
            self.state_manager.set_region_image(
                file_path, 
                region.page_idx, 
                region.region_idx, 
                region_img
            )
    
    def process_all_regions(self, file_path: str):
        """Process all regions for initial preview."""
        regions = self.state_manager.get_all_regions(file_path)
        if not regions:
            return
        
        pages = self.state_manager.get_pdf_pages(file_path)
        if not pages:
            return
        
        # Get the stacked preview dimensions
        preview_size = (
            st.session_state.current_image.width,
            st.session_state.current_image.height
        )
        
        # Process each region
        for region in regions:
            # Get the page image
            page_img = pages[region.page_idx - 1].convert('RGB')
            
            # Scale coordinates from preview to actual page size
            page_size = (page_img.width, page_img.height)
            scaled_coords = self.scale_coordinates(
                region.coordinates,
                from_size=preview_size,
                to_size=page_size
            )
            
            # Crop and save region
            region_img = page_img.crop(scaled_coords)
            self.state_manager.set_region_image(
                file_path,
                region.page_idx,
                region.region_idx,
                region_img
            )
    
    def scale_coordinates(self, coords: Tuple[int, int, int, int], 
                         from_size: Tuple[int, int], 
                         to_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Scale coordinates between different image sizes."""
        scale_x = to_size[0] / from_size[0]
        scale_y = to_size[1] / from_size[1]
        return (
            int(coords[0] * scale_x),
            int(coords[1] * scale_y),
            int(coords[2] * scale_x),
            int(coords[3] * scale_y)
        )
    
    def create_stacked_preview(self, pages: List[Image.Image], alpha: float = 0.3) -> Optional[Image.Image]:
        """Create a preview with all pages stacked with transparency."""
        if not pages:
            return None
        
        base_image = pages[0].copy().convert('RGBA')
        
        for page in pages[1:]:
            overlay = page.convert('RGBA')
            if overlay.size != base_image.size:
                overlay = overlay.resize(base_image.size, Image.Resampling.LANCZOS)
            
            # Create transparent version
            transparent = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
            transparent.paste(overlay, (0, 0))
            
            # Adjust alpha
            data = transparent.getdata()
            newData = [(r, g, b, int(a * alpha)) for r, g, b, a in data]
            transparent.putdata(newData)
            
            # Composite images
            base_image = Image.alpha_composite(base_image, transparent)
        
        return base_image
