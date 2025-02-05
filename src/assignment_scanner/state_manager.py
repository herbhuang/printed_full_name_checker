"""State management for the assignment scanner application."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from PIL import Image
import streamlit as st
from pathlib import Path

@dataclass
class RegionInfo:
    """Class to store information about a region in a document."""
    page_idx: int
    region_idx: int
    coordinates: Tuple[int, int, int, int]
    is_dirty: bool = False

class StateManager:
    """Class to manage application state and region information."""
    def __init__(self):
        # Basic state
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'step' not in st.session_state:
            st.session_state.step = 1
        if 'canvas_key' not in st.session_state:
            st.session_state.canvas_key = 0
            
        # Region management
        if 'regions' not in st.session_state:
            st.session_state.regions = {}  # file_path -> List[RegionInfo]
        if 'region_images' not in st.session_state:
            st.session_state.region_images = {}  # region_key -> Image
        if 'dirty_regions' not in st.session_state:
            st.session_state.dirty_regions = set()  # Set of region_keys that need updating
            
        # PDF pages cache
        if 'pdf_pages' not in st.session_state:
            st.session_state.pdf_pages = {}  # file_path -> List[Image]
            
        # Redrawing state
        if 'redrawing' not in st.session_state:
            st.session_state.redrawing = False
        if 'redraw_page' not in st.session_state:
            st.session_state.redraw_page = None
        if 'redraw_region' not in st.session_state:
            st.session_state.redraw_region = None
            
        # UI state
        if 'show_regions' not in st.session_state:
            st.session_state.show_regions = False
    
    def get_region_key(self, file_path: str, page_idx: int, region_idx: int) -> str:
        """Generate a unique key for a region."""
        return f"{Path(file_path).stem}_p{page_idx}_r{region_idx}"
    
    def save_canvas_regions(self, file_path: str, canvas_result: dict, scale_factor: float):
        """Save regions drawn on canvas and replicate them across all pages."""
        if not canvas_result or 'objects' not in canvas_result:
            return
        
        # Get all pages
        pages = self.get_pdf_pages(file_path)
        if not pages:
            return
            
        num_pages = len(pages)
        regions = []
        
        # Convert canvas objects to regions
        canvas_regions = []
        for i, obj in enumerate(canvas_result['objects'], 1):
            coords = (
                max(0, int(obj['left'] / scale_factor)),
                max(0, int(obj['top'] / scale_factor)),
                min(st.session_state.current_image.width, int((obj['left'] + obj['width']) / scale_factor)),
                min(st.session_state.current_image.height, int((obj['top'] + obj['height']) / scale_factor))
            )
            canvas_regions.append((i, coords))
        
        # Replicate regions across all pages
        for page_num in range(1, num_pages + 1):
            for region_idx, coords in canvas_regions:
                regions.append(RegionInfo(page_num, region_idx, coords, True))
        
        st.session_state.regions[file_path] = regions
        
        # Mark all regions as dirty
        for region in regions:
            region_key = self.get_region_key(file_path, region.page_idx, region.region_idx)
            st.session_state.dirty_regions.add(region_key)
    
    def update_region(self, file_path: str, page_idx: int, region_idx: int, new_coords: Tuple[int, int, int, int]):
        """Update a specific region's coordinates."""
        if file_path in st.session_state.regions:
            regions = st.session_state.regions[file_path]
            for region in regions:
                if region.page_idx == page_idx and region.region_idx == region_idx:
                    region.coordinates = new_coords
                    region.is_dirty = True
                    region_key = self.get_region_key(file_path, page_idx, region_idx)
                    st.session_state.dirty_regions.add(region_key)
                    break
    
    def get_region_image(self, file_path: str, page_idx: int, region_idx: int) -> Optional[Image.Image]:
        """Get the image for a specific region."""
        region_key = self.get_region_key(file_path, page_idx, region_idx)
        return st.session_state.region_images.get(region_key)
    
    def set_region_image(self, file_path: str, page_idx: int, region_idx: int, image: Image.Image):
        """Set the image for a specific region."""
        region_key = self.get_region_key(file_path, page_idx, region_idx)
        st.session_state.region_images[region_key] = image
        if region_key in st.session_state.dirty_regions:
            st.session_state.dirty_regions.remove(region_key)
    
    def get_dirty_regions(self, file_path: str) -> List[RegionInfo]:
        """Get all regions that need to be reprocessed."""
        if file_path not in st.session_state.regions:
            return []
        return [r for r in st.session_state.regions[file_path] if r.is_dirty]
    
    def clear_regions(self, file_path: str):
        """Clear all regions for a file."""
        if file_path in st.session_state.regions:
            del st.session_state.regions[file_path]
            # Clear associated region images
            for key in list(st.session_state.region_images.keys()):
                if key.startswith(Path(file_path).stem):
                    del st.session_state.region_images[key]
            st.session_state.dirty_regions.clear()
            st.session_state.show_regions = False
    
    def cache_pdf_pages(self, file_path: str, pages: List[Image.Image]):
        """Cache PDF pages for a file."""
        st.session_state.pdf_pages[file_path] = pages
    
    def get_pdf_pages(self, file_path: str) -> Optional[List[Image.Image]]:
        """Get cached PDF pages for a file."""
        return st.session_state.pdf_pages.get(file_path)
    
    def start_redraw(self, page_idx: int, region_idx: int):
        """Start redrawing a specific region."""
        st.session_state.redrawing = True
        st.session_state.redraw_page = page_idx
        st.session_state.redraw_region = region_idx
        st.session_state.canvas_key += 1
    
    def end_redraw(self):
        """End redrawing mode."""
        st.session_state.redrawing = False
        st.session_state.redraw_page = None
        st.session_state.redraw_region = None
    
    def get_all_regions(self, file_path: str) -> List[RegionInfo]:
        """Get all regions for a file."""
        return st.session_state.regions.get(file_path, [])

    def remove_region(self, file_path: str, page_idx: int, region_idx: int):
        """Remove a specific region from a file."""
        if file_path in st.session_state.regions:
            # Get current regions
            regions = st.session_state.regions[file_path]
            # Filter out the region to remove
            st.session_state.regions[file_path] = [
                r for r in regions 
                if not (r.page_idx == page_idx and r.region_idx == region_idx)
            ]
            # Remove the region image
            region_key = self.get_region_key(file_path, page_idx, region_idx)
            if region_key in st.session_state.region_images:
                del st.session_state.region_images[region_key]
            if region_key in st.session_state.dirty_regions:
                st.session_state.dirty_regions.remove(region_key)
