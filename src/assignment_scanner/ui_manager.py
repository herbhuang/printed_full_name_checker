from typing import List, Tuple, Optional
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from .state_manager import StateManager
from .region_processor import RegionProcessor

class UIManager:
    def __init__(self, state_manager: StateManager, region_processor: RegionProcessor):
        self.state_manager = state_manager
        self.region_processor = region_processor
    
    def add_border(self, img: Image.Image, border_width: int = 2, border_color: str = 'red') -> Image.Image:
        """Add a colored border to an image."""
        img_copy = img.copy()
        bordered_img = ImageDraw.Draw(img_copy)
        bordered_img.rectangle([(0, 0), (img.width-1, img.height-1)], 
                             outline=border_color, width=border_width)
        return img_copy
    
    def render_region_preview(self, region_img: Image.Image, page_num: int, region_num: int):
        """Render a region preview with controls in a horizontal layout."""
        # Start single line container
        st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.1rem 0;">', unsafe_allow_html=True)
        
        # Create columns with equal widths for controls
        edit_col, check_col, page_col, region_col, img_col = st.columns([0.05, 0.05, 0.05, 0.05, 0.8])
        
        # Resize to 60% of original size
        preview_scale = 0.6
        new_width = int(region_img.width * preview_scale)
        new_height = int(region_img.height * preview_scale)
        region_img_resized = region_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        with edit_col:
            def start_redraw():
                self.state_manager.start_redraw(page_num, region_num)
                st.session_state.needs_rerun = True
            st.button("✏️", key=f"edit_p{page_num}_r{region_num}", 
                     on_click=start_redraw)
        
        with check_col:
            st.checkbox("", value=True, key=f"select_p{page_num}_r{region_num}")
        
        with page_col:
            st.write(f"P{page_num}")
        
        with region_col:
            st.write(f"R{region_num}")
        
        with img_col:
            img_with_border = self.add_border(region_img_resized)
            st.image(img_with_border)
        
        # End single line container
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_drawing_interface(self, display_image: Image.Image):
        """Render the main drawing interface including canvas and controls."""
        if st.session_state.redrawing:
            return self.render_redraw_interface(display_image)
            
        DISPLAY_HEIGHT = 600
        scale_factor = DISPLAY_HEIGHT / display_image.height
        display_width = int(display_image.width * scale_factor)
        
        display_image = display_image.resize(
            (display_width, DISPLAY_HEIGHT), 
            Image.Resampling.LANCZOS
        )
        
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            def clear_regions():
                self.state_manager.clear_regions(st.session_state.current_file)
                st.session_state.canvas_key += 1
                st.session_state.needs_rerun = True
            st.button("Clear Regions", on_click=clear_regions)
        
        with col2:
            def preview_regions():
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    self.state_manager.save_canvas_regions(
                        st.session_state.current_file, 
                        canvas_result.json_data, 
                        scale_factor
                    )
                    self.region_processor.process_all_regions(st.session_state.current_file)
                    st.session_state.show_regions = True
                    st.session_state.needs_rerun = True
            st.button("Preview Regions", type="primary", on_click=preview_regions)
        
        with col3:
            def save_and_next():
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    self.state_manager.save_canvas_regions(
                        st.session_state.current_file, 
                        canvas_result.json_data, 
                        scale_factor
                    )
                    st.session_state.step = 3
                    st.session_state.needs_rerun = True
            st.button("Save and Next", type="primary", on_click=save_and_next)
        
        return canvas_result, scale_factor
    
    def render_redraw_interface(self, page_img: Image.Image):
        """Render the redraw interface."""
        DISPLAY_HEIGHT = 600
        scale_factor = DISPLAY_HEIGHT / page_img.height
        display_width = int(page_img.width * scale_factor)
        
        display_image = page_img.resize(
            (display_width, DISPLAY_HEIGHT),
            Image.Resampling.LANCZOS
        )
        
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
        
        button_col, header_col = st.columns([0.2, 0.8])
        with button_col:
            def cancel_redraw():
                self.state_manager.end_redraw()
                st.session_state.needs_rerun = True
            st.button("Cancel Redraw", on_click=cancel_redraw, use_container_width=True)
        
        with header_col:
            st.markdown(f"**Redrawing Region {st.session_state.redraw_region} on Page {st.session_state.redraw_page}**")
        
        # Handle redraw result
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
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
                self.state_manager.update_region(
                    st.session_state.current_file,
                    st.session_state.redraw_page,
                    st.session_state.redraw_region,
                    original_coords
                )
                
                # Process all regions to ensure everything is up to date
                self.region_processor.process_all_regions(st.session_state.current_file)
                
                # Exit redraw mode
                self.state_manager.end_redraw()
                st.session_state.needs_rerun = True
                st.rerun()
        
        return canvas_result, scale_factor
    
    def render_region_previews(self):
        """Render previews of all regions with controls."""
        if not st.session_state.show_regions:
            return
            
        if 'needs_rerun' not in st.session_state:
            st.session_state.needs_rerun = False
            
        if st.session_state.needs_rerun:
            st.session_state.needs_rerun = False
            st.rerun()
        
        st.markdown("### Region Previews")
        
        regions = self.state_manager.get_all_regions(st.session_state.current_file)
        if not regions:
            st.warning("No regions drawn yet.")
            return
            
        regions_by_page = {}
        for region in regions:
            if region.page_idx not in regions_by_page:
                regions_by_page[region.page_idx] = []
            regions_by_page[region.page_idx].append(region)
            
        first_page = True
        for page_num in sorted(regions_by_page.keys()):
            if not first_page:
                st.markdown('<hr style="margin: 0.1rem 0; border-width: 0.5px;">', unsafe_allow_html=True)
            else:
                first_page = False
            
            page_regions = regions_by_page[page_num]
            for region in page_regions:
                region_img = self.state_manager.get_region_image(
                    st.session_state.current_file,
                    page_num,
                    region.region_idx
                )
                if region_img:
                    self.render_region_preview(region_img, page_num, region.region_idx)
