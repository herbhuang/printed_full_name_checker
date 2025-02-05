"""
Hugging Face API model implementation.
"""

import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional

from src.ocr.base import BaseAPILLM, ModelConfig, OCRResult

class HuggingFaceAPIModel(BaseAPILLM):
    """Hugging Face API model implementation."""
    
    def _call_api(
        self,
        image: Image.Image,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make API call to Hugging Face Inference API."""
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare payload based on model type
            if "llava" in self.config.model_path.lower():
                payload = {
                    "inputs": {
                        "image": img_str,
                        "prompt": prompt
                    }
                }
            else:
                # Default format for most vision-language models
                payload = {
                    "inputs": [
                        {
                            "image": img_str,
                            "text": prompt
                        }
                    ]
                }
            
            # Make API call
            response = requests.post(
                self.config.api_url or f"https://api-inference.huggingface.co/models/{self.config.model_path}",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        with_region: bool = False,
        **kwargs
    ) -> OCRResult:
        """Process image with Hugging Face API."""
        try:
            # Preprocess image
            image = self._preprocess_image(image)
            
            # Use default prompts if none provided
            if not prompt:
                prompt = (
                    "Please analyze this image and identify text regions. For each region, "
                    "extract only the first word and provide its location in the format "
                    "'first_word: (x1, y1, x2, y2)'. Ignore all other words in each region."
                ) if with_region else (
                    "Please extract only the first word from any text visible in this image. "
                    "Ignore all other words and just return the first word you see."
                )
            
            # Call API
            response = self._call_api(image, prompt)
            
            # Parse response based on model type
            if isinstance(response, list):
                # Most models return a list with a single item
                output_text = response[0].get('generated_text', '')
            elif isinstance(response, dict):
                # Some models return a dict with the text
                output_text = response.get('generated_text', '')
            else:
                output_text = str(response)
            
            # Parse regions if requested and available in response
            regions = None
            if with_region and isinstance(response, dict) and 'regions' in response:
                regions = response['regions']
            
            return OCRResult(
                text=output_text.strip(),
                method=f'huggingface_{self.config.model_path.split("/")[-1]}',
                confidence=response.get('confidence', 1.0),
                regions=regions,
                raw_output=response
            )
            
        except Exception as e:
            return OCRResult(
                text='',
                method='huggingface_api',
                error=str(e)
            ) 