"""
Florence-2 model implementation using Hugging Face transformers.
"""

import torch
from PIL import Image
from typing import Dict, Any, Optional

from src.ocr.base import BaseLocalLLM, ModelConfig, OCRResult

class FlorenceModel(BaseLocalLLM):
    """Florence-2 model implementation."""
    
    def load_model(self) -> None:
        """Load Florence model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM 

            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                import gc
                gc.collect()
            
            # Set up device and dtype
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Florence model: {str(e)}")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        with_region: bool = False,
        **kwargs
    ) -> OCRResult:
        """Process image with Florence-2."""
        try:
            # Ensure model is loaded
            if not self.is_available():
                raise RuntimeError("Florence model is not initialized")
            
            # Preprocess image
            image = self._preprocess_image(image)
            
            # Create prompt based on mode
            if not prompt:
                prompt = "<OCR_WITH_REGION>" if with_region else "<OCR>"
            
            # Process inputs
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate output
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=False,
                    num_beams=3
                )
            
            # Decode output
            generated_text = self._processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]
            
            # Post-process output if region detection is requested
            if with_region:
                # Extract regions using regex
                import re
                regions = []
                pattern = r'([^:]+):\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
                matches = re.finditer(pattern, generated_text)
                
                for match in matches:
                    text = match.group(1).strip()
                    coords = [int(match.group(i)) for i in range(2, 6)]
                    regions.append({
                        'text': text,
                        'box': coords
                    })
                
                return OCRResult(
                    text=generated_text.strip(),
                    method='florence_with_region',
                    confidence=1.0,  # Florence doesn't provide confidence scores
                    regions=regions,
                    raw_output={'generated_text': generated_text}
                )
            else:
                return OCRResult(
                    text=generated_text.strip(),
                    method='florence',
                    confidence=1.0,
                    raw_output={'generated_text': generated_text}
                )
            
        except Exception as e:
            return OCRResult(
                text='',
                method='florence',
                error=str(e)
            )