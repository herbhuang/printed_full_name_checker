"""
Qwen2.5-VL model implementation.
"""

import torch
from PIL import Image
from typing import Dict, Any, Optional
import re

from src.ocr.base import BaseLocalLLM, ModelConfig, OCRResult

class QwenVLModel(BaseLocalLLM):
    """Qwen2.5-VL model implementation."""
    
    def load_model(self) -> None:
        """Load Qwen model and processor."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            from qwen_vl_utils import process_vision_info

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                import gc
                gc.collect()
            
            # Load model with flash attention
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            # Store process_vision_info function for later use
            self._process_vision_info = process_vision_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qwen model: {str(e)}")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        with_region: bool = False,
        **kwargs
    ) -> OCRResult:
        """Process image with Qwen2.5-VL."""
        try:
            # Ensure model is loaded
            if not self.is_available():
                raise RuntimeError("Qwen model is not initialized")
            
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
            
            # Create messages in the format expected by Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ]
            
            # Get device
            device = next(self._model.parameters()).device
            
            try:
                # Process conversation
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Process vision info
                image_inputs, video_inputs = self._process_vision_info(messages)
                
                # Process inputs
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Generate output
                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        do_sample=False,
                        temperature=self.config.temperature,
                        pad_token_id=self._processor.tokenizer.pad_token_id,
                        bos_token_id=self._processor.tokenizer.bos_token_id,
                        eos_token_id=self._processor.tokenizer.eos_token_id
                    )
                
                # Move outputs back to CPU for decoding
                outputs = outputs.cpu()
                
                # Decode output
                generated_text = self._processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the model's response
                if "Assistant:" in generated_text:
                    generated_text = generated_text.split("Assistant:")[-1].strip()
                elif "assistant" in generated_text.lower():
                    generated_text = generated_text.split("assistant")[-1].strip(":\n ").strip()
                
                # Parse regions if requested
                regions = None
                if with_region:
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
                    method='qwen_with_region' if with_region else 'qwen',
                    confidence=1.0,  # Qwen doesn't provide confidence scores
                    regions=regions,
                    raw_output={'generated_text': generated_text}
                )
                
            except Exception as e:
                raise RuntimeError(f"Error during processing: {str(e)}")
            
        except Exception as e:
            return OCRResult(
                text='',
                method='qwen',
                error=str(e)
            )
