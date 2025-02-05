"""
OCR processor module supporting multiple OCR engines.
"""

import pytesseract
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import json
from abc import ABC, abstractmethod
import base64
from io import BytesIO
import torch


class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    def _preprocess_image(self, image: Any) -> Image.Image:
        """Preprocess image to ensure correct format."""
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Handle grayscale images
            if len(image.shape) == 2:
                # Convert to RGB by stacking the grayscale channel
                image = Image.fromarray(image).convert('RGB')
            # Handle RGBA images
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = Image.fromarray(image[:, :, :3])
            else:
                image = Image.fromarray(image)
            return image
        elif isinstance(image, Image.Image):
            # Convert to RGB if needed
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    @abstractmethod
    def process_image(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Process a single image."""
        pass


class TesseractEngine(OCREngine):
    """Tesseract OCR engine implementation."""
    
    def process_image(
        self,
        image: Image.Image,
        lang: str = "eng",
        psm_mode: int = 6,
        oem_mode: int = 3,
        whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        **kwargs
    ) -> Dict[str, Any]:
        """Process image with Tesseract OCR."""
        # Preprocess image
        image = self._preprocess_image(image)
        
        custom_config = f'--oem {oem_mode} --psm {psm_mode} -c tessedit_char_whitelist="{whitelist}"'
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        return {
            'text': text.strip(),
            'method': 'tesseract',
            'lang': lang
        }


class FlorenceEngine(OCREngine):
    """Florence-2 OCR engine implementation using HuggingFace transformers."""
    
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FlorenceEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str = "microsoft/Florence-2-large-ft"):
        """Initialize Florence engine with HuggingFace transformers."""
        if self._model is not None:
            return
            
        try:
            print(f"Initializing Florence model with path: {model_path}")
            
            # First check if transformers is installed
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                print("Successfully imported transformers")
            except ImportError as e:
                raise ImportError("transformers is not installed. Please install it with: pip install transformers") from e
            
            # Clean up GPU memory first
            if torch.cuda.is_available():
                print("Cleaning up GPU memory...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                import gc
                gc.collect()
                
            # Set up device and dtype
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"Using device: {device}, dtype: {torch_dtype}")
            
            # Initialize model and processor
            print("Loading model and processor...")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(device)
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.device = device
            self.torch_dtype = torch_dtype
            print("Model and processor loaded successfully")
            
        except Exception as e:
            print(f"Error initializing Florence model: {str(e)}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            self._model = None
            self._processor = None
            raise RuntimeError(f"Failed to initialize Florence model: {str(e)}")

    def process_image(
        self,
        image: Image.Image,
        with_region: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Process image with Florence-2 OCR."""
        try:
            # Check if model is properly initialized
            if self._model is None or self._processor is None:
                raise RuntimeError("Florence model is not initialized")

            # Preprocess image and ensure RGB
            image = self._preprocess_image(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create prompt based on mode
            prompt = "<OCR_WITH_REGION>" if with_region else "<OCR>"
            
            # Process inputs
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)
            
            # Generate output
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
            
            # Decode output
            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]
            
            # Post-process output if region detection is requested
            if with_region:
                parsed_output = self._processor.post_process_generation(
                    generated_text,
                    task="<OCR_WITH_REGION>",
                    image_size=(image.width, image.height)
                )
                return {
                    'text': generated_text.strip(),
                    'method': 'florence_with_region',
                    'raw_output': parsed_output,
                    'boxes': parsed_output.get('boxes', []),
                    'labels': parsed_output.get('labels', [])
                }
            else:
                return {
                    'text': generated_text.strip(),
                    'method': 'florence',
                    'raw_output': {
                        'generated_text': generated_text
                    }
                }
            
        except Exception as e:
            print(f"Error processing image with Florence: {str(e)}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'method': 'florence',
                'text': '',
            }


class QwenEngine(OCREngine):
    """Qwen2-VL-7B OCR engine implementation using HuggingFace transformers."""
    
    _instance = None
    _model = None
    _processor = None
    _tokenizer = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QwenEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """Initialize Qwen engine with HuggingFace transformers."""
        if self._model is not None:
            return
            
        try:
            print(f"Initializing Qwen model with path: {model_path}")
            
            # First check if transformers is installed
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
                from qwen_vl_utils import process_vision_info
                print("Successfully imported transformers and qwen_vl_utils")
            except ImportError as e:
                raise ImportError("Required packages not installed. Please install with: pip install transformers qwen_vl_utils") from e
            
            # Clean up GPU memory first
            if torch.cuda.is_available():
                print("Cleaning up GPU memory...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                import gc
                gc.collect()
            
            # Initialize model with flash attention
            print("Loading model and processor...")
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            
            # Initialize processor with custom pixel settings for better memory usage
            min_pixels = 256*28*28  # Minimum size for good OCR
            max_pixels = 1280*28*28  # Maximum size to balance memory and quality
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            
            # Store process_vision_info function
            self.process_vision_info = process_vision_info
            print("Model and processor loaded successfully")
            
        except Exception as e:
            print(f"Error initializing Qwen model: {str(e)}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            self._model = None
            self._processor = None
            raise RuntimeError(f"Failed to initialize Qwen model: {str(e)}")

    def process_image(
        self,
        image: Image.Image,
        with_region: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Process image with Qwen2-VL OCR."""
        try:
            # Check if model is properly initialized
            if self._model is None or self._processor is None:
                raise RuntimeError("Qwen model is not initialized")

            # Preprocess image and ensure RGB
            image = self._preprocess_image(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create messages with appropriate prompt
            prompt = (
                "Please analyze this image and identify all text regions. For each text region, "
                "provide the text content and its location in the format 'text: (x1, y1, x2, y2)'."
            ) if with_region else "Please extract and transcribe all text visible in this image."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare inputs for inference
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to appropriate device
            device = next(self._model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate output with better parameters
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                num_beams=3,
                length_penalty=1.0,
                repetition_penalty=1.1
            )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse region information if requested
            if with_region:
                try:
                    # Extract region information using regex
                    import re
                    regions = []
                    pattern = r'([^:]+):\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
                    matches = re.finditer(pattern, output_text)
                    
                    for match in matches:
                        text = match.group(1).strip()
                        coords = [int(match.group(i)) for i in range(2, 6)]
                        regions.append({
                            'text': text,
                            'box': coords
                        })
                    
                    return {
                        'text': output_text.strip(),
                        'method': 'qwen_with_region',
                        'regions': regions,
                        'raw_output': {
                            'generated_text': output_text
                        }
                    }
                except Exception as e:
                    print(f"Error parsing region information: {e}")
            
            return {
                'text': output_text.strip(),
                'method': 'qwen_with_region' if with_region else 'qwen',
                'raw_output': {
                    'generated_text': output_text
                }
            }
            
        except Exception as e:
            print(f"Error processing image with Qwen: {str(e)}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'method': 'qwen',
                'text': '',
            }


class OCRProcessor:
    """Main class for OCR processing, supporting multiple OCR engines."""
    
    def __init__(self, skip_vllm: bool = False):
        """Initialize OCR processor with multiple engines.
        
        Args:
            skip_vllm (bool): If True, skip initialization of VLLM-based models
        """
        # Initialize Tesseract engine (always available)
        self.tesseract_engine = TesseractEngine()
        
        # Initialize transformer-based engines only if not skipped
        self.florence_engine = None
        self.qwen_engine = None
        if not skip_vllm:
            try:
                self.florence_engine = FlorenceEngine()
            except Exception as e:
                print(f"Failed to initialize Florence engine: {e}")
                self.florence_engine = None
                
            try:
                self.qwen_engine = QwenEngine()
            except Exception as e:
                print(f"Failed to initialize Qwen engine: {e}")
                self.qwen_engine = None
    
    def process_batch(self, images: List[Image.Image], method: str = "tesseract", **kwargs) -> List[Dict]:
        """Process a batch of images with the specified OCR method.
        
        Args:
            images: List of PIL images to process
            method: OCR method to use ('tesseract', 'florence', 'qwen', etc.)
            **kwargs: Additional arguments for the specific OCR method
        
        Returns:
            List of dictionaries containing OCR results
        """
        if not images:
            return []
            
        results = []
        for img in images:
            if method == "tesseract":
                result = self.tesseract_engine.process_image(img, **kwargs)
            elif method in ["florence", "florence_with_region"]:
                if self.florence_engine is None:
                    raise RuntimeError("Florence engine is not available. Please check installation and GPU requirements.")
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('with_region', None)
                result = self.florence_engine.process_image(
                    img, 
                    with_region=(method == "florence_with_region"),
                    **kwargs_copy
                )
            elif method in ["qwen", "qwen_with_region"]:
                if self.qwen_engine is None:
                    raise RuntimeError("Qwen engine is not available. Please check installation and GPU requirements.")
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('with_region', None)
                result = self.qwen_engine.process_image(
                    img, 
                    with_region=(method == "qwen_with_region"),
                    **kwargs_copy
                )
            else:
                raise ValueError(f"Unknown OCR method: {method}")
            
            results.append(result)
        
        return results

    def add_engine(self, name: str, engine: OCREngine):
        """Add a new OCR engine."""
        self.engines[name] = engine 