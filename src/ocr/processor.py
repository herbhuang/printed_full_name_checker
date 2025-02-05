"""
OCR processor module supporting multiple OCR engines.
"""

from typing import Dict, List, Optional, Type, Union
from PIL import Image

from src.ocr.base import BaseLLM, ModelConfig, OCRResult
from src.ocr.qwen import QwenVLModel
from src.ocr.florence import FlorenceModel
from src.ocr.huggingface_api import HuggingFaceAPIModel

print("Loading OCRProcessor module...")  # Debug print

class OCRProcessor:
    """Main class for OCR processing, supporting multiple OCR engines."""
    
    # Default model configurations
    DEFAULT_CONFIGS = {
        'qwen': ModelConfig(
            name='Qwen2.5-VL',
            model_path='Qwen/Qwen2.5-VL-7B-Instruct',
            model_type='local',
            requires_gpu=True,
            max_tokens=1024,
            temperature=0.0
        ),
        'florence': ModelConfig(
            name='Florence-2',
            model_path='microsoft/florence-2-base',  # Using base model by default
            model_type='local',
            requires_gpu=True,
            max_tokens=1024,
            temperature=0.0
        ),
        'llava': ModelConfig(
            name='LLaVA',
            model_path='llava-hf/llava-1.5-7b-hf',
            model_type='api',
            requires_gpu=False,
            max_tokens=1024,
            temperature=0.0
        )
    }
    
    def __init__(self, skip_local_models: bool = False):
        """Initialize OCR processor with multiple engines."""
        print(f"Initializing OCRProcessor with skip_local_models={skip_local_models}")  # Debug print
        
        self.models: Dict[str, BaseLLM] = {}
        
        # Initialize available_models with all default configs
        self.available_models = self.DEFAULT_CONFIGS.copy()
        print(f"Initial available models: {list(self.available_models.keys())}")  # Debug print
        
        # Initialize local models if not skipped
        if not skip_local_models:
            try:
                print("Attempting to initialize Qwen model...")  # Debug print
                self.models['qwen'] = QwenVLModel(self.DEFAULT_CONFIGS['qwen'])
                self.models['qwen'].load_model()
                print("Qwen model initialized successfully")  # Debug print
            except Exception as e:
                print(f"Failed to initialize Qwen model: {e}")  # Debug print
                self.available_models.pop('qwen', None)
            
            try:
                print("Attempting to initialize Florence model...")  # Debug print
                self.models['florence'] = FlorenceModel(self.DEFAULT_CONFIGS['florence'])
                self.models['florence'].load_model()
                print("Florence model initialized successfully")  # Debug print
            except Exception as e:
                print(f"Failed to initialize Florence model: {e}")  # Debug print
                self.available_models.pop('florence', None)
        else:
            print("Skipping local models initialization")  # Debug print
            self.available_models.pop('qwen', None)
            self.available_models.pop('florence', None)
        
        print(f"Final available models: {list(self.available_models.keys())}")  # Debug print
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        # Ensure we only return models that are actually loaded
        available = [name for name, model in self.models.items() if model.is_available()]
        print(f"get_available_models called, returning: {available}")  # Debug print
        return available
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        print(f"get_model_config called for {model_name}")  # Debug print
        return self.available_models.get(model_name)
    
    def add_model(
        self,
        name: str,
        model_class: Type[BaseLLM],
        config: ModelConfig
    ) -> None:
        """Add a new model to the processor.
        
        Args:
            name: Unique identifier for the model
            model_class: Class implementing BaseLLM
            config: Model configuration
        """
        try:
            model = model_class(config)
            model.load_model()
            self.models[name] = model
            self.available_models[name] = config
            print(f"Successfully added model: {name}")  # Debug print
        except Exception as e:
            print(f"Failed to add model {name}: {str(e)}")  # Debug print
            raise RuntimeError(f"Failed to add model {name}: {str(e)}")
    
    def add_api_model(
        self,
        name: str,
        model_path: str,
        api_key: str,
        api_url: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a new API-based model.
        
        Args:
            name: Unique identifier for the model
            model_path: Hugging Face model path or API model identifier
            api_key: API key for authentication
            api_url: Optional custom API endpoint
            **kwargs: Additional configuration options
        """
        config = ModelConfig(
            name=name,
            model_path=model_path,
            model_type='api',
            api_key=api_key,
            api_url=api_url,
            **kwargs
        )
        self.add_model(name, HuggingFaceAPIModel, config)
    
    def process_batch(
        self,
        images: List[Image.Image],
        model_name: str,
        prompt: Optional[str] = None,
        with_region: bool = False,
        **kwargs
    ) -> List[OCRResult]:
        """Process a batch of images with the specified model.
        
        Args:
            images: List of PIL images to process
            model_name: Name of the model to use
            prompt: Optional custom prompt/instruction
            with_region: Whether to extract region information
            **kwargs: Additional model-specific parameters
        
        Returns:
            List of OCR results
        """
        if not images:
            return []
        
        # Initialize model if it's available but not loaded
        if model_name in self.available_models and model_name not in self.models:
            config = self.available_models[model_name]
            if config.model_type == 'api':
                self.add_api_model(
                    name=model_name,
                    model_path=config.model_path,
                    api_key=config.api_key,
                    api_url=config.api_url
                )
            else:
                if config.requires_gpu:
                    self.add_model(model_name, QwenVLModel if 'qwen' in model_name.lower() else FlorenceModel, config)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {self.get_available_models()}")
        
        model = self.models[model_name]
        results = []
        
        for img in images:
            try:
                result = model.process_image(
                    image=img,
                    prompt=prompt,
                    with_region=with_region,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                results.append(OCRResult(
                    text='',
                    method=model_name,
                    error=str(e)
                ))
        
        return results 