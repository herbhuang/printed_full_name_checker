"""
Base classes and utilities for OCR/LLM models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np

@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str
    model_path: str
    model_type: str  # 'local' or 'api'
    requires_gpu: bool = False
    max_tokens: int = 1024
    temperature: float = 0.0
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    additional_config: Optional[Dict[str, Any]] = None

@dataclass
class OCRResult:
    """Standardized OCR result format."""
    text: str
    method: str
    confidence: float = 0.0
    regions: Optional[List[Dict[str, Any]]] = None
    raw_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BaseLLM(ABC):
    """Abstract base class for LLM models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._processor = None
        
    def _preprocess_image(self, image: Any) -> Image.Image:
        """Preprocess image to ensure correct format."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = Image.fromarray(image).convert('RGB')
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = Image.fromarray(image[:, :, :3])
            else:
                image = Image.fromarray(image)
            return image
        elif isinstance(image, Image.Image):
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        with_region: bool = False,
        **kwargs
    ) -> OCRResult:
        """Process a single image."""
        pass
    
    def is_available(self) -> bool:
        """Check if the model is available and loaded."""
        return self._model is not None and self._processor is not None

class BaseAPILLM(BaseLLM):
    """Base class for API-based LLM models."""
    
    def load_model(self) -> None:
        """No model loading needed for API-based models."""
        if not self.config.api_key:
            raise ValueError(f"API key required for {self.config.name}")
        if not self.config.api_url:
            raise ValueError(f"API URL required for {self.config.name}")
        self._model = True  # Mark as loaded
        self._processor = True
    
    @abstractmethod
    def _call_api(
        self,
        image: Image.Image,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make API call to the service."""
        pass

class BaseLocalLLM(BaseLLM):
    """Base class for local LLM models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if self.config.requires_gpu:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.config.requires_gpu and self.device == "cpu":
                raise RuntimeError(f"{self.config.name} requires GPU but none is available") 