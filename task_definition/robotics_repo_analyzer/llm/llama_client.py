"""
Client for local Llama models.
"""

import logging
import hashlib
import json
import re
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LlamaClient:
    """
    Client for local Llama models.
    """
    
    def __init__(self, model_path=None, context_length=2048, n_threads=4):
        """
        Initialize the Llama client.
        
        Args:
            model_path: Path to the Llama model
            context_length: Context length for the model
            n_threads: Number of threads to use
        """
        self.model_path = model_path
        self.context_length = context_length
        self.n_threads = n_threads
        self.request_count = 0
        self.cache = {}  # Simple in-memory cache
        
        # Import llama_cpp conditionally to avoid dependency issues
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=model_path or self._find_default_model(),
                n_ctx=context_length,
                n_threads=n_threads
            )
            self.available = True
            logger.info(f"Initialized Llama model from {self.model_path}")
        except ImportError:
            logger.warning("llama_cpp package not installed. Install with: pip install llama-cpp-python")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Llama: {e}")
            self.available = False
    
    def analyze_code(self, prompt: str, cache_key: Optional[str] = None) -> str:
        """
        Send code to Llama for analysis.
        
        Args:
            prompt: The prompt to send to Llama
            cache_key: Optional cache key to avoid duplicate requests
            
        Returns:
            The Llama response
        """
        if not self.available:
            return "Llama model not available"
        
        # Check cache first
        if cache_key and cache_key in self.cache:
            logger.debug(f"Using cached response for {cache_key[:10]}...")
            return self.cache[cache_key]
        
        # Generate cache key if not provided
        if not cache_key:
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
        
        # Prepare prompt for Llama
        llama_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Generate response
        try:
            response = self.llm(
                llama_prompt,
                max_tokens=1024,
                temperature=0.1,
                stop=["</s>"]
            )
            result = response["choices"][0]["text"]
            logger.info(f"Llama response: {result[:200]}...")
        except Exception as e:
            logger.error(f"Error calling Llama: {e}")
            result = f"Error: {str(e)}"
        
        # Cache the result
        self.cache[cache_key] = result
        
        # Update stats
        self.request_count += 1
        
        return result
    
    def _find_default_model(self) -> str:
        """
        Find a default Llama model.
        
        Returns:
            Path to the default model
        """
        # Common locations for Llama models
        common_locations = [
            os.path.expanduser("~/models"),
            os.path.expanduser("~/.cache/huggingface/hub"),
            "/models",
            "/usr/local/share/models",
            "/workspace/models"
        ]
        
        # Common model filenames
        model_patterns = [
            "llama-2-7b*.gguf",
            "llama-7b*.gguf",
            "llama2-7b*.gguf",
            "llama-2-7b*.bin",
            "llama-7b*.bin",
            "llama2-7b*.bin"
        ]
        
        # Search for models
        for location in common_locations:
            if not os.path.exists(location):
                continue
            
            for root, _, files in os.walk(location):
                for file in files:
                    for pattern in model_patterns:
                        if re.match(pattern.replace("*", ".*"), file):
                            return os.path.join(root, file)
        
        # Default fallback
        return "models/llama-2-7b-chat.gguf"
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'provider': 'llama_local',
            'model': os.path.basename(self.model_path) if self.model_path else 'unknown',
            'request_count': self.request_count,
            'context_length': self.context_length,
            'n_threads': self.n_threads
        }