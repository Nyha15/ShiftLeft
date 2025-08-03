"""
LLM client for external API-based language models.
"""

import logging
import hashlib
import json
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for external API-based language models.
    """
    
    def __init__(self, provider="openai", api_key=None, model=None):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider (e.g., 'openai', 'anthropic')
            api_key: API key for the provider
            model: Model name to use
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._get_default_model(provider)
        self.request_count = 0
        self.total_tokens = 0
        self.cache = {}  # Simple in-memory cache
        
        # Check if we have the required dependencies
        self.available = self._check_availability()
        if not self.available:
            logger.warning(f"{provider} client not available. Install the required dependencies.")
    
    def analyze_code(self, prompt: str, cache_key: Optional[str] = None) -> str:
        """
        Send code to LLM for analysis.
        
        Args:
            prompt: The prompt to send to the LLM
            cache_key: Optional cache key to avoid duplicate requests
            
        Returns:
            The LLM response
        """
        if not self.available:
            return f"{self.provider} client not available"
        
        # Check cache first
        if cache_key and cache_key in self.cache:
            logger.debug(f"Using cached response for {cache_key[:10]}...")
            return self.cache[cache_key]
        
        # Generate cache key if not provided
        if not cache_key:
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
        
        # Call the appropriate provider
        if self.provider == "openai":
            response = self._call_openai(prompt)
        elif self.provider == "anthropic":
            response = self._call_anthropic(prompt)
        else:
            response = f"Unsupported provider: {self.provider}"
        
        # Cache the result
        self.cache[cache_key] = response
        
        # Update stats
        self.request_count += 1
        
        return response
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The API response
        """
        try:
            import openai
            
            # Set API key
            openai.api_key = self.api_key
            
            # Call API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes robotics code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Update token count
            self.total_tokens += response.usage.total_tokens
            
            # Extract content
            return response.choices[0].message.content
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return "OpenAI package not installed"
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error: {str(e)}"
    
    def _call_anthropic(self, prompt: str) -> str:
        """
        Call the Anthropic API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The API response
        """
        try:
            import anthropic
            
            # Create client
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Call API
            response = client.completions.create(
                model=self.model,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=1500,
                temperature=0.1
            )
            
            # Extract content
            return response.completion
        except ImportError:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            return "Anthropic package not installed"
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return f"Error: {str(e)}"
    
    def _get_default_model(self, provider: str) -> str:
        """
        Get the default model for a provider.
        
        Args:
            provider: LLM provider
            
        Returns:
            Default model name
        """
        if provider == "openai":
            return "gpt-3.5-turbo"
        elif provider == "anthropic":
            return "claude-2"
        else:
            return "unknown"
    
    def _check_availability(self) -> bool:
        """
        Check if the required dependencies are available.
        
        Returns:
            True if available, False otherwise
        """
        if self.provider == "openai":
            try:
                import openai
                return True
            except ImportError:
                return False
        elif self.provider == "anthropic":
            try:
                import anthropic
                return True
            except ImportError:
                return False
        return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        # Calculate approximate cost
        cost = 0.0
        if self.provider == "openai":
            if "gpt-4" in self.model:
                cost = self.total_tokens / 1000 * 0.06  # Approximate cost for GPT-4
            else:
                cost = self.total_tokens / 1000 * 0.002  # Approximate cost for GPT-3.5
        elif self.provider == "anthropic":
            cost = self.total_tokens / 1000 * 0.01  # Approximate cost for Claude
        
        return {
            'provider': self.provider,
            'model': self.model,
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'estimated_cost': cost
        }