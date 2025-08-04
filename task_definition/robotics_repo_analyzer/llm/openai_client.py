"""
OpenAI Client

This module provides a client for the OpenAI API.
"""

import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Client for the OpenAI API.
    """

    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            logger.warning("No OpenAI API key provided. LLM functionality will be disabled.")
            self.enabled = False
        else:
            try:
                import openai
                openai.api_key = self.api_key
                self.openai = openai
                self.enabled = True
                logger.info(f"OpenAI client initialized with model: {model}")
            except ImportError:
                logger.warning("Failed to import openai. Please install it with 'pip install openai'.")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.enabled = False

    def analyze(self, prompt: str) -> str:
        """
        Analyze a prompt using the OpenAI API.

        Args:
            prompt: Prompt to analyze

        Returns:
            Response from the OpenAI API
        """
        if not self.enabled:
            logger.warning("OpenAI client is not enabled. Returning empty response.")
            return ""

        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a robotics expert analyzing code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Error calling OpenAI API: {e}")
            return ""