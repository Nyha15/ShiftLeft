"""
Batch processor for LLM requests.
"""

import logging
import re
import json
import hashlib
from typing import Dict, List, Any, Optional

from robotics_repo_analyzer.llm.prompts import BATCH_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Batch processor for LLM requests.
    """
    
    def __init__(self, llm_client, batch_size=3):
        """
        Initialize the batch processor.
        
        Args:
            llm_client: LLM client for code analysis
            batch_size: Maximum number of code sections per batch
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
    
    def process_complex_sections(self, complex_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process complex code sections in batches for efficiency.
        
        Args:
            complex_sections: List of complex code sections
            
        Returns:
            List of processed results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(complex_sections), self.batch_size):
            batch = complex_sections[i:i+self.batch_size]
            
            # Create combined prompt for the batch
            combined_prompt = self._create_batch_prompt(batch)
            
            # Create cache key
            cache_key = hashlib.md5(combined_prompt.encode()).hexdigest()
            
            # Send to LLM
            response = self.llm_client.analyze_code(combined_prompt, cache_key=cache_key)
            
            # Parse batch response
            batch_results = self._parse_batch_response(response, batch)
            results.extend(batch_results)
        
        return results
    
    def _create_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for a batch of code sections.
        
        Args:
            batch: Batch of code sections
            
        Returns:
            Prompt for the batch
        """
        sections_text = ""
        for i, section in enumerate(batch):
            sections_text += f"""
            SECTION {i+1}:
            {'Class' if section.get('type') == 'class' else 'Function'}: {section['name']}
            File: {section.get('file', 'unknown')}
            
            ```python
            {section['code']}
            ```
            
            """
        
        return BATCH_ANALYSIS_PROMPT.format(
            num_sections=len(batch),
            sections_text=sections_text
        )
    
    def _parse_batch_response(self, response: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse a batch response from the LLM.
        
        Args:
            response: LLM response
            batch: Batch of code sections
            
        Returns:
            List of parsed results
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response
            
            # Clean up the JSON string
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)  # Remove comments
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Map results back to original sections
            results = []
            for section_data in data.get('sections', []):
                section_id = section_data.get('section_id', 1) - 1
                if 0 <= section_id < len(batch):
                    section = batch[section_id]
                    results.append({
                        'name': section['name'],
                        'file': section.get('file', 'unknown'),
                        'robot_config': section_data.get('robot_config', {}),
                        'tasks': section_data.get('tasks', {}),
                        'parameters': section_data.get('parameters', {}),
                        'purpose': section_data.get('purpose', ''),
                        '_meta': {
                            'complexity': section.get('complexity', 0.0),
                            'relevance': section.get('relevance', 0.0),
                            'type': section.get('type', 'function'),
                            'lineno': section.get('lineno', 0)
                        }
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            return []