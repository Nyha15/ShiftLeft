"""
Pattern Utilities Module

This module provides utility functions for pattern matching.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Pattern, Match

logger = logging.getLogger(__name__)

def compile_patterns(patterns: List[str]) -> List[Pattern]:
    """
    Compile a list of regex patterns.
    
    Args:
        patterns: List of regex patterns
        
    Returns:
        List of compiled patterns
    """
    compiled_patterns = []
    for pattern in patterns:
        try:
            compiled_patterns.append(re.compile(pattern))
        except re.error as e:
            logger.warning(f"Error compiling pattern '{pattern}': {e}")
    return compiled_patterns

def find_matches(text: str, patterns: List[Pattern]) -> List[Match]:
    """
    Find all matches of patterns in text.
    
    Args:
        text: Text to search
        patterns: List of compiled patterns
        
    Returns:
        List of matches
    """
    matches = []
    for pattern in patterns:
        matches.extend(pattern.finditer(text))
    return matches

def extract_named_groups(match: Match) -> Dict[str, str]:
    """
    Extract named groups from a regex match.
    
    Args:
        match: Regex match
        
    Returns:
        Dictionary mapping group names to values
    """
    return match.groupdict()

def extract_values(text: str, pattern: str) -> List[str]:
    """
    Extract values from text using a regex pattern.
    
    Args:
        text: Text to search
        pattern: Regex pattern
        
    Returns:
        List of extracted values
    """
    try:
        matches = re.findall(pattern, text)
        return matches
    except re.error as e:
        logger.warning(f"Error extracting values with pattern '{pattern}': {e}")
        return []

def contains_pattern(text: str, pattern: str) -> bool:
    """
    Check if text contains a pattern.
    
    Args:
        text: Text to search
        pattern: Regex pattern
        
    Returns:
        True if the text contains the pattern, False otherwise
    """
    try:
        return bool(re.search(pattern, text))
    except re.error as e:
        logger.warning(f"Error checking pattern '{pattern}': {e}")
        return False