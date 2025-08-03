"""
Confidence Utilities Module

This module provides utility functions for calculating confidence scores.
"""

import logging
from typing import Dict, List, Any, Set, Tuple, Optional

logger = logging.getLogger(__name__)

def calculate_confidence(data: Dict[str, Any], required_keys: List[str]) -> float:
    """
    Calculate a confidence score based on the presence and quality of required keys.
    
    Args:
        data: Data to calculate confidence for
        required_keys: List of required keys
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not data:
        return 0.0
    
    # Check if the data already has a confidence score
    if 'confidence' in data and isinstance(data['confidence'], (int, float)):
        return float(data['confidence'])
    
    # Calculate confidence based on required keys
    total_score = 0.0
    max_score = len(required_keys)
    
    for key in required_keys:
        value = data.get(key)
        
        if value is None:
            continue
            
        # Different scoring based on value type
        if isinstance(value, (list, tuple, set)):
            if value:
                # Score based on list length (more items = higher confidence)
                score = min(1.0, len(value) / 10.0)
                total_score += score
        elif isinstance(value, dict):
            if value:
                # Score based on dictionary size
                score = min(1.0, len(value) / 5.0)
                total_score += score
        elif isinstance(value, (int, float)):
            if value > 0:
                total_score += 1.0
        elif isinstance(value, str):
            if value:
                total_score += 1.0
        elif value:
            total_score += 1.0
    
    # Normalize score
    if max_score > 0:
        return total_score / max_score
    else:
        return 0.0

def combine_confidences(confidences: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Combine multiple confidence scores.
    
    Args:
        confidences: List of confidence scores
        weights: Optional list of weights
        
    Returns:
        Combined confidence score
    """
    if not confidences:
        return 0.0
    
    # Use equal weights if not provided
    if weights is None:
        weights = [1.0] * len(confidences)
    
    # Ensure weights and confidences have the same length
    if len(weights) != len(confidences):
        weights = weights[:len(confidences)] + [1.0] * (len(confidences) - len(weights))
    
    # Calculate weighted average
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(c * w for c, w in zip(confidences, weights))
    return weighted_sum / total_weight

def adjust_confidence(base_confidence: float, adjustment: float) -> float:
    """
    Adjust a confidence score.
    
    Args:
        base_confidence: Base confidence score
        adjustment: Adjustment to apply
        
    Returns:
        Adjusted confidence score
    """
    adjusted = base_confidence + adjustment
    return max(0.0, min(1.0, adjusted))  # Clamp to [0.0, 1.0]