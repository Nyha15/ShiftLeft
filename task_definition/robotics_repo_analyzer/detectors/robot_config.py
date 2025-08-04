"""
Robot Configuration Detector

This module detects robot configurations from various sources in a robotics repository.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import xml.etree.ElementTree as ET

from task_definition.robotics_repo_analyzer.analyzers.code_analyzer import extract_robot_specs_from_python
from task_definition.robotics_repo_analyzer.analyzers.xml_analyzer import extract_robot_specs_from_xml
from task_definition.robotics_repo_analyzer.analyzers.config_analyzer import extract_robot_specs_from_config
from task_definition.robotics_repo_analyzer.utils.confidence import calculate_confidence

logger = logging.getLogger(__name__)

class RobotConfigDetector:
    """
    Detector for robot configurations in a repository.
    """
    
    def __init__(self):
        """Initialize the robot configuration detector."""
        self.robot_specs = {
            'name': None,
            'dof': None,
            'joint_names': [],
            'joint_limits': [],
            'joint_types': [],
            'sources': [],
            'confidence': 0.0
        }
        
    def detect(self, files_by_type: Dict[str, List[Path]]) -> Dict[str, Any]:
        """
        Detect robot configurations from various sources.
        
        Args:
            files_by_type: Dictionary mapping file types to lists of file paths
            
        Returns:
            Dictionary containing detected robot configurations
        """
        logger.info("Detecting robot configurations...")
        
        # Extract robot specs from different file types
        python_specs = self._extract_from_python(files_by_type['python'])
        xml_specs = self._extract_from_xml(files_by_type['xml'])
        config_specs = self._extract_from_config(files_by_type['config'])
        
        # Merge specs from different sources
        merged_specs = self._merge_specs([python_specs, xml_specs, config_specs])
        
        # Calculate overall confidence
        merged_specs['confidence'] = calculate_confidence(
            merged_specs, 
            ['dof', 'joint_names', 'joint_limits']
        )
        
        logger.info(f"Robot configuration detection complete. "
                   f"Confidence: {merged_specs['confidence']:.2f}")
        
        return merged_specs
    
    def _extract_from_python(self, python_files: List[Path]) -> Dict[str, Any]:
        """
        Extract robot specifications from Python files.
        
        Args:
            python_files: List of Python file paths
            
        Returns:
            Dictionary containing robot specifications extracted from Python files
        """
        logger.debug(f"Extracting robot specs from {len(python_files)} Python files")
        
        all_specs = []
        for file_path in python_files:
            try:
                specs = extract_robot_specs_from_python(file_path)
                if specs:
                    specs['source'] = str(file_path)
                    all_specs.append(specs)
            except Exception as e:
                logger.warning(f"Error extracting robot specs from {file_path}: {e}")
        
        # Merge specs from all Python files
        merged_specs = self._merge_specs(all_specs)
        merged_specs['source_type'] = 'python'
        
        return merged_specs
    
    def _extract_from_xml(self, xml_files: List[Path]) -> Dict[str, Any]:
        """
        Extract robot specifications from XML/URDF/SDF files.
        
        Args:
            xml_files: List of XML file paths
            
        Returns:
            Dictionary containing robot specifications extracted from XML files
        """
        logger.debug(f"Extracting robot specs from {len(xml_files)} XML files")
        
        all_specs = []
        for file_path in xml_files:
            try:
                specs = extract_robot_specs_from_xml(file_path)
                if specs:
                    specs['source'] = str(file_path)
                    all_specs.append(specs)
            except Exception as e:
                logger.warning(f"Error extracting robot specs from {file_path}: {e}")
        
        # Merge specs from all XML files
        merged_specs = self._merge_specs(all_specs)
        merged_specs['source_type'] = 'xml'
        
        return merged_specs
    
    def _extract_from_config(self, config_files: List[Path]) -> Dict[str, Any]:
        """
        Extract robot specifications from config files.
        
        Args:
            config_files: List of config file paths
            
        Returns:
            Dictionary containing robot specifications extracted from config files
        """
        logger.debug(f"Extracting robot specs from {len(config_files)} config files")
        
        all_specs = []
        for file_path in config_files:
            try:
                specs = extract_robot_specs_from_config(file_path)
                if specs:
                    specs['source'] = str(file_path)
                    all_specs.append(specs)
            except Exception as e:
                logger.warning(f"Error extracting robot specs from {file_path}: {e}")
        
        # Merge specs from all config files
        merged_specs = self._merge_specs(all_specs)
        merged_specs['source_type'] = 'config'
        
        return merged_specs
    
    def _merge_specs(self, specs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge robot specifications from multiple sources.
        
        Args:
            specs_list: List of robot specification dictionaries
            
        Returns:
            Merged robot specifications
        """
        if not specs_list:
            return {
                'name': None,
                'dof': None,
                'joint_names': [],
                'joint_limits': [],
                'joint_types': [],
                'sources': [],
                'confidence': 0.0
            }
        
        # Sort specs by confidence (if available)
        sorted_specs = sorted(
            [s for s in specs_list if s],
            key=lambda s: s.get('confidence', 0),
            reverse=True
        )
        
        # Start with the highest confidence specs
        merged = dict(sorted_specs[0])
        
        # Track sources
        sources = set()
        if 'source' in merged:
            sources.add(merged['source'])
        
        # Merge in other specs
        for specs in sorted_specs[1:]:
            if 'source' in specs:
                sources.add(specs['source'])
            
            # Fill in missing values
            for key, value in specs.items():
                if key in ['source', 'confidence', 'source_type']:
                    continue
                
                if not merged.get(key) and value:
                    merged[key] = value
                    
                # Special handling for lists
                if key in ['joint_names', 'joint_limits', 'joint_types'] and value:
                    if not merged.get(key):
                        merged[key] = value
                    elif len(value) > len(merged[key]):
                        merged[key] = value
        
        # Update sources
        merged['sources'] = list(sources)
        
        return merged