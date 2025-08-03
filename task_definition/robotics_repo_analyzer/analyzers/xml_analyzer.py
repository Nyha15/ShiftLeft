"""
XML Analyzer Module

This module provides functionality to analyze XML files (URDF, SDF, etc.) for robotics-related information.
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# XML namespaces
NAMESPACES = {
    'sdf': 'http://sdformat.org/schemas/root.xsd',
    'mjcf': 'http://www.mujoco.org/mjcf',
    'robot': ''  # URDF has no namespace
}

def extract_robot_specs_from_xml(file_path: Path) -> Dict[str, Any]:
    """
    Extract robot specifications from an XML file (URDF, SDF, etc.).
    
    Args:
        file_path: Path to the XML file
        
    Returns:
            Dictionary containing robot specifications
    """
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Initialize robot specs
        robot_specs = {
            'name': None,
            'dof': 0,
            'joint_names': [],
            'joint_limits': [],
            'joint_types': [],
            'confidence': 0.0
        }
        
        # Determine file type based on root tag
        file_type = root.tag.split('}')[-1]  # Remove namespace if present
        
        if file_type == 'robot':  # URDF
            return _extract_from_urdf(root, robot_specs)
        elif file_type == 'sdf':  # SDF
            return _extract_from_sdf(root, robot_specs)
        elif file_type in ['mujoco', 'worldbody']:  # MuJoCo
            return _extract_from_mujoco(root, robot_specs)
        else:
            logger.warning(f"Unknown XML file type: {file_type}")
            return None
    except Exception as e:
        logger.error(f"Error extracting robot specs from XML file {file_path}: {e}")
        return None


def _extract_from_urdf(root: ET.Element, robot_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract robot specifications from a URDF file.
    
    Args:
        root: Root element of the URDF file
        robot_specs: Dictionary to store robot specifications
        
    Returns:
        Updated robot specifications dictionary
    """
    # Extract robot name
    robot_specs['name'] = root.get('name')
    
    # Extract joints
    joints = root.findall('./joint')
    
    for joint in joints:
        joint_type = joint.get('type')
        
        # Skip fixed joints (they don't contribute to DOF)
        if joint_type == 'fixed':
            continue
            
        joint_name = joint.get('name')
        
        if joint_name:
            robot_specs['joint_names'].append(joint_name)
            robot_specs['joint_types'].append(joint_type)
            
            # Extract joint limits
            limit_elem = joint.find('./limit')
            if limit_elem is not None:
                lower = limit_elem.get('lower')
                upper = limit_elem.get('upper')
                
                if lower is not None and upper is not None:
                    try:
                        lower_val = float(lower)
                        upper_val = float(upper)
                        robot_specs['joint_limits'].append([lower_val, upper_val])
                    except ValueError:
                        # If conversion fails, use None
                        robot_specs['joint_limits'].append([None, None])
                else:
                    robot_specs['joint_limits'].append([None, None])
            else:
                robot_specs['joint_limits'].append([None, None])
    
    # Calculate DOF
    robot_specs['dof'] = len([jt for jt in robot_specs['joint_types'] if jt != 'fixed'])
    
    # Calculate confidence
    confidence = 0.0
    if robot_specs['name']:
        confidence += 0.1
    if robot_specs['dof'] > 0:
        confidence += 0.3
    if robot_specs['joint_names']:
        confidence += 0.3
    if robot_specs['joint_limits'] and any(limit[0] is not None for limit in robot_specs['joint_limits']):
        confidence += 0.3
    
    robot_specs['confidence'] = confidence
    
    return robot_specs


def _extract_from_sdf(root: ET.Element, robot_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract robot specifications from an SDF file.
    
    Args:
        root: Root element of the SDF file
        robot_specs: Dictionary to store robot specifications
        
    Returns:
        Updated robot specifications dictionary
    """
    # Find the model/robot element
    model = root.find('.//model')
    
    if model is None:
        return None
    
    # Extract robot name
    robot_specs['name'] = model.get('name')
    
    # Extract joints
    joints = model.findall('.//joint')
    
    for joint in joints:
        joint_type = joint.get('type')
        
        # Skip fixed joints (they don't contribute to DOF)
        if joint_type == 'fixed':
            continue
            
        joint_name = joint.get('name')
        
        if joint_name:
            robot_specs['joint_names'].append(joint_name)
            robot_specs['joint_types'].append(joint_type)
            
            # Extract joint limits
            axis_elem = joint.find('.//axis')
            if axis_elem is not None:
                limit_elem = axis_elem.find('.//limit')
                if limit_elem is not None:
                    lower = limit_elem.find('.//lower')
                    upper = limit_elem.find('.//upper')
                    
                    if lower is not None and upper is not None:
                        try:
                            lower_val = float(lower.text)
                            upper_val = float(upper.text)
                            robot_specs['joint_limits'].append([lower_val, upper_val])
                        except (ValueError, AttributeError):
                            # If conversion fails, use None
                            robot_specs['joint_limits'].append([None, None])
                    else:
                        robot_specs['joint_limits'].append([None, None])
                else:
                    robot_specs['joint_limits'].append([None, None])
            else:
                robot_specs['joint_limits'].append([None, None])
    
    # Calculate DOF
    robot_specs['dof'] = len([jt for jt in robot_specs['joint_types'] if jt != 'fixed'])
    
    # Calculate confidence
    confidence = 0.0
    if robot_specs['name']:
        confidence += 0.1
    if robot_specs['dof'] > 0:
        confidence += 0.3
    if robot_specs['joint_names']:
        confidence += 0.3
    if robot_specs['joint_limits'] and any(limit[0] is not None for limit in robot_specs['joint_limits']):
        confidence += 0.3
    
    robot_specs['confidence'] = confidence
    
    return robot_specs


def _extract_from_mujoco(root: ET.Element, robot_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract robot specifications from a MuJoCo XML file.
    
    Args:
        root: Root element of the MuJoCo XML file
        robot_specs: Dictionary to store robot specifications
        
    Returns:
        Updated robot specifications dictionary
    """
    # Extract robot name from model name
    model_name = root.get('model')
    if model_name:
        robot_specs['name'] = model_name
    
    # Find the worldbody element
    worldbody = root.find('.//worldbody')
    if worldbody is None:
        worldbody = root  # Some MuJoCo files have worldbody as root
    
    # Extract joints
    joints = root.findall('.//joint')
    
    for joint in joints:
        joint_type = joint.get('type', 'hinge')  # Default is hinge in MuJoCo
        joint_name = joint.get('name')
        
        if joint_name:
            robot_specs['joint_names'].append(joint_name)
            robot_specs['joint_types'].append(joint_type)
            
            # Extract joint limits
            range_attr = joint.get('range')
            if range_attr:
                try:
                    limits = [float(x) for x in range_attr.split()]
                    if len(limits) >= 2:
                        robot_specs['joint_limits'].append([limits[0], limits[1]])
                    else:
                        robot_specs['joint_limits'].append([None, None])
                except ValueError:
                    robot_specs['joint_limits'].append([None, None])
            else:
                robot_specs['joint_limits'].append([None, None])
    
    # Calculate DOF
    robot_specs['dof'] = len(robot_specs['joint_names'])
    
    # Calculate confidence
    confidence = 0.0
    if robot_specs['name']:
        confidence += 0.1
    if robot_specs['dof'] > 0:
        confidence += 0.3
    if robot_specs['joint_names']:
        confidence += 0.3
    if robot_specs['joint_limits'] and any(limit[0] is not None for limit in robot_specs['joint_limits']):
        confidence += 0.3
    
    robot_specs['confidence'] = confidence
    
    return robot_specs