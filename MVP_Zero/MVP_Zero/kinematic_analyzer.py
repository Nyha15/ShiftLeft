#!/usr/bin/env python3
"""
Kinematic Analyzer
==================

Advanced URDF parsing and kinematic analysis with urdfpy integration and XML fallback.
"""

import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict, Any

from data_models import JointInfo, LinkInfo, RobotKinematics

# Optional dependencies with graceful fallback
try:
    import numpy as np
    from urdfpy import URDF
    URDF_AVAILABLE = True
except ImportError:
    URDF_AVAILABLE = False

logger = logging.getLogger(__name__)

class KinematicAnalyzer:
    """Advanced URDF and kinematic analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.KinematicAnalyzer")
        if not URDF_AVAILABLE:
            self.logger.warning("urdfpy not available. Using XML fallback for URDF parsing.")
    
    def discover_config_files(self, repo_path: Path) -> List[Path]:
        """Discover robot configuration files in the repository."""
        config_files = []
        
        # Look for URDF files
        for urdf_file in repo_path.rglob('*.urdf'):
            config_files.append(urdf_file)
        
        # Look for XML files that might be robot descriptions
        for xml_file in repo_path.rglob('*.xml'):
            if self._is_robot_xml(xml_file):
                config_files.append(xml_file)
        
        return config_files
    
    def _is_robot_xml(self, xml_path: Path) -> bool:
        """Check if XML file contains robot description."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Check for URDF-style robot tag
            if root.tag == 'robot':
                return True
            
            # Check for MuJoCo robot models
            if root.tag == 'mujoco':
                # Look for joints, bodies, or actuators
                if (root.find('.//joint') is not None or 
                    root.find('.//body') is not None or
                    root.find('.//actuator') is not None):
                    return True
            
            # Check for common robotics XML patterns
            robot_indicators = ['joint', 'link', 'actuator', 'sensor']
            for indicator in robot_indicators:
                if root.find(f'.//{indicator}') is not None:
                    return True
            
            return False
        except Exception:
            return False
    
    def find_urdf_files(self, repo_path: Path) -> List[Path]:
        """Find all URDF files in the repository"""
        urdf_files = []
        for pattern in ['*.urdf', '*.xacro']:
            urdf_files.extend(repo_path.rglob(pattern))
        return urdf_files
    
    def parse_urdf_file(self, urdf_path: Path) -> Optional[RobotKinematics]:
        """Parse URDF file with fallback methods"""
        try:
            # Check if this is a MuJoCo XML file
            if urdf_path.suffix.lower() == '.xml':
                tree = ET.parse(urdf_path)
                root = tree.getroot()
                if root.tag == 'mujoco':
                    return self._parse_mujoco_xml(root, urdf_path)
            
            if URDF_AVAILABLE:
                return self._parse_with_urdfpy(urdf_path)
            else:
                return self._parse_with_xml(urdf_path)
        except Exception as e:
            self.logger.error(f"Failed to parse URDF {urdf_path}: {e}")
            return None
    
    def _parse_with_urdfpy(self, urdf_path: Path) -> Optional[RobotKinematics]:
        """Parse using urdfpy library"""
        try:
            robot = URDF.load(str(urdf_path))
            
            joints = []
            for joint in robot.joints:
                joint_info = JointInfo(
                    name=joint.name,
                    joint_type=joint.joint_type,
                    axis=joint.axis.tolist() if joint.axis is not None else [0, 0, 0],
                    origin=joint.origin.reshape(-1).tolist() if joint.origin is not None else [0, 0, 0, 0, 0, 0, 1],
                    limits=self._extract_joint_limits(joint),
                    parent_link=joint.parent,
                    child_link=joint.child
                )
                joints.append(joint_info)
            
            links = []
            for link in robot.links:
                link_info = LinkInfo(
                    name=link.name,
                    visual_geometry=self._extract_geometry(link.visuals[0].geometry) if link.visuals else None,
                    collision_geometry=self._extract_geometry(link.collisions[0].geometry) if link.collisions else None,
                    inertial=self._extract_inertial(link.inertial) if link.inertial else None
                )
                links.append(link_info)
            
            dof = sum(1 for joint in joints if joint.joint_type in ['revolute', 'prismatic', 'continuous'])
            end_effector = self._identify_end_effector(joints, links)
            base_link = self._identify_base_link(joints, links)
            
            return RobotKinematics(
                name=robot.name or urdf_path.stem,
                joints=joints,
                links=links,
                base_link=base_link,
                end_effector=end_effector,
                dof=dof,
                urdf_path=str(urdf_path)
            )
            
        except Exception as e:
            self.logger.error(f"urdfpy parsing failed: {e}")
            return None
    
    def _parse_with_xml(self, urdf_path: Path) -> Optional[RobotKinematics]:
        """Fallback XML parsing"""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            if root.tag != 'robot':
                return None
            
            robot_name = root.get('name', urdf_path.stem)
            
            joints = []
            for joint_elem in root.findall('joint'):
                joint_info = self._parse_joint_xml(joint_elem)
                if joint_info:
                    joints.append(joint_info)
            
            links = []
            for link_elem in root.findall('link'):
                link_info = self._parse_link_xml(link_elem)
                if link_info:
                    links.append(link_info)
            
            dof = sum(1 for joint in joints if joint.joint_type in ['revolute', 'prismatic', 'continuous'])
            end_effector = self._identify_end_effector(joints, links)
            base_link = self._identify_base_link(joints, links)
            
            return RobotKinematics(
                name=robot_name,
                joints=joints,
                links=links,
                base_link=base_link,
                end_effector=end_effector,
                dof=dof,
                urdf_path=str(urdf_path)
            )
            
        except Exception as e:
            self.logger.error(f"XML parsing failed: {e}")
            return None
    
    def _extract_joint_limits(self, joint) -> Optional[Dict[str, float]]:
        """Extract joint limits from urdfpy joint"""
        if joint.limit is None:
            return None
        
        limits = {}
        if joint.limit.lower is not None:
            limits['lower'] = float(joint.limit.lower)
        if joint.limit.upper is not None:
            limits['upper'] = float(joint.limit.upper)
        if joint.limit.effort is not None:
            limits['effort'] = float(joint.limit.effort)
        if joint.limit.velocity is not None:
            limits['velocity'] = float(joint.limit.velocity)
        
        return limits if limits else None
    
    def _extract_geometry(self, geometry) -> Dict[str, Any]:
        """Extract geometry information"""
        if hasattr(geometry, 'box') and geometry.box is not None:
            return {'type': 'box', 'size': geometry.box.size.tolist()}
        elif hasattr(geometry, 'cylinder') and geometry.cylinder is not None:
            return {'type': 'cylinder', 'radius': float(geometry.cylinder.radius), 'length': float(geometry.cylinder.length)}
        elif hasattr(geometry, 'sphere') and geometry.sphere is not None:
            return {'type': 'sphere', 'radius': float(geometry.sphere.radius)}
        elif hasattr(geometry, 'mesh') and geometry.mesh is not None:
            return {'type': 'mesh', 'filename': str(geometry.mesh.filename), 'scale': geometry.mesh.scale.tolist() if geometry.mesh.scale is not None else [1, 1, 1]}
        return {'type': 'unknown'}
    
    def _extract_inertial(self, inertial) -> Dict[str, Any]:
        """Extract inertial information"""
        result = {}
        if hasattr(inertial, 'mass') and inertial.mass is not None:
            result['mass'] = float(inertial.mass)
        if hasattr(inertial, 'inertia') and inertial.inertia is not None:
            result['inertia'] = inertial.inertia.tolist()
        return result
    
    def _parse_joint_xml(self, joint_elem) -> Optional[JointInfo]:
        """Parse joint from XML element"""
        try:
            name = joint_elem.get('name')
            joint_type = joint_elem.get('type', 'fixed')
            
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            
            if parent_elem is None or child_elem is None:
                return None
            
            parent_link = parent_elem.get('link')
            child_link = child_elem.get('link')
            
            # Parse axis
            axis_elem = joint_elem.find('axis')
            axis = [0, 0, 1]
            if axis_elem is not None:
                xyz = axis_elem.get('xyz', '0 0 1')
                axis = [float(x) for x in xyz.split()]
            
            # Parse origin
            origin_elem = joint_elem.find('origin')
            origin = [0, 0, 0, 0, 0, 0, 1]
            if origin_elem is not None:
                xyz = origin_elem.get('xyz', '0 0 0')
                rpy = origin_elem.get('rpy', '0 0 0')
                xyz_vals = [float(x) for x in xyz.split()]
                rpy_vals = [float(x) for x in rpy.split()]
                origin = xyz_vals + rpy_vals + [1]
            
            # Parse limits
            limits = None
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                limits = {}
                for attr in ['lower', 'upper', 'effort', 'velocity']:
                    val = limit_elem.get(attr)
                    if val is not None:
                        limits[attr] = float(val)
            
            return JointInfo(
                name=name,
                joint_type=joint_type,
                axis=axis,
                origin=origin,
                limits=limits,
                parent_link=parent_link,
                child_link=child_link
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse joint XML: {e}")
            return None
    
    def _parse_link_xml(self, link_elem) -> Optional[LinkInfo]:
        """Parse link from XML element"""
        try:
            name = link_elem.get('name')
            return LinkInfo(name=name)
        except Exception as e:
            self.logger.warning(f"Failed to parse link XML: {e}")
            return None
    
    def _identify_end_effector(self, joints: List[JointInfo], links: List[LinkInfo]) -> Optional[str]:
        """Identify end effector using naming patterns and topology"""
        end_effector_patterns = [
            r'.*gripper.*', r'.*hand.*', r'.*end.*effector.*', r'.*tool.*',
            r'.*finger.*', r'.*tip.*', r'.*tcp.*'
        ]
        
        # Find leaf links (not parent to any joint)
        parent_links = {joint.parent_link for joint in joints}
        child_links = {joint.child_link for joint in joints}
        leaf_links = child_links - parent_links
        
        # Check leaf links for end effector patterns
        for link_name in leaf_links:
            for pattern in end_effector_patterns:
                if re.search(pattern, link_name.lower()):
                    return link_name
        
        return next(iter(leaf_links)) if leaf_links else None
    
    def _identify_base_link(self, joints: List[JointInfo], links: List[LinkInfo]) -> str:
        """Identify base link"""
        base_patterns = [r'.*base.*', r'.*root.*', r'.*world.*', r'.*link0.*']
        
        # Find root links (not child to any joint)
        parent_links = {joint.parent_link for joint in joints}
        child_links = {joint.child_link for joint in joints}
        root_links = parent_links - child_links
        
        # Check root links for base patterns
        for link_name in root_links:
            for pattern in base_patterns:
                if re.search(pattern, link_name.lower()):
                    return link_name
        
        return next(iter(root_links)) if root_links else (links[0].name if links else "base_link")
    
    def _parse_mujoco_xml(self, root, xml_path: Path) -> RobotKinematics:
        """Parse MuJoCo XML format."""
        joints = []
        links = []
        
        # Parse MuJoCo joints
        for joint_elem in root.findall('.//joint'):
            joint_name = joint_elem.get('name', 'unnamed_joint')
            joint_type = joint_elem.get('type', 'hinge')  # MuJoCo default
            
            # Convert MuJoCo joint types to URDF equivalents
            type_mapping = {
                'hinge': 'revolute',
                'slide': 'prismatic',
                'ball': 'spherical',
                'free': 'floating'
            }
            joint_type = type_mapping.get(joint_type, joint_type)
            
            # Extract axis
            axis_str = joint_elem.get('axis', '0 0 1')
            axis = [float(x) for x in axis_str.split()]
            
            # Extract range (MuJoCo limits)
            range_str = joint_elem.get('range')
            limits = None
            if range_str:
                range_vals = [float(x) for x in range_str.split()]
                if len(range_vals) >= 2:
                    limits = {
                        'lower': range_vals[0],
                        'upper': range_vals[1],
                        'effort': 100.0,  # default
                        'velocity': 10.0   # default
                    }
            
            # Find parent body (ElementTree doesn't have getparent, so we'll use a different approach)
            parent_name = 'unknown'
            # Look for the body that contains this joint
            for body_elem in root.findall('.//body'):
                if joint_elem in body_elem:
                    parent_name = body_elem.get('name', 'unknown')
                    break
            
            joint_info = JointInfo(
                name=joint_name,
                joint_type=joint_type,
                axis=axis,
                origin=[0, 0, 0, 0, 0, 0, 1],  # default
                limits=limits,
                parent_link=parent_name,
                child_link=joint_name + '_child',
                safety_limits=None
            )
            joints.append(joint_info)
        
        # Parse MuJoCo bodies as links
        for body_elem in root.findall('.//body'):
            body_name = body_elem.get('name', 'unnamed_body')
            
            # Extract inertial properties
            inertial_elem = body_elem.find('inertial')
            mass = 1.0  # default
            inertia_matrix = [1, 0, 0, 1, 0, 1]  # default
            
            if inertial_elem is not None:
                mass = float(inertial_elem.get('mass', '1.0'))
                
                # MuJoCo uses different inertia format
                fullinertia = inertial_elem.get('fullinertia')
                if fullinertia:
                    inertia_vals = [float(x) for x in fullinertia.split()]
                    if len(inertia_vals) >= 6:
                        inertia_matrix = inertia_vals[:6]
                else:
                    diaginertia = inertial_elem.get('diaginertia')
                    if diaginertia:
                        diag_vals = [float(x) for x in diaginertia.split()]
                        if len(diag_vals) >= 3:
                            inertia_matrix = [diag_vals[0], 0, 0, diag_vals[1], 0, diag_vals[2]]
            
            link_info = LinkInfo(
                name=body_name
            )
            links.append(link_info)
        
        # Calculate DOF (count non-fixed joints)
        dof = len([j for j in joints if j.joint_type in ['revolute', 'prismatic', 'continuous']])
        
        # Identify base and end-effector
        base_link = 'link0' if any(l.name == 'link0' for l in links) else (links[0].name if links else 'base_link')
        end_effector = 'hand' if any(l.name == 'hand' for l in links) else (links[-1].name if links else 'end_effector')
        
        return RobotKinematics(
            name=xml_path.stem,
            joints=joints,
            links=links,
            dof=dof,
            base_link=base_link,
            end_effector=end_effector,
            urdf_path=str(xml_path)
        )
