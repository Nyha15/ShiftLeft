#!/usr/bin/env python3
"""
Data Models for Robotics Analyzer
=================================

Structured data classes for robot kinematics and task information.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class JointInfo:
    """Structured joint information"""
    name: str
    joint_type: str  # 'revolute', 'prismatic', 'continuous', 'fixed'
    axis: List[float]
    origin: List[float]
    limits: Optional[Dict[str, float]]
    parent_link: str
    child_link: str
    safety_limits: Optional[Dict[str, float]] = None

@dataclass
class LinkInfo:
    """Structured link information"""
    name: str
    visual_geometry: Optional[Dict] = None
    collision_geometry: Optional[Dict] = None
    inertial: Optional[Dict] = None

@dataclass
class RobotKinematics:
    """Complete robot kinematics information"""
    name: str
    joints: List[JointInfo]
    links: List[LinkInfo]
    base_link: str
    end_effector: Optional[str]
    dof: int
    urdf_path: str

@dataclass
class TaskInfo:
    """Task information extracted from code"""
    name: str
    description: str
    task_type: str  # 'manipulation', 'navigation', 'grasping', etc.
    required_actions: List[str]
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: float
    complexity: str  # 'simple', 'medium', 'complex'
    confidence: float
    file_path: str

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    repository_path: str
    robots: List[RobotKinematics]
    tasks: List[TaskInfo]
    config_files: List[str]
    summary: Dict[str, Any]
    confidence: float
    analysis_time: str
