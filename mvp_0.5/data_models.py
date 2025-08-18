#!/usr/bin/env python3
"""
Data Models for Robotics Analyzer
=================================

Structured data classes for robot kinematics and task information.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from enum import Enum

class SweepMethod(Enum):
    """Parameter sweep methods"""
    SOBOL = "sobol"
    OAT = "oat"  # One-At-a-Time
    LATIN_HYPERCUBE = "latin_hypercube"
    RANDOM = "random"

class ParameterPriority(Enum):
    """Parameter priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ParameterSweep:
    """Parameter sweep configuration"""
    method: SweepMethod
    range: List[float]  # [min, max]
    priority: ParameterPriority
    samples: int = 100
    nominal_value: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None

@dataclass
class SensitivityResult:
    """Sensitivity analysis result for a parameter"""
    parameter_name: str
    sobol_index: float
    variance_contribution: float
    priority: ParameterPriority
    confidence: float

@dataclass
class SimulationResult:
    """Result from a single simulation run"""
    seed: int
    parameters: Dict[str, float]
    success: bool
    metrics: Dict[str, float]  # KPI values
    failure_reason: Optional[str] = None

@dataclass
class TaskParameters:
    """Enhanced task parameters with sweep definitions"""
    name: str
    nominal_value: float
    unit: str
    sweep: ParameterSweep
    source: str  # 'urdf', 'config', 'code', 'user'
    confidence: float

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

@dataclass
class CIConfig:
    """CI pipeline configuration"""
    repo_url: str
    local_path: str
    simulation_engine: str  # 'mujoco', 'pybullet', 'gazebo'
    sensitivity_method: SweepMethod
    max_samples: int
    random_seed: int
    output_dir: str

@dataclass
class CIRunResult:
    """Complete CI pipeline run result"""
    config: CIConfig
    extracted_parameters: Dict[str, TaskParameters]
    sensitivity_results: List[SensitivityResult]
    simulation_results: List[SimulationResult]
    updated_yamls: List[str]
    artifacts_dir: str
    run_time: float
    success: bool
