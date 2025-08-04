"""
Repository Scanner Module

This module provides functionality to scan a robotics repository and categorize
files for further analysis.
"""

import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple, Any

from task_definition.robotics_repo_analyzer.detectors.robot_config import RobotConfigDetector
from task_definition.robotics_repo_analyzer.detectors.entry_points import EntryPointDetector
from task_definition.robotics_repo_analyzer.detectors.action_sequences import ActionSequenceDetector
from task_definition.robotics_repo_analyzer.detectors.parameters import ParameterDetector
from task_definition.robotics_repo_analyzer.detectors.task_organizer import TaskOrganizer

logger = logging.getLogger(__name__)

# File extensions to analyze
PYTHON_EXTENSIONS = {'.py', '.pyw'}
XML_EXTENSIONS = {'.xml', '.urdf', '.sdf', '.xacro'}
CONFIG_EXTENSIONS = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
NOTEBOOK_EXTENSIONS = {'.ipynb'}
LAUNCH_EXTENSIONS = {'.launch'}  # ROS launch files

# Directories to exclude
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'venv', 'env', '.env', 'node_modules',
    'build', 'dist', '.eggs', '*.egg-info', '.pytest_cache', '.mypy_cache',
    '.tox', '.coverage', 'htmlcov', '.idea', '.vscode'
}

class RepositoryScanner:
    """
    Scanner for robotics repositories that categorizes files and runs detectors.
    """
    
    def __init__(self, repo_path: str, use_llm: bool = False, llm_client = None, 
                complexity_threshold: float = 0.7):
        """
        Initialize the repository scanner.
        
        Args:
            repo_path: Path to the repository
            use_llm: Whether to use LLM for analysis
            llm_client: LLM client for code analysis
            complexity_threshold: Threshold for using LLM (0.0-1.0)
        """
        self.repo_path = Path(repo_path).resolve()
        self.files_by_type: Dict[str, List[Path]] = {
            'python': [],
            'xml': [],
            'config': [],
            'notebook': [],
            'launch': [],
            'other': []
        }
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.complexity_threshold = complexity_threshold
        
        # Initialize framework analyzers
        self.framework_analyzers = []
        try:
            from task_definition.robotics_repo_analyzer.frameworks import (
                MujocoAnalyzer, PyBulletAnalyzer, ROSAnalyzer
            )
            self.framework_analyzers = [
                MujocoAnalyzer(),
                PyBulletAnalyzer(),
                ROSAnalyzer()
            ]
        except ImportError as e:
            logger.warning(f"Could not import framework analyzers: {e}")
        
        # Initialize detectors
        self.detectors = {
            'robot_config': RobotConfigDetector(),
            'entry_points': EntryPointDetector(),
            'action_sequences': ActionSequenceDetector(),
            'parameters': ParameterDetector(),
            'task_organizer': TaskOrganizer()
        }
        
    def _should_exclude(self, path: Path) -> bool:
        """
        Check if a path should be excluded from analysis.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path should be excluded, False otherwise
        """
        for part in path.parts:
            if any(part == exclude or part.startswith(exclude.rstrip('*')) 
                  for exclude in EXCLUDE_DIRS if exclude.endswith('*')):
                return True
            if part in EXCLUDE_DIRS:
                return True
        return False
    
    def _categorize_file(self, file_path: Path) -> None:
        """
        Categorize a file based on its extension.
        
        Args:
            file_path: Path to the file
        """
        suffix = file_path.suffix.lower()
        
        if suffix in PYTHON_EXTENSIONS:
            self.files_by_type['python'].append(file_path)
        elif suffix in XML_EXTENSIONS:
            self.files_by_type['xml'].append(file_path)
        elif suffix in CONFIG_EXTENSIONS:
            self.files_by_type['config'].append(file_path)
        elif suffix in NOTEBOOK_EXTENSIONS:
            self.files_by_type['notebook'].append(file_path)
        elif suffix in LAUNCH_EXTENSIONS:
            self.files_by_type['launch'].append(file_path)
        else:
            self.files_by_type['other'].append(file_path)
    
    def _scan_files(self) -> None:
        """Scan and categorize all files in the repository."""
        logger.info(f"Scanning repository: {self.repo_path}")
        
        for root, dirs, files in os.walk(self.repo_path):
            # Filter out excluded directories in-place
            dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d)]
            
            for file in files:
                file_path = Path(root) / file
                if not self._should_exclude(file_path):
                    self._categorize_file(file_path)
        
        # Log summary of files found
        for file_type, files in self.files_by_type.items():
            logger.info(f"Found {len(files)} {file_type} files")
    
    def scan(self) -> Dict[str, Any]:
        """
        Scan the repository and run all detectors.
        
        Returns:
            Dictionary containing all scan results
        """
        self._scan_files()
        
        # First pass: Detect frameworks
        frameworks_detected = self._detect_frameworks()
        logger.info(f"Detected frameworks: {frameworks_detected}")
        
        # Second pass: Run framework-specific analyzers
        framework_results = self._run_framework_analyzers(frameworks_detected)
        
        # Third pass: Run traditional detectors
        traditional_results = self._run_traditional_detectors()
        
        # Fourth pass: Use LLM for complex code sections if enabled
        llm_results = {}
        if self.use_llm and self.llm_client:
            llm_results = self._run_llm_analysis()
        
        # Merge all results
        results = self._merge_results(traditional_results, framework_results, llm_results)
        
        # Add file statistics to results
        results['file_stats'] = {
            file_type: len(files) for file_type, files in self.files_by_type.items()
        }
        
        # Add detected frameworks
        results['frameworks_detected'] = frameworks_detected
        
        return results
        
    def _detect_frameworks(self) -> List[str]:
        """
        Detect robotics frameworks used in the repository.
        
        Returns:
            List of detected framework names
        """
        frameworks = []
        
        # Check Python files for framework imports
        for analyzer in self.framework_analyzers:
            if analyzer.detect_in_files(self.files_by_type['python']):
                # Get the class name without 'Analyzer'
                framework_name = analyzer.__class__.__name__.replace('Analyzer', '').lower()
                frameworks.append(framework_name)
        
        return frameworks
        
    def _run_framework_analyzers(self, frameworks_detected: List[str]) -> Dict[str, Any]:
        """
        Run framework-specific analyzers.
        
        Args:
            frameworks_detected: List of detected frameworks
            
        Returns:
            Dictionary containing framework-specific analysis results
        """
        framework_results = {
            'robot_config': {},
            'action_sequences': {},
            'parameters': {},
            'tasks': {}
        }
        
        # Skip if no frameworks detected
        if not frameworks_detected:
            return framework_results
        
        # Run analyzers for detected frameworks
        for analyzer in self.framework_analyzers:
            # Get the framework name
            framework_name = analyzer.__class__.__name__.replace('Analyzer', '').lower()
            
            # Skip if framework not detected
            if framework_name not in frameworks_detected:
                continue
            
            logger.info(f"Running {framework_name} analyzer...")
            
            # Analyze Python files
            for file_path in self.files_by_type['python']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Extract robot configuration
                    robot_config = analyzer.extract_robot_config(code, file_path)
                    if robot_config and robot_config.get('confidence', 0) > 0:
                        framework_results['robot_config'][str(file_path)] = robot_config
                    
                    # Extract action sequences
                    action_sequences = analyzer.extract_action_sequences(code, file_path)
                    if action_sequences and action_sequences.get('confidence', 0) > 0:
                        framework_results['action_sequences'][str(file_path)] = action_sequences
                        
                        # Organize action sequences into tasks
                        tasks = self.detectors['task_organizer'].organize_tasks(action_sequences)
                        if tasks and tasks.get('confidence', 0) > 0:
                            framework_results['tasks'][str(file_path)] = tasks
                    
                    # Extract parameters
                    parameters = analyzer.extract_parameters(code, file_path)
                    if parameters and parameters.get('confidence', 0) > 0:
                        framework_results['parameters'][str(file_path)] = parameters
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} with {framework_name} analyzer: {e}")
        
        return framework_results
        
    def _run_traditional_detectors(self) -> Dict[str, Any]:
        """
        Run traditional detectors.
        
        Returns:
            Dictionary containing detector results
        """
        results = {}
        
        # Run detectors in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                'robot_config': executor.submit(
                    self.detectors['robot_config'].detect, 
                    self.files_by_type
                ),
                'entry_points': executor.submit(
                    self.detectors['entry_points'].detect, 
                    self.files_by_type
                ),
                'action_sequences': executor.submit(
                    self.detectors['action_sequences'].detect, 
                    self.files_by_type
                ),
                'parameters': executor.submit(
                    self.detectors['parameters'].detect, 
                    self.files_by_type
                )
            }
            
            # Collect results
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Error in {name} detector: {e}")
                    results[name] = {'error': str(e)}
        
        # Organize action sequences into tasks
        if 'action_sequences' in results and isinstance(results['action_sequences'], dict) and \
           not results['action_sequences'].get('error'):
            try:
                # Use the task organizer to organize action sequences into tasks
                results['tasks'] = self.detectors['task_organizer'].organize_tasks(results['action_sequences'])
                logger.info(f"Organized action sequences into {len(results['tasks'].get('tasks', {}))} tasks")
            except Exception as e:
                logger.error(f"Error organizing tasks: {e}")
                results['tasks'] = {'error': str(e)}
        
        return results
        
    def _run_llm_analysis(self) -> Dict[str, Any]:
        """
        Run LLM analysis on complex code sections.
        
        Returns:
            Dictionary containing LLM analysis results
        """
        llm_results = {
            'robot_config': {},
            'action_sequences': {},
            'parameters': {}
        }
        
        try:
            from task_definition.robotics_repo_analyzer.llm.analyzer import HybridCodeAnalyzer
            
            # Initialize hybrid analyzer
            hybrid_analyzer = HybridCodeAnalyzer(
                llm_client=self.llm_client,
                complexity_threshold=self.complexity_threshold
            )
            
            # Analyze Python files
            for file_path in self.files_by_type['python']:
                try:
                    # Skip very large files
                    if file_path.stat().st_size > 1_000_000:  # 1MB
                        logger.warning(f"Skipping large file for LLM analysis: {file_path}")
                        continue
                    
                    # Analyze file
                    logger.info(f"Running LLM analysis on {file_path}")
                    analysis = hybrid_analyzer.analyze_file(file_path)
                    
                    # Check for errors in analysis
                    if 'error' in analysis:
                        logger.warning(f"Error in LLM analysis of {file_path}: {analysis['error']}")
                        continue
                    
                    # Extract LLM-derived robot configuration
                    if 'llm_robot_configs' in analysis and analysis['llm_robot_configs']:
                        for config in analysis['llm_robot_configs']:
                            if isinstance(config, dict) and 'config' in config:
                                llm_results['robot_config'][str(file_path)] = {
                                    'config': config['config'],
                                    'confidence': config.get('confidence', 0.6),
                                    'source': str(file_path)
                                }
                    
                    # Extract LLM-derived action sequences
                    if 'llm_analysis' in analysis and analysis['llm_analysis']:
                        for func_name, func_analysis in analysis['llm_analysis'].items():
                            if isinstance(func_analysis, dict) and 'tasks' in func_analysis:
                                tasks = func_analysis['tasks']
                                if tasks:  # Check if tasks is not empty
                                    llm_results['action_sequences'][f"{file_path}:{func_name}"] = {
                                        'sequences': [{
                                            'name': func_name,
                                            'source': str(file_path),
                                            'confidence': 0.6,
                                            'steps': self._convert_llm_tasks_to_steps(tasks)
                                        }],
                                        'confidence': 0.6
                                    }
                            
                            # Extract LLM-derived parameters
                            if isinstance(func_analysis, dict) and 'parameters' in func_analysis:
                                params = func_analysis['parameters']
                                if params:  # Check if parameters is not empty
                                    llm_results['parameters'][f"{file_path}:{func_name}"] = {
                                        'parameters': self._convert_llm_parameters(params, file_path),
                                        'confidence': 0.6
                                    }
                except Exception as e:
                    logger.warning(f"Error in LLM analysis of {file_path}: {e}")
        except ImportError as e:
            logger.warning(f"Could not import LLM analyzer: {e}")
        
        return llm_results
        
    def _convert_llm_tasks_to_steps(self, tasks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert LLM-derived tasks to action steps.
        
        Args:
            tasks: Tasks from LLM analysis
            
        Returns:
            List of action steps
        """
        steps = []
        
        # Handle different task formats
        if isinstance(tasks, list):
            # List of tasks
            for i, task in enumerate(tasks):
                if isinstance(task, str):
                    # Simple string task
                    steps.append({
                        'step': i + 1,
                        'action': task,
                        'parameters': {},
                        'source': 'LLM analysis'
                    })
                elif isinstance(task, dict):
                    # Dictionary task
                    steps.append({
                        'step': i + 1,
                        'action': task.get('action', 'unknown'),
                        'parameters': task.get('parameters', {}),
                        'source': 'LLM analysis'
                    })
        elif isinstance(tasks, dict):
            # Dictionary of tasks
            if 'task_name' in tasks and 'steps' in tasks:
                # Handle the specific format from our prompts
                if isinstance(tasks['steps'], list):
                    for i, step in enumerate(tasks['steps']):
                        if isinstance(step, dict) and 'action' in step:
                            steps.append({
                                'step': i + 1,
                                'action': step.get('action', 'unknown'),
                                'parameters': step.get('parameters', {}),
                                'source': 'LLM analysis'
                            })
            else:
                # Generic dictionary of tasks
                for i, (name, task) in enumerate(tasks.items()):
                    if isinstance(task, str):
                        # Simple string task
                        steps.append({
                            'step': i + 1,
                            'action': name,
                            'parameters': {'description': task},
                            'source': 'LLM analysis'
                        })
                    elif isinstance(task, dict):
                        # Dictionary task
                        steps.append({
                            'step': i + 1,
                            'action': name,
                            'parameters': task,
                            'source': 'LLM analysis'
                        })
        
        return steps
        
    def _convert_llm_parameters(self, parameters: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """
        Convert LLM-derived parameters to parameter list.
        
        Args:
            parameters: Parameters from LLM analysis
            file_path: Path to the file
            
        Returns:
            List of parameters
        """
        param_list = []
        
        # Handle different parameter formats
        if isinstance(parameters, list):
            # List of parameters
            for param in parameters:
                if isinstance(param, str):
                    # Simple string parameter
                    param_list.append({
                        'name': param,
                        'value': None,
                        'source': str(file_path),
                        'confidence': 0.5
                    })
                elif isinstance(param, dict):
                    # Dictionary parameter
                    param_list.append({
                        'name': param.get('name', 'unknown'),
                        'value': param.get('value'),
                        'source': str(file_path),
                        'confidence': 0.6
                    })
        elif isinstance(parameters, dict):
            # Dictionary of parameters
            for name, value in parameters.items():
                # Skip non-serializable values
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    # Convert lists and dicts to strings if they're too complex
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        value = str(value)[:100] + "..."
                    
                    param_list.append({
                        'name': name,
                        'value': value,
                        'source': str(file_path),
                        'confidence': 0.6
                    })
        
        return param_list
        
    def _merge_results(self, traditional_results: Dict[str, Any], 
                      framework_results: Dict[str, Any],
                      llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge results from different analysis methods.
        
        Args:
            traditional_results: Results from traditional detectors
            framework_results: Results from framework-specific analyzers
            llm_results: Results from LLM analysis
            
        Returns:
            Merged results
        """
        merged = dict(traditional_results)
        
        # Merge robot configuration
        if 'robot_config' in merged:
            # Add framework-specific robot configurations
            for file_path, config in framework_results.get('robot_config', {}).items():
                # If we already have a config with higher confidence, skip
                if merged['robot_config'].get('confidence', 0) > config.get('confidence', 0):
                    continue
                
                # Merge with existing config
                for key, value in config.items():
                    if key == 'confidence':
                        # Take the higher confidence
                        merged['robot_config'][key] = max(
                            merged['robot_config'].get(key, 0),
                            value
                        )
                    elif key == 'sources':
                        # Combine sources
                        merged['robot_config'][key] = list(set(
                            merged['robot_config'].get(key, []) + value
                        ))
                    elif not merged['robot_config'].get(key) and value:
                        # Add missing values
                        merged['robot_config'][key] = value
            
            # Add LLM-derived robot configurations
            for file_path, config_data in llm_results.get('robot_config', {}).items():
                config = config_data.get('config', {})
                confidence = config_data.get('confidence', 0.6)
                
                # If we already have a config with higher confidence, skip
                if merged['robot_config'].get('confidence', 0) > confidence:
                    continue
                
                # Merge with existing config
                for key, value in config.items():
                    if not merged['robot_config'].get(key) and value:
                        # Add missing values
                        merged['robot_config'][key] = value
                        
                # Add source
                if 'sources' in merged['robot_config']:
                    merged['robot_config']['sources'].append(file_path)
                else:
                    merged['robot_config']['sources'] = [file_path]
        
        # Merge action sequences
        if 'action_sequences' in merged:
            # Add framework-specific action sequences
            for file_path, sequences in framework_results.get('action_sequences', {}).items():
                if 'sequences' in sequences:
                    merged['action_sequences']['sequences'].extend(sequences['sequences'])
                    
                    # Update confidence
                    merged['action_sequences']['confidence'] = max(
                        merged['action_sequences'].get('confidence', 0),
                        sequences.get('confidence', 0)
                    )
            
            # Add LLM-derived action sequences
            for file_path, sequences in llm_results.get('action_sequences', {}).items():
                if 'sequences' in sequences:
                    merged['action_sequences']['sequences'].extend(sequences['sequences'])
                    
                    # Update confidence
                    merged['action_sequences']['confidence'] = max(
                        merged['action_sequences'].get('confidence', 0),
                        sequences.get('confidence', 0)
                    )
        
        # Merge parameters
        if 'parameters' in merged:
            # Add framework-specific parameters
            for file_path, params in framework_results.get('parameters', {}).items():
                if 'parameters' in params:
                    merged['parameters']['parameters'].extend(params['parameters'])
                    
                    # Update confidence
                    merged['parameters']['confidence'] = max(
                        merged['parameters'].get('confidence', 0),
                        params.get('confidence', 0)
                    )
                    
                    # Merge grouped parameters
                    for group, group_params in params.get('grouped_parameters', {}).items():
                        if group not in merged['parameters'].get('grouped_parameters', {}):
                            merged['parameters'].setdefault('grouped_parameters', {})[group] = []
                        merged['parameters']['grouped_parameters'][group].extend(group_params)
            
            # Add LLM-derived parameters
            for file_path, params in llm_results.get('parameters', {}).items():
                if 'parameters' in params:
                    merged['parameters']['parameters'].extend(params['parameters'])
                    
                    # Update confidence
                    merged['parameters']['confidence'] = max(
                        merged['parameters'].get('confidence', 0),
                        params.get('confidence', 0)
                    )
        
        # Merge framework-specific tasks
        if 'tasks' in merged:
            # Add framework-specific tasks
            for file_path, tasks_data in framework_results.get('tasks', {}).items():
                if 'tasks' in tasks_data:
                    # Merge tasks by name
                    for task_name, task_data in tasks_data['tasks'].items():
                        if task_name not in merged['tasks'].get('tasks', {}):
                            merged['tasks'].setdefault('tasks', {})[task_name] = task_data
                        else:
                            # Merge steps
                            merged['tasks']['tasks'][task_name]['steps'].extend(task_data.get('steps', []))
                            
                            # Update confidence
                            merged['tasks']['tasks'][task_name]['confidence'] = max(
                                merged['tasks']['tasks'][task_name].get('confidence', 0),
                                task_data.get('confidence', 0)
                            )
                    
                    # Update overall confidence
                    if 'confidence' in tasks_data:
                        merged['tasks']['confidence'] = max(
                            merged['tasks'].get('confidence', 0),
                            tasks_data.get('confidence', 0)
                        )
            
            # Add LLM-derived tasks
            if 'tasks' in llm_results:
                # Merge LLM-derived tasks with traditional tasks
                for task_name, task_data in llm_results['tasks'].get('tasks', {}).items():
                    if task_name not in merged['tasks'].get('tasks', {}):
                        merged['tasks'].setdefault('tasks', {})[task_name] = task_data
                    else:
                        # Merge steps
                        merged['tasks']['tasks'][task_name]['steps'].extend(task_data.get('steps', []))
                        
                        # Update confidence
                        merged['tasks']['tasks'][task_name]['confidence'] = max(
                            merged['tasks']['tasks'][task_name].get('confidence', 0),
                            task_data.get('confidence', 0)
                        )
                
                # Update overall confidence
                if 'confidence' in llm_results['tasks']:
                    merged['tasks']['confidence'] = max(
                        merged['tasks'].get('confidence', 0),
                        llm_results['tasks'].get('confidence', 0)
                    )
        
        return merged