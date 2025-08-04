"""
Action Sequence Detector

This module detects action sequences in a robotics repository.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
import re

from task_definition.robotics_repo_analyzer.analyzers.code_analyzer import (
    extract_action_sequences_from_python,
    extract_action_sequences_from_notebook
)
from task_definition.robotics_repo_analyzer.utils.confidence import calculate_confidence

logger = logging.getLogger(__name__)

class ActionSequenceDetector:
    """
    Detector for action sequences in a robotics repository.
    """
    
    def __init__(self):
        """Initialize the action sequence detector."""
        pass
        
    def detect(self, files_by_type: Dict[str, List[Path]]) -> Dict[str, Any]:
        """
        Detect action sequences in the repository.
        
        Args:
            files_by_type: Dictionary mapping file types to lists of file paths
            
        Returns:
            Dictionary containing detected action sequences
        """
        logger.info("Detecting action sequences...")
        
        # Extract action sequences from Python files
        python_sequences = self._extract_from_python(files_by_type['python'])
        
        # Extract action sequences from notebooks
        notebook_sequences = self._extract_from_notebooks(files_by_type['notebook'])
        
        # Combine all sequences
        all_sequences = python_sequences + notebook_sequences
        
        # Sort by confidence
        sorted_sequences = sorted(
            all_sequences,
            key=lambda seq: seq.get('confidence', 0),
            reverse=True
        )
        
        # Calculate overall confidence
        overall_confidence = 0.0
        if sorted_sequences:
            # Average of top 3 sequences, weighted by their confidence
            top_sequences = sorted_sequences[:min(3, len(sorted_sequences))]
            weights = [seq.get('confidence', 0) for seq in top_sequences]
            if sum(weights) > 0:
                overall_confidence = sum(w * i / sum(weights) 
                                        for i, w in enumerate(reversed(weights), 1)) / len(weights)
        
        logger.info(f"Action sequence detection complete. "
                   f"Found {len(sorted_sequences)} sequences. "
                   f"Overall confidence: {overall_confidence:.2f}")
        
        return {
            'sequences': sorted_sequences,
            'confidence': overall_confidence
        }
    
    def _extract_from_python(self, python_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Extract action sequences from Python files.
        
        Args:
            python_files: List of Python file paths
            
        Returns:
            List of dictionaries containing action sequences
        """
        all_sequences = []
        
        # First, analyze entry points to prioritize them
        entry_point_scores = {}
        for file_path in python_files:
            try:
                # Simple heuristic to identify potential entry points
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                score = 0.0
                if '__main__' in content:
                    score += 0.3
                if 'def main' in content:
                    score += 0.2
                if any(framework in content for framework in 
                       ['mujoco', 'pybullet', 'gym', 'ros', 'moveit']):
                    score += 0.3
                
                entry_point_scores[file_path] = score
            except Exception as e:
                logger.warning(f"Error analyzing {file_path} for entry point score: {e}")
        
        # Sort files by entry point score
        sorted_files = sorted(
            python_files,
            key=lambda f: entry_point_scores.get(f, 0),
            reverse=True
        )
        
        # Analyze top files first, then others
        for file_path in sorted_files:
            try:
                sequences = extract_action_sequences_from_python(file_path)
                if sequences:
                    # Group sequences by task type
                    task_groups = self._group_sequences_by_task(sequences, file_path)
                    
                    # Add each task group as a separate sequence
                    for task_name, task_data in task_groups.items():
                        task_data['source'] = str(file_path)
                        task_data['type'] = 'python'
                        all_sequences.append(task_data)
            except Exception as e:
                logger.warning(f"Error extracting action sequences from {file_path}: {e}")
        
        return all_sequences
        
    def _group_sequences_by_task(self, sequences: List[Dict[str, Any]], file_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Group sequences by task type.
        
        Args:
            sequences: List of sequences to group
            file_path: Path to the file containing the sequences
            
        Returns:
            Dictionary mapping task names to task data
        """
        task_groups = {}
        
        # Define common robotics tasks and their related keywords
        task_keywords = {
            'pick_and_place': ['pick', 'place', 'grasp', 'release', 'pick_and_place', 'pick_place'],
            'navigation': ['navigate', 'move_to', 'goto', 'path', 'trajectory', 'navigation'],
            'manipulation': ['manipulate', 'push', 'pull', 'slide', 'rotate', 'manipulation'],
            'perception': ['perceive', 'detect', 'recognize', 'track', 'perception'],
            'grasping': ['grasp', 'grip', 'hold', 'grasping'],
            'pushing': ['push', 'pushing'],
            'sliding': ['slide', 'sliding']
        }
        
        # First pass: assign sequences to tasks based on name
        for sequence in sequences:
            name = sequence.get('name', '').lower()
            assigned = False
            
            # Check if the sequence name matches any task keywords
            for task, keywords in task_keywords.items():
                if any(keyword in name for keyword in keywords):
                    if task not in task_groups:
                        task_groups[task] = {
                            'name': task,
                            'confidence': sequence.get('confidence', 0.5),
                            'steps': [],
                            'source_line': sequence.get('source_line')
                        }
                    
                    # Add steps from this sequence
                    task_groups[task]['steps'].extend(sequence.get('steps', []))
                    
                    # Update confidence if this sequence has higher confidence
                    if sequence.get('confidence', 0) > task_groups[task].get('confidence', 0):
                        task_groups[task]['confidence'] = sequence.get('confidence')
                    
                    assigned = True
                    break
            
            # If not assigned to any task, check the steps
            if not assigned:
                steps = sequence.get('steps', [])
                step_actions = [step.get('action', '').lower() for step in steps]
                
                # Count occurrences of task-related actions
                task_counts = {}
                for task, keywords in task_keywords.items():
                    count = sum(1 for action in step_actions if any(keyword in action for keyword in keywords))
                    if count > 0:
                        task_counts[task] = count
                
                # Assign to the task with the most matches
                if task_counts:
                    best_task = max(task_counts.items(), key=lambda x: x[1])[0]
                    
                    if best_task not in task_groups:
                        task_groups[best_task] = {
                            'name': best_task,
                            'confidence': sequence.get('confidence', 0.5),
                            'steps': [],
                            'source_line': sequence.get('source_line')
                        }
                    
                    # Add steps from this sequence
                    task_groups[best_task]['steps'].extend(sequence.get('steps', []))
                    
                    # Update confidence if this sequence has higher confidence
                    if sequence.get('confidence', 0) > task_groups[best_task].get('confidence', 0):
                        task_groups[best_task]['confidence'] = sequence.get('confidence')
                    
                    assigned = True
            
            # If still not assigned, create a new task with the sequence name
            if not assigned:
                task_name = name if name else f"task_{len(task_groups)}"
                
                if task_name not in task_groups:
                    task_groups[task_name] = {
                        'name': task_name,
                        'confidence': sequence.get('confidence', 0.5),
                        'steps': [],
                        'source_line': sequence.get('source_line')
                    }
                
                # Add steps from this sequence
                task_groups[task_name]['steps'].extend(sequence.get('steps', []))
        
        # Second pass: renumber steps in each task
        for task_name, task_data in task_groups.items():
            for i, step in enumerate(task_data['steps']):
                step['step'] = i + 1
        
        return task_groups
    
    def _extract_from_notebooks(self, notebook_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Extract action sequences from Jupyter notebooks.
        
        Args:
            notebook_files: List of notebook file paths
            
        Returns:
            List of dictionaries containing action sequences
        """
        all_sequences = []
        
        for file_path in notebook_files:
            try:
                sequences = extract_action_sequences_from_notebook(file_path)
                if sequences:
                    for sequence in sequences:
                        sequence['source'] = str(file_path)
                        sequence['type'] = 'notebook'
                        all_sequences.append(sequence)
            except Exception as e:
                logger.warning(f"Error extracting action sequences from notebook {file_path}: {e}")
        
        return all_sequences