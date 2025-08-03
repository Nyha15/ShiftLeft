"""
Task Organizer

This module organizes action sequences into meaningful tasks.
"""

import logging
from typing import Dict, List, Any
import re

logger = logging.getLogger(__name__)

class TaskOrganizer:
    """
    Organizes action sequences into meaningful tasks.
    """
    
    def __init__(self):
        """Initialize the task organizer."""
        # Define common robotics tasks and their related keywords
        self.task_keywords = {
            'pick_and_place': ['pick', 'place', 'grasp', 'release', 'pick_and_place', 'pick_place'],
            'navigation': ['navigate', 'move_to', 'goto', 'path', 'trajectory', 'navigation'],
            'manipulation': ['manipulate', 'push', 'pull', 'slide', 'rotate', 'manipulation'],
            'perception': ['perceive', 'detect', 'recognize', 'track', 'perception'],
            'grasping': ['grasp', 'grip', 'hold', 'grasping'],
            'pushing': ['push', 'pushing'],
            'sliding': ['slide', 'sliding']
        }
    
    def organize_tasks(self, action_sequences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize action sequences into meaningful tasks.
        
        Args:
            action_sequences: Dictionary containing action sequences
            
        Returns:
            Dictionary containing organized tasks
        """
        # Extract sequences
        sequences = action_sequences.get('sequences', [])
        
        # Group sequences by task type
        task_groups = self._group_sequences_by_task(sequences)
        
        # Calculate overall confidence
        overall_confidence = action_sequences.get('confidence', 0.0)
        
        return {
            'tasks': task_groups,
            'confidence': overall_confidence
        }
    
    def _group_sequences_by_task(self, sequences: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group sequences by task type.
        
        Args:
            sequences: List of sequences to group
            
        Returns:
            Dictionary mapping task names to task data
        """
        task_groups = {}
        
        # First pass: assign sequences to tasks based on name
        for sequence in sequences:
            name = sequence.get('name', '').lower()
            assigned = False
            
            # Check if the sequence name matches any task keywords
            for task, keywords in self.task_keywords.items():
                if any(keyword in name for keyword in keywords):
                    if task not in task_groups:
                        task_groups[task] = {
                            'name': task,
                            'confidence': sequence.get('confidence', 0.5),
                            'steps': [],
                            'source': sequence.get('source')
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
                for task, keywords in self.task_keywords.items():
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
                            'source': sequence.get('source')
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
                        'source': sequence.get('source')
                    }
                
                # Add steps from this sequence
                task_groups[task_name]['steps'].extend(sequence.get('steps', []))
        
        # Second pass: renumber steps in each task
        for task_name, task_data in task_groups.items():
            for i, step in enumerate(task_data['steps']):
                step['step'] = i + 1
        
        return task_groups