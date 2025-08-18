#!/usr/bin/env python3
"""
Test Script for Enhanced Robotics Repository Analyzer
====================================================

Demonstrates the enhanced system with a sample robotics repository.
"""

import tempfile
import shutil
from pathlib import Path
from enhanced_main_analyzer import EnhancedRoboticsAnalyzer

def create_sample_repository():
    """Create a sample robotics repository for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create sample Python files with robotics tasks
    sample_files = {
        "robot_controller.py": '''
class RobotController:
    """Controls robot arm movements and gripper operations."""
    
    def __init__(self, max_velocity=1.0, position_tolerance=0.01, gripper_force=50.0):
        self.max_velocity = max_velocity
        self.position_tolerance = position_tolerance
        self.gripper_force = gripper_force
    
    def move_to_target(self, target, velocity=None):
        """Move robot to target position."""
        if velocity is None:
            velocity = self.max_velocity
        
        success = self._execute_motion(target, velocity)
        accuracy = self._check_position_accuracy(target)
        
        return success and (accuracy < self.position_tolerance)
    
    def _execute_motion(self, target, velocity):
        """Execute motion to target."""
        return True
    
    def _check_position_accuracy(self, target):
        """Check position accuracy."""
        return 0.005  # 5mm accuracy
''',
        
        "pick_and_place.py": '''
def pick_and_place_object(target_position, gripper_force=50.0, approach_distance=0.1):
    """
    Pick up an object and place it at the target position.
    
    Args:
        target_position: Target placement position [x, y, z]
        gripper_force: Force applied by gripper (N)
        approach_distance: Distance to approach object (m)
    """
    success = grasp_object(gripper_force)
    if success:
        move_to_position(target_position)
        release_object()
        return True
    return False

def grasp_object(force):
    """Grasp object with specified force."""
    return True

def move_to_position(position):
    """Move to specified position."""
    pass

def release_object():
    """Release grasped object."""
    pass
''',
        
        "kinematics_solver.py": '''
import numpy as np

class KinematicsSolver:
    """Solves forward and inverse kinematics for robot arm."""
    
    def __init__(self, joint_limits=(-np.pi, np.pi), max_iterations=100):
        self.joint_limits = joint_limits
        self.max_iterations = max_iterations
    
    def solve_inverse_kinematics(self, target_pose, initial_guess=None):
        """
        Solve inverse kinematics for target pose.
        
        Args:
            target_pose: Target end-effector pose [x, y, z, roll, pitch, yaw]
            initial_guess: Initial joint configuration guess
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        success = self._iterative_solve(target_pose, initial_guess)
        final_config = self._get_current_config()
        
        return success, final_config
    
    def _iterative_solve(self, target, guess):
        """Iteratively solve IK."""
        return True
    
    def _get_current_config(self):
        """Get current joint configuration."""
        return np.zeros(6)
''',
        
        "vision_system.py": '''
def detect_objects(image, confidence_threshold=0.8, max_objects=10):
    """
    Detect objects in image using computer vision.
    
    Args:
        image: Input image array
        confidence_threshold: Minimum confidence for detection
        max_objects: Maximum number of objects to detect
    """
    detections = []
    
    # Simulate object detection
    for i in range(min(3, max_objects)):
        detection = {
            'class': f'object_{i}',
            'confidence': 0.9,
            'bbox': [100*i, 100*i, 200, 200]
        }
        if detection['confidence'] >= confidence_threshold:
            detections.append(detection)
    
    return detections

def segment_workspace(image, workspace_bounds):
    """Segment workspace from image."""
    return True
'''
    }
    
    # Create files
    for filename, content in sample_files.items():
        file_path = temp_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
    
    return temp_dir

def test_enhanced_system():
    """Test the enhanced system with sample repository"""
    print("Creating sample robotics repository...")
    sample_repo = create_sample_repository()
    
    try:
        print(f"Sample repository created at: {sample_repo}")
        
        # Initialize enhanced analyzer
        analyzer = EnhancedRoboticsAnalyzer()
        
        # Run analysis
        print("\nRunning enhanced analysis...")
        result = analyzer.analyze_repository(str(sample_repo))
        
        # Print results
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"Total Tasks: {result['total_tasks']}")
        print(f"Overall Confidence: {result['overall_confidence']:.2f}")
        print(f"Analysis Time: {result['analysis_time']:.2f}s")
        
        # Print task categories
        if result['summary']['task_categories']:
            print("\nTask Categories:")
            for category, count in result['summary']['task_categories'].items():
                print(f"  {category}: {count}")
        
        # Print confidence distribution
        confidence_dist = result['summary']['confidence_distribution']
        print(f"\nConfidence Distribution:")
        print(f"  High (≥0.7): {confidence_dist['high']}")
        print(f"  Medium (0.4-0.7): {confidence_dist['medium']}")
        print(f"  Low (<0.4): {confidence_dist['low']}")
        
        # Print generated files
        print(f"\nGenerated Files:")
        for file_path in result['generated_files']:
            print(f"  {file_path}")
        
        # Generate master YAML
        print("\nGenerating master YAML...")
        tasks = analyzer.task_extractor.extract_enhanced_tasks(sample_repo)
        output_dir = Path(result['generated_files'][0]).parent if result['generated_files'] else sample_repo / "task_analysis_output"
        master_file = analyzer.generate_master_yaml(tasks, output_dir)
        print(f"Master YAML generated: {master_file}")
        
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print("="*60)
        print(f"Check the output directory for generated YAMLs: {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\nCleaning up sample repository: {sample_repo}")
        shutil.rmtree(sample_repo)

if __name__ == "__main__":
    test_enhanced_system() 