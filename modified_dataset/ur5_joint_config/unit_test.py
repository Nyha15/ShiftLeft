# Unit tests for: training.py (UR5 joint configuration)
# Tests the set_ur5_joints.py functionality

import pytest
import numpy as np
import math
from training import *

def test_calculate_quaternion_from_angle():
    """Test quaternion calculation for different joint angles"""
    
    # Test zero angle
    quat_zero = calculate_quaternion_from_angle(0.0, 0)
    assert np.allclose(quat_zero, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    
    # Test 90 degree angle
    quat_90 = calculate_quaternion_from_angle(math.pi/2, 1)
    expected_90 = [math.cos(math.pi/4), 0, math.sin(math.pi/4), 0]
    assert np.allclose(quat_90, expected_90, atol=1e-6)
    
    # Test 180 degree angle
    quat_180 = calculate_quaternion_from_angle(math.pi, 2)
    expected_180 = [0.0, 0.0, 0.0, 1.0]
    assert np.allclose(quat_180, expected_180, atol=1e-6)

def test_get_ur5_home_configuration():
    """Test UR5 home configuration values"""
    
    home_config = get_ur5_home_configuration()
    
    assert len(home_config) == 6
    assert home_config[0] == -90.0  # base
    assert home_config[1] == -175.0  # shoulder
    assert home_config[2] == -5.0    # elbow
    assert home_config[3] == -180.0  # wrist1
    assert home_config[4] == -90.0   # wrist2
    assert home_config[5] == -180.0  # wrist3

def test_get_ur5_zero_configuration():
    """Test UR5 zero configuration values"""
    
    zero_config = get_ur5_zero_configuration()
    
    assert len(zero_config) == 6
    assert all(angle == 0.0 for angle in zero_config)

def test_degree_to_radian_conversion():
    """Test degree to radian conversion"""
    
    test_angles_degrees = [0, 90, 180, 270, 360]
    expected_radians = [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
    
    for deg, expected_rad in zip(test_angles_degrees, expected_radians):
        converted_rad = math.radians(deg)
        assert abs(converted_rad - expected_rad) < 1e-10

def test_quaternion_properties():
    """Test quaternion mathematical properties"""
    
    # Test quaternion normalization
    for joint_idx in range(6):
        angle = math.pi/4
        quat = calculate_quaternion_from_angle(angle, joint_idx)
        quat_magnitude = math.sqrt(sum(q*q for q in quat))
        assert abs(quat_magnitude - 1.0) < 1e-6

def test_joint_index_validation():
    """Test quaternion calculation for valid joint indices"""
    
    valid_indices = [0, 1, 2, 3, 4, 5]
    test_angle = math.pi/6
    
    for joint_idx in valid_indices:
        quat = calculate_quaternion_from_angle(test_angle, joint_idx)
        assert len(quat) == 4
        assert not any(np.isnan(q) for q in quat)

def test_angle_range_validation():
    """Test quaternion calculation for various angle ranges"""
    
    test_angles = [-2*math.pi, -math.pi, -math.pi/2, 0, math.pi/2, math.pi, 2*math.pi]
    joint_idx = 0
    
    for angle in test_angles:
        quat = calculate_quaternion_from_angle(angle, joint_idx)
        assert len(quat) == 4
        assert not any(np.isnan(q) for q in quat)

def test_base_quaternion_values():
    """Test base quaternion values for each joint"""
    
    expected_base_quats = {
        0: [0.681998, 0, 0, -0.731354],      # shoulder_link
        1: [0.707107, 0, 0.707107, 0],       # upper_arm_link
        2: [1, 0, 0, 0],                     # forearm_link
        3: [0.707107, 0, 0.707107, 0],       # wrist_1_link
        4: [0.5, 0.5, -0.5, 0.5],            # wrist_2_link
        5: [0.5, 0.5, -0.5, 0.5]             # wrist_3_link
    }
    
    for joint_idx, expected_quat in expected_base_quats.items():
        quat = calculate_quaternion_from_angle(0.0, joint_idx)
        assert np.allclose(quat, expected_quat, atol=1e-6)

def test_rotation_axis_consistency():
    """Test that rotation axes are consistent for each joint type"""
    
    # Y-axis rotations (joints 1 and 3)
    y_axis_joints = [1, 3]
    for joint_idx in y_axis_joints:
        angle = math.pi/4
        quat = calculate_quaternion_from_angle(angle, joint_idx)
        assert abs(quat[2]) > 0  # Y component should be non-zero
        assert abs(quat[3]) < 1e-6  # Z component should be zero
    
    # Z-axis rotations (joints 4 and 5)
    z_axis_joints = [4, 5]
    for joint_idx in z_axis_joints:
        angle = math.pi/4
        quat = calculate_quaternion_from_angle(angle, joint_idx)
        assert abs(quat[3]) > 0  # Z component should be non-zero
        assert abs(quat[2]) < 1e-6  # Y component should be zero

def test_quaternion_continuity():
    """Test that quaternions change continuously with angle changes"""
    
    joint_idx = 0
    angles = np.linspace(0, math.pi, 10)
    quats = []
    
    for angle in angles:
        quat = calculate_quaternion_from_angle(angle, joint_idx)
        quats.append(quat)
    
    # Check that quaternions change smoothly
    for i in range(1, len(quats)):
        quat_diff = np.linalg.norm(np.array(quats[i]) - np.array(quats[i-1]))
        assert quat_diff < 0.5  # Should not have large jumps 