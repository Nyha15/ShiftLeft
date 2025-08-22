# Original code from: https://github.com/roboticsleeds/mujoco-ur5-model
# File: scripts/set_ur5_joints.py

import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import math

def set_ur5_joints(xml_file_path, joint_values_degrees):
    """
    Set UR5 joint values in degrees and convert to MuJoCo quaternions
    
    Args:
        xml_file_path: Path to the MuJoCo XML file
        joint_values_degrees: List of 6 joint values in degrees
    """
    
    # Convert degrees to radians
    joint_values_rad = [math.radians(angle) for angle in joint_values_degrees]
    
    # Parse XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find UR5 body elements
    ur5_bodies = ["shoulder_link", "upper_arm_link", "forearm_link", 
                   "wrist_1_link", "wrist_2_link", "wrist_3_link"]
    
    # Update joint positions based on UR5 configuration
    for i, body_name in enumerate(ur5_bodies):
        body_elem = root.find(f".//body[@name='{body_name}']")
        if body_elem is not None:
            # Calculate new quaternion based on joint angle
            angle = joint_values_rad[i]
            quat = calculate_quaternion_from_angle(angle, i)
            
            # Update quaternion attribute
            quat_elem = body_elem.find("quat")
            if quat_elem is not None:
                quat_elem.text = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
    
    # Save modified XML
    tree.write(xml_file_path)
    print(f"Updated {xml_file_path} with UR5 joint values: {joint_values_degrees}")

def calculate_quaternion_from_angle(angle, joint_index):
    """Calculate quaternion for a given joint angle and joint index"""
    
    # Base quaternions for each joint
    base_quats = {
        0: [0.681998, 0, 0, -0.731354],      # shoulder_link
        1: [0.707107, 0, 0.707107, 0],       # upper_arm_link
        2: [1, 0, 0, 0],                     # forearm_link
        3: [0.707107, 0, 0.707107, 0],       # wrist_1_link
        4: [0.5, 0.5, -0.5, 0.5],            # wrist_2_link
        5: [0.5, 0.5, -0.5, 0.5]             # wrist_3_link
    }
    
    base_quat = base_quats[joint_index]
    
    # Create rotation quaternion for the angle
    half_angle = angle / 2.0
    rotation_quat = [math.cos(half_angle), 0, 0, math.sin(half_angle)]
    
    # Multiply quaternions (simplified)
    if joint_index in [1, 3]:  # Y-axis rotation
        rotation_quat = [math.cos(half_angle), 0, math.sin(half_angle), 0]
    elif joint_index in [4, 5]:  # Z-axis rotation
        rotation_quat = [math.cos(half_angle), 0, 0, math.sin(half_angle)]
    
    return rotation_quat

def load_ur5_model(xml_file_path):
    """Load UR5 model from XML file"""
    try:
        model = mujoco.MjModel.from_xml_path(xml_file_path)
        data = mujoco.MjData(model)
        return model, data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def test_ur5_configuration(xml_file_path, joint_values_degrees):
    """Test UR5 configuration with given joint values"""
    
    model, data = load_ur5_model(xml_file_path)
    if model is None:
        return False
    
    # Set joint values
    joint_values_rad = [math.radians(angle) for angle in joint_values_degrees]
    data.qpos[:] = joint_values_rad
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get end effector position
    end_effector_pos = data.xpos[model.body("wrist_3_link").id]
    
    print(f"End effector position: {end_effector_pos}")
    return True

def get_ur5_home_configuration():
    """Get UR5 home configuration"""
    return [-90.0, -175.0, -5.0, -180.0, -90.0, -180.0]

def get_ur5_zero_configuration():
    """Get UR5 zero configuration"""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

if __name__ == "__main__":
    # Example usage
    xml_file = "ur5_ridgeback.xml"
    home_config = get_ur5_home_configuration()
    
    print("Setting UR5 home configuration...")
    set_ur5_joints(xml_file, home_config)
    
    print("Testing configuration...")
    test_ur5_configuration(xml_file, home_config) 