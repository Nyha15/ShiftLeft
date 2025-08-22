# Original code from: https://github.com/PaulDanielML/MuJoCo_RL_UR5
# File: perception_module.py, vision_processing.py, camera_control.py

import mujoco
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List, Optional
import math

class VisionProcessor:
    """Vision processing module for UR5 robot"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Camera parameters
        self.camera_names = ["front_camera", "side_camera", "top_camera"]
        self.image_width = 640
        self.image_height = 480
        self.fov = 60.0  # Field of view in degrees
        
        # Vision processing parameters
        self.blob_detection_threshold = 0.1
        self.color_detection_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)]
        }
        
    def get_camera_image(self, camera_name: str) -> np.ndarray:
        """Get image from specified camera"""
        
        if camera_name not in self.camera_names:
            raise ValueError(f"Camera {camera_name} not found")
        
        # Get camera position and orientation
        camera_id = self.model.body(camera_name).id
        camera_pos = self.data.xpos[camera_id]
        camera_quat = self.data.xquat[camera_id]
        
        # Simulate camera image (in real implementation, this would use MuJoCo's camera rendering)
        # For now, create a synthetic image based on scene geometry
        image = self._generate_synthetic_image(camera_pos, camera_quat)
        
        return image
    
    def _generate_synthetic_image(self, camera_pos: np.ndarray, camera_quat: np.ndarray) -> np.ndarray:
        """Generate synthetic camera image based on scene geometry"""
        
        # Create base image
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Add some geometric shapes to simulate objects
        # This is a simplified version - real implementation would use MuJoCo's rendering
        
        # Add a red circle (simulating a target object)
        center_x, center_y = self.image_width // 2, self.image_height // 2
        cv2.circle(image, (center_x, center_y), 50, (0, 0, 255), -1)
        
        # Add a green rectangle (simulating another object)
        cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)
        
        # Add some noise to make it more realistic
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the image"""
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        detected_objects = []
        
        for color_name, (lower, upper) in self.color_detection_ranges.items():
            # Create mask for color
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Get center point
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    detected_objects.append({
                        'color': color_name,
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(area / 1000.0, 1.0)  # Normalize confidence
                    })
        
        return detected_objects
    
    def estimate_object_depth(self, image: np.ndarray, object_bbox: Tuple[int, int, int, int]) -> float:
        """Estimate depth of object using stereo vision principles"""
        
        x, y, w, h = object_bbox
        
        # Simplified depth estimation based on object size
        # In real implementation, this would use stereo cameras or depth sensors
        
        # Assume larger objects are closer (inverse relationship)
        object_area = w * h
        image_area = self.image_width * self.image_height
        
        # Normalize area ratio
        area_ratio = object_area / image_area
        
        # Convert to depth (closer = smaller depth value)
        # This is a heuristic - real implementation would use proper depth estimation
        estimated_depth = 1.0 / (area_ratio + 0.1)
        
        return estimated_depth
    
    def pixel_to_world_coordinates(self, pixel_x: int, pixel_y: int, depth: float, 
                                 camera_pos: np.ndarray, camera_quat: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to world coordinates"""
        
        # Convert pixel coordinates to normalized device coordinates (-1 to 1)
        ndc_x = (2.0 * pixel_x / self.image_width) - 1.0
        ndc_y = 1.0 - (2.0 * pixel_y / self.image_height)
        
        # Convert to camera space coordinates
        fov_rad = math.radians(self.fov)
        camera_x = ndc_x * depth * math.tan(fov_rad / 2.0)
        camera_y = ndc_y * depth * math.tan(fov_rad / 2.0)
        camera_z = depth
        
        # Convert to world coordinates using camera transform
        # This is a simplified version - real implementation would use proper transformation matrices
        
        # For now, just add camera position
        world_coords = camera_pos + np.array([camera_x, camera_y, camera_z])
        
        return world_coords

class PerceptionModule:
    """Perception module combining vision and other sensors"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.vision_processor = VisionProcessor(model, data)
        
        # Sensor parameters
        self.proximity_sensor_range = 0.5
        self.force_sensor_threshold = 10.0
        
    def get_environment_perception(self, camera_name: str = "front_camera") -> Dict[str, Any]:
        """Get comprehensive environment perception"""
        
        # Get visual information
        image = self.vision_processor.get_camera_image(camera_name)
        detected_objects = self.vision_processor.detect_objects(image)
        
        # Get proximity sensor data
        proximity_data = self._get_proximity_sensor_data()
        
        # Get force sensor data
        force_data = self._get_force_sensor_data()
        
        # Get joint sensor data
        joint_data = self._get_joint_sensor_data()
        
        perception = {
            "visual": {
                "image": image,
                "detected_objects": detected_objects,
                "camera_name": camera_name
            },
            "proximity": proximity_data,
            "force": force_data,
            "joints": joint_data,
            "timestamp": self.data.time
        }
        
        return perception
    
    def _get_proximity_sensor_data(self) -> Dict[str, float]:
        """Get data from proximity sensors"""
        
        # Simulate proximity sensor readings
        # In real implementation, this would read actual sensor data
        
        proximity_data = {}
        
        # Check distance to various objects in the scene
        end_effector_pos = self._get_end_effector_position()
        
        # Check distance to table surface
        table_distance = end_effector_pos[2]  # Z coordinate
        proximity_data["table_distance"] = max(0.0, table_distance)
        
        # Check distance to walls (simplified)
        wall_distances = {
            "front_wall": abs(end_effector_pos[0] - 1.0),
            "back_wall": abs(end_effector_pos[0] + 1.0),
            "left_wall": abs(end_effector_pos[1] - 0.5),
            "right_wall": abs(end_effector_pos[1] + 0.5)
        }
        
        proximity_data.update(wall_distances)
        
        return proximity_data
    
    def _get_force_sensor_data(self) -> Dict[str, np.ndarray]:
        """Get data from force sensors"""
        
        # Simulate force sensor readings
        # In real implementation, this would read actual sensor data
        
        force_data = {}
        
        # Get contact forces
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get contact force
            force = np.array([contact.efc_force[0], contact.efc_force[1], contact.efc_force[2]])
            
            if np.linalg.norm(force) > self.force_sensor_threshold:
                force_data[f"contact_{i}"] = {
                    "force": force,
                    "position": contact.pos,
                    "normal": contact.normal
                }
        
        return force_data
    
    def _get_joint_sensor_data(self) -> Dict[str, np.ndarray]:
        """Get data from joint sensors"""
        
        joint_data = {}
        
        # Get joint positions and velocities
        joint_positions = self.data.qpos[:6].copy()  # First 6 joints for UR5
        joint_velocities = self.data.qvel[:6].copy()
        joint_forces = self.data.qfrc_applied[:6].copy()
        
        joint_data["positions"] = joint_positions
        joint_data["velocities"] = joint_velocities
        joint_data["forces"] = joint_forces
        
        return joint_data
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Get end effector position"""
        return self.data.xpos[self.model.body("wrist_3_link").id].copy()
    
    def detect_collisions(self) -> List[Dict[str, Any]]:
        """Detect collisions in the environment"""
        
        collisions = []
        
        # Check contact forces for collision detection
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force_magnitude = np.linalg.norm(contact.efc_force[:3])
            
            if force_magnitude > self.force_sensor_threshold:
                collision_info = {
                    "contact_id": i,
                    "force_magnitude": force_magnitude,
                    "position": contact.pos.copy(),
                    "normal": contact.normal.copy(),
                    "penetration_depth": contact.dist
                }
                collisions.append(collision_info)
        
        return collisions
    
    def estimate_object_pose(self, detected_object: Dict[str, Any], 
                           camera_name: str = "front_camera") -> Dict[str, Any]:
        """Estimate 3D pose of detected object"""
        
        # Get camera parameters
        camera_id = self.model.body(camera_name).id
        camera_pos = self.data.xpos[camera_id]
        camera_quat = self.data.xquat[camera_id]
        
        # Get object center in pixels
        center_x, center_y = detected_object["center"]
        
        # Estimate depth
        depth = self.vision_processor.estimate_object_depth(
            np.zeros((self.image_height, self.image_width, 3)), 
            detected_object["bbox"]
        )
        
        # Convert to world coordinates
        world_pos = self.vision_processor.pixel_to_world_coordinates(
            center_x, center_y, depth, camera_pos, camera_quat
        )
        
        # Estimate orientation (simplified)
        # In real implementation, this would use more sophisticated computer vision techniques
        orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        pose_estimate = {
            "position": world_pos,
            "orientation": orientation,
            "confidence": detected_object["confidence"],
            "object_type": detected_object["color"],
            "bbox_2d": detected_object["bbox"]
        }
        
        return pose_estimate

class VisualServoingController:
    """Visual servoing controller for UR5 robot"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.perception_module = PerceptionModule(model, data)
        
        # Control parameters
        self.position_gain = 50.0
        self.orientation_gain = 30.0
        self.max_velocity = 1.0
        
    def compute_visual_servoing_action(self, target_object: Dict[str, Any], 
                                     current_image: np.ndarray) -> np.ndarray:
        """Compute action for visual servoing to target object"""
        
        # Get current perception
        perception = self.perception_module.get_environment_perception()
        
        # Find target object in current image
        detected_objects = perception["visual"]["detected_objects"]
        target_detected = None
        
        for obj in detected_objects:
            if obj["color"] == target_object["color"]:
                target_detected = obj
                break
        
        if target_detected is None:
            # Target not visible, return zero action
            return np.zeros(6)
        
        # Estimate current pose
        current_pose = self.perception_module.estimate_object_pose(target_detected)
        
        # Get target pose
        target_pose = target_object["pose"]
        
        # Compute error
        position_error = target_pose["position"] - current_pose["position"]
        orientation_error = target_pose["orientation"] - current_pose["orientation"]
        
        # Compute control action
        position_action = self.position_gain * position_error
        orientation_action = self.orientation_gain * orientation_error
        
        # Combine actions
        action = np.concatenate([position_action, orientation_action])
        
        # Limit velocity
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > self.max_velocity:
            action = action * (self.max_velocity / action_magnitude)
        
        return action
    
    def execute_visual_servoing(self, target_object: Dict[str, Any], 
                               max_iterations: int = 100) -> bool:
        """Execute visual servoing to target object"""
        
        for iteration in range(max_iterations):
            # Get current image
            current_image = self.perception_module.get_environment_perception()["visual"]["image"]
            
            # Compute action
            action = self.compute_visual_servoing_action(target_object, current_image)
            
            # Check if target reached
            if np.linalg.norm(action) < 0.01:
                return True
            
            # Apply action (simplified - would need proper robot control)
            # For now, just update end effector position
            end_effector_pos = self.perception_module._get_end_effector_position()
            new_pos = end_effector_pos + action[:3] * 0.01  # Small step
            
            # Update robot state (simplified)
            # In real implementation, this would use proper inverse kinematics
            
        return False

if __name__ == "__main__":
    # Example usage
    print("Vision and Perception Module for UR5 Robot")
    print("This module provides:")
    print("- Object detection and tracking")
    print("- Depth estimation")
    print("- Visual servoing control")
    print("- Multi-sensor perception")
    print("- Collision detection")
    
    # Note: This code requires a proper MuJoCo model with cameras and sensors
    # to run the actual functionality 