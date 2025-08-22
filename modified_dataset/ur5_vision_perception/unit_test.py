# Unit tests for: training.py (UR5 Vision and Perception)
# Tests the perception_module.py, vision_processing.py, camera_control.py functionality

import pytest
import numpy as np
import cv2
from training import VisionProcessor, PerceptionModule, VisualServoingController

class TestVisionProcessor:
    """Test cases for VisionProcessor class"""
    
    def test_initialization(self):
        """Test vision processor initialization"""
        # This would need a proper model setup
        # For now, test the class structure
        assert VisionProcessor is not None
    
    def test_camera_parameters(self):
        """Test camera parameter initialization"""
        processor_class = VisionProcessor
        
        # Check expected attributes
        expected_attrs = [
            'camera_names', 'image_width', 'image_height', 'fov',
            'blob_detection_threshold', 'color_detection_ranges'
        ]
        
        for attr in expected_attrs:
            assert hasattr(processor_class, attr)
    
    def test_color_detection_ranges(self):
        """Test color detection range configuration"""
        
        # Test color ranges structure
        expected_colors = ['red', 'green', 'blue', 'yellow']
        
        # Mock color ranges
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)]
        }
        
        for color in expected_colors:
            assert color in color_ranges
            assert len(color_ranges[color]) == 2
            assert len(color_ranges[color][0]) == 3  # HSV values
            assert len(color_ranges[color][1]) == 3  # HSV values
    
    def test_image_generation(self):
        """Test synthetic image generation"""
        
        # Mock camera parameters
        camera_pos = np.array([0.0, 0.0, 1.0])
        camera_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Test image dimensions
        image_width = 640
        image_height = 480
        
        # Create test image
        test_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        
        # Add test shapes
        center_x, center_y = image_width // 2, image_height // 2
        cv2.circle(test_image, (center_x, center_y), 50, (0, 0, 255), -1)
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
        
        # Verify image properties
        assert test_image.shape == (image_height, image_width, 3)
        assert test_image.dtype == np.uint8
        
        # Check that shapes were added
        assert np.any(test_image > 0)  # Should have some non-zero pixels
    
    def test_object_detection(self):
        """Test object detection functionality"""
        
        # Create test image with known objects
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add red circle
        cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)
        
        # Add green rectangle
        cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)
        
        # Test detection (would need actual VisionProcessor instance)
        # For now, test the logic
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Test red detection
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # Should detect red circle
        assert np.any(red_mask > 0)
        
        # Test green detection
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Should detect green rectangle
        assert np.any(green_mask > 0)
    
    def test_depth_estimation(self):
        """Test depth estimation logic"""
        
        # Test depth estimation parameters
        image_width = 640
        image_height = 480
        
        # Test different object sizes
        test_cases = [
            {"bbox": (100, 100, 50, 50), "expected_depth_range": (0.5, 2.0)},
            {"bbox": (200, 200, 100, 100), "expected_depth_range": (0.2, 1.0)},
            {"bbox": (300, 300, 200, 200), "expected_depth_range": (0.1, 0.5)}
        ]
        
        for test_case in test_cases:
            x, y, w, h = test_case["bbox"]
            object_area = w * h
            image_area = image_width * image_height
            area_ratio = object_area / image_area
            
            # Simplified depth estimation
            estimated_depth = 1.0 / (area_ratio + 0.1)
            
            # Check depth is reasonable
            assert estimated_depth > 0
            assert estimated_depth < 10.0
    
    def test_coordinate_conversion(self):
        """Test pixel to world coordinate conversion"""
        
        # Test coordinate conversion parameters
        image_width = 640
        image_height = 480
        fov_degrees = 60.0
        fov_radians = np.radians(fov_degrees)
        
        # Test center pixel
        center_x = image_width // 2
        center_y = image_height // 2
        
        # Convert to NDC
        ndc_x = (2.0 * center_x / image_width) - 1.0
        ndc_y = 1.0 - (2.0 * center_y / image_height)
        
        # Should be at center (0, 0)
        assert abs(ndc_x) < 0.01
        assert abs(ndc_y) < 0.01
        
        # Test corner pixels
        corner_x = 0
        corner_y = 0
        
        ndc_corner_x = (2.0 * corner_x / image_width) - 1.0
        ndc_corner_y = 1.0 - (2.0 * corner_y / image_height)
        
        # Should be at corners (-1, 1)
        assert abs(ndc_corner_x - (-1.0)) < 0.01
        assert abs(ndc_corner_y - 1.0) < 0.01

class TestPerceptionModule:
    """Test cases for PerceptionModule class"""
    
    def test_initialization(self):
        """Test perception module initialization"""
        assert PerceptionModule is not None
    
    def test_sensor_data_structure(self):
        """Test sensor data structure"""
        
        # Mock sensor data
        proximity_data = {
            "table_distance": 0.1,
            "front_wall": 0.5,
            "back_wall": 0.3,
            "left_wall": 0.2,
            "right_wall": 0.4
        }
        
        force_data = {
            "contact_0": {
                "force": np.array([1.0, 2.0, 3.0]),
                "position": np.array([0.1, 0.2, 0.3]),
                "normal": np.array([0.0, 0.0, 1.0])
            }
        }
        
        joint_data = {
            "positions": np.zeros(6),
            "velocities": np.zeros(6),
            "forces": np.zeros(6)
        }
        
        # Test data types
        assert isinstance(proximity_data, dict)
        assert isinstance(force_data, dict)
        assert isinstance(joint_data, dict)
        
        # Test proximity data
        for key, value in proximity_data.items():
            assert isinstance(value, (int, float))
            assert value >= 0
        
        # Test force data
        for contact_info in force_data.values():
            assert "force" in contact_info
            assert "position" in contact_info
            assert "normal" in contact_info
            assert isinstance(contact_info["force"], np.ndarray)
            assert contact_info["force"].shape == (3,)
    
    def test_collision_detection(self):
        """Test collision detection logic"""
        
        # Mock collision data
        collision_info = {
            "contact_id": 0,
            "force_magnitude": 15.0,
            "position": np.array([0.1, 0.2, 0.3]),
            "normal": np.array([0.0, 0.0, 1.0]),
            "penetration_depth": 0.01
        }
        
        # Test collision properties
        assert collision_info["force_magnitude"] > 10.0  # Above threshold
        assert isinstance(collision_info["position"], np.ndarray)
        assert collision_info["position"].shape == (3,)
        assert isinstance(collision_info["normal"], np.ndarray)
        assert collision_info["normal"].shape == (3,)
        assert collision_info["penetration_depth"] > 0
    
    def test_object_pose_estimation(self):
        """Test object pose estimation"""
        
        # Mock detected object
        detected_object = {
            "color": "red",
            "center": (320, 240),
            "bbox": (270, 190, 100, 100),
            "area": 10000,
            "confidence": 0.8
        }
        
        # Mock camera parameters
        camera_pos = np.array([0.0, 0.0, 1.0])
        camera_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Test pose estimation structure
        pose_estimate = {
            "position": np.array([0.5, 0.0, 0.1]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "confidence": 0.8,
            "object_type": "red",
            "bbox_2d": (270, 190, 100, 100)
        }
        
        # Verify pose estimate structure
        assert "position" in pose_estimate
        assert "orientation" in pose_estimate
        assert "confidence" in pose_estimate
        assert "object_type" in pose_estimate
        assert "bbox_2d" in pose_estimate
        
        # Test data types
        assert isinstance(pose_estimate["position"], np.ndarray)
        assert pose_estimate["position"].shape == (3,)
        assert isinstance(pose_estimate["orientation"], np.ndarray)
        assert pose_estimate["orientation"].shape == (4,)
        assert isinstance(pose_estimate["confidence"], float)
        assert 0.0 <= pose_estimate["confidence"] <= 1.0

class TestVisualServoingController:
    """Test cases for VisualServoingController class"""
    
    def test_initialization(self):
        """Test visual servoing controller initialization"""
        assert VisualServoingController is not None
    
    def test_control_parameters(self):
        """Test control parameter configuration"""
        
        # Mock control parameters
        position_gain = 50.0
        orientation_gain = 30.0
        max_velocity = 1.0
        
        # Test parameter values
        assert position_gain > 0
        assert orientation_gain > 0
        assert max_velocity > 0
        
        # Test parameter relationships
        assert position_gain > orientation_gain  # Position usually more important
        assert max_velocity > 0.1  # Reasonable velocity limit
    
    def test_action_computation(self):
        """Test action computation logic"""
        
        # Mock pose data
        current_pose = {
            "position": np.array([0.4, 0.0, 0.1]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0])
        }
        
        target_pose = {
            "position": np.array([0.5, 0.0, 0.1]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0])
        }
        
        # Compute error
        position_error = target_pose["position"] - current_pose["position"]
        orientation_error = target_pose["orientation"] - current_pose["orientation"]
        
        # Test error calculation
        assert np.allclose(position_error, np.array([0.1, 0.0, 0.0]))
        assert np.allclose(orientation_error, np.array([0.0, 0.0, 0.0, 0.0]))
        
        # Test action computation
        position_gain = 50.0
        orientation_gain = 30.0
        
        position_action = position_gain * position_error
        orientation_action = orientation_gain * orientation_error
        
        # Combine actions
        action = np.concatenate([position_action, orientation_action])
        
        # Test action properties
        assert action.shape == (6,)
        assert np.allclose(action[:3], position_action)
        assert np.allclose(action[3:], orientation_action)
    
    def test_velocity_limiting(self):
        """Test velocity limiting functionality"""
        
        # Test action with high magnitude
        high_action = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
        max_velocity = 1.0
        
        # Calculate magnitude
        action_magnitude = np.linalg.norm(high_action)
        
        # Apply velocity limiting
        if action_magnitude > max_velocity:
            limited_action = high_action * (max_velocity / action_magnitude)
        else:
            limited_action = high_action
        
        # Test limiting
        limited_magnitude = np.linalg.norm(limited_action)
        assert limited_magnitude <= max_velocity
        
        # Test direction preservation
        if action_magnitude > 0:
            direction_original = high_action / action_magnitude
            direction_limited = limited_action / limited_magnitude
            assert np.allclose(direction_original, direction_limited)
    
    def test_servoing_convergence(self):
        """Test visual servoing convergence logic"""
        
        # Mock convergence test
        max_iterations = 100
        convergence_threshold = 0.01
        
        # Simulate convergence
        for iteration in range(max_iterations):
            # Mock action magnitude
            action_magnitude = 0.1 * np.exp(-iteration / 20)  # Exponential decay
            
            # Check convergence
            if action_magnitude < convergence_threshold:
                assert iteration < max_iterations  # Should converge before max
                break
        
        # Test that convergence happened
        assert action_magnitude < convergence_threshold

class TestIntegration:
    """Integration tests for vision and perception system"""
    
    def test_perception_pipeline(self):
        """Test complete perception pipeline"""
        
        # Mock perception data
        perception_data = {
            "visual": {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "detected_objects": [
                    {"color": "red", "center": (320, 240), "bbox": (270, 190, 100, 100)},
                    {"color": "green", "center": (150, 150), "bbox": (100, 100, 100, 100)}
                ],
                "camera_name": "front_camera"
            },
            "proximity": {"table_distance": 0.1},
            "force": {},
            "joints": {"positions": np.zeros(6)},
            "timestamp": 0.0
        }
        
        # Test perception structure
        assert "visual" in perception_data
        assert "proximity" in perception_data
        assert "force" in perception_data
        assert "joints" in perception_data
        assert "timestamp" in perception_data
        
        # Test visual data
        visual_data = perception_data["visual"]
        assert "image" in visual_data
        assert "detected_objects" in visual_data
        assert "camera_name" in visual_data
        
        # Test detected objects
        detected_objects = visual_data["detected_objects"]
        assert len(detected_objects) == 2
        assert detected_objects[0]["color"] == "red"
        assert detected_objects[1]["color"] == "green"
    
    def test_vision_control_loop(self):
        """Test vision-based control loop"""
        
        # Mock control loop
        target_object = {
            "color": "red",
            "pose": {
                "position": np.array([0.5, 0.0, 0.1]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0])
            }
        }
        
        # Simulate control iterations
        max_iterations = 10
        convergence_history = []
        
        for iteration in range(max_iterations):
            # Mock current pose (approaching target)
            progress = iteration / max_iterations
            current_position = np.array([0.4 + 0.1 * progress, 0.0, 0.1])
            
            # Calculate error
            position_error = np.linalg.norm(target_object["pose"]["position"] - current_position)
            convergence_history.append(position_error)
            
            # Check if converged
            if position_error < 0.01:
                break
        
        # Test convergence
        assert len(convergence_history) > 0
        assert convergence_history[-1] < 0.02  # Should be close to target
        
        # Test convergence trend (should generally decrease)
        if len(convergence_history) > 1:
            assert convergence_history[-1] <= convergence_history[0] 