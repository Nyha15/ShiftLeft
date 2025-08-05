# Robotics Repository Analysis Summary

**Repository:** `/home/abhishek/llm_robot/test_repo`  
**Analysis Date:** 2025-08-05T16:18:18.495167  
**Overall Confidence:** 0.78  
**Analyzer Version:** 1.0.0

## 📊 Executive Summary

This repository contains **7 robotics tasks** and **9 robot specifications** with an average task complexity of **medium**.

### Key Findings
- **Total Robots:** 9
- **Total Tasks:** 7
- **Configuration Files:** 11
- **Total DOF:** 58
- **Average Task Complexity:** medium

## 🤖 Robot Specifications

### world(box)
- **DOF:** 0
- **Base Link:** box1
- **End Effector:** box2
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/world(box).xml`
- **Joints:** 2
- **Links:** 2

### cable
- **DOF:** 1
- **Base Link:** CB0
- **End Effector:** plane
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/cable.xml`
- **Joints:** 1
- **Links:** 2

### panda1
- **DOF:** 14
- **Base Link:** link0_1
- **End Effector:** right_finger1
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/panda1.xml`
- **Joints:** 14
- **Links:** 11

### panda_nohand
- **DOF:** 9
- **Base Link:** link0
- **End Effector:** attachment
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/panda_nohand.xml`
- **Joints:** 9
- **Links:** 9

### world(cable)
- **DOF:** 1
- **Base Link:** CB0
- **End Effector:** CB0
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/world(cable).xml`
- **Joints:** 2
- **Links:** 1

### hand
- **DOF:** 7
- **Base Link:** hand
- **End Effector:** hand
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/hand.xml`
- **Joints:** 7
- **Links:** 3

### mjx_panda
- **DOF:** 12
- **Base Link:** link0
- **End Effector:** hand
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/mjx_panda.xml`
- **Joints:** 12
- **Links:** 11

### mjx_single_cube
- **DOF:** 0
- **Base Link:** box
- **End Effector:** mocap_target
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/mjx_single_cube.xml`
- **Joints:** 0
- **Links:** 2

### panda2
- **DOF:** 14
- **Base Link:** link0_2
- **End Effector:** right_finger2
- **URDF Path:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/panda2.xml`
- **Joints:** 14
- **Links:** 11

## 🎯 Task Analysis

### Task Breakdown by Type

- **Manipulation:** 3 tasks
- **Navigation:** 1 tasks
- **Grasping:** 1 tasks
- **Control:** 2 tasks

### Individual Tasks

#### move_to
- **Type:** navigation
- **Complexity:** medium
- **Confidence:** 0.88
- **Duration:** 12.9s
- **Actions:** move, verify, compute, wait
- **Source:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/panda_mujoco.py`
- **Description:** Move the robot's end-effector to a desired position and orientation.
        Args:
            robot (str): Identifier for the robot (e.g., 'panda1' or 'panda2').
            y_pos_d (ndarray): Desired position of the end-effector (3x1 array).
            y_quat_d (ndarray): Desired orientation quaternion of the end-effector (4x1 array).
            length (float): Duration of the movement in seconds.

#### move_sim_rob
- **Type:** control
- **Complexity:** medium
- **Confidence:** 0.88
- **Duration:** 10.3s
- **Actions:** wait, verify, compute, rotate
- **Source:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/utils/utils.py`
- **Description:** Simulates robot motion to a desired position and orientation in the background simulation.
        Args:
            robot (str): The robot identifier (e.g., 'panda1' or 'panda2').
            y_pos_d (np.array): Desired position.
            y_quat_d (np.array): Desired orientation (quaternion).
            length (float): Duration of the motion.
        Returns:
            bool: True if the motion was successfully completed, False otherwise.

#### step
- **Type:** manipulation
- **Complexity:** complex
- **Confidence:** 0.62
- **Duration:** 24.8s
- **Actions:** actuate, move, rotate
- **Source:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/rrt_pick_place.py`
- **Description:** Performs a single operational step for both robots ('panda1' and 'panda2').
        The function includes logic for path planning, object manipulation, and returning to a home position.

#### gripper
- **Type:** manipulation
- **Complexity:** medium
- **Confidence:** 0.50
- **Duration:** 10.5s
- **Actions:** actuate, wait
- **Source:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/panda_mujoco.py`
- **Description:** Control the gripper of the robot.
        Args:
            robot (str): Identifier for the robot (e.g., 'panda1' or 'panda2').
            open (bool): If True, opens the gripper; otherwise, closes it.

#### get_robot_points
- **Type:** control
- **Complexity:** medium
- **Confidence:** 0.50
- **Duration:** 14.7s
- **Actions:** verify, rotate
- **Source:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/utils/utils.py`
- **Description:** Retrieves interpolated points along the robot's structure for collision checking.
        Args:
            robot_name (str): Name of the robot (e.g., 'panda1').
        Returns:
            list: A list of 3D points representing the robot's structure.

#### hold
- **Type:** grasping
- **Complexity:** medium
- **Confidence:** 0.38
- **Duration:** 11.6s
- **Actions:** verify, wait
- **Source:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/panda_mujoco.py`
- **Description:** Hold the robot in its current position for a specified duration.
        Args:
            robot (str): Identifier for the robot (e.g., 'panda1' or 'panda2').
            length (float): Duration to hold the position in seconds.

#### render
- **Type:** manipulation
- **Complexity:** medium
- **Confidence:** 0.38
- **Duration:** 18.3s
- **Actions:** actuate, wait, rotate
- **Source:** `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/panda_mujoco.py`
- **Description:** Render the simulation using GLFW and allow user interaction via mouse.

## 📁 Configuration Files

11 configuration files detected:

- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/world(box).xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/cable.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/panda1.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/panda_nohand.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/world(cable).xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/hand.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/mjx_panda.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/scene.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/mjx_single_cube.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/mjx_scene.xml`
- `/home/abhishek/llm_robot/test_repo/Robot-Pick-and-Place-Simulation/simulation_env/panda2.xml`

## 🔍 Analysis Details

### Methodology
This analysis used advanced AST parsing for Python code and URDF/XML parsing for robot specifications. Tasks were filtered for robotics relevance and scored for confidence.

### Confidence Scoring
- **Robot Detection:** Based on URDF parsing success and completeness
- **Task Detection:** Based on robotics keyword matching and code complexity
- **Overall Confidence:** Weighted average favoring robot specifications

### Limitations
- Analysis limited to Python files and URDF/XML configurations
- Task detection relies on naming patterns and keyword matching
- Complex task relationships may not be fully captured

---

*Generated by Robotics Repository Analyzer MVP v1.0.0*
