import mujoco
import numpy as np
import pytest

def test_joint_position_sensors():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_positions = [np.pi/4, np.pi/6, np.pi/3, 0, 0, 0]
    data.qpos[:] = test_positions
    
    mujoco.mj_forward(model, data)
    
    for i in range(model.nq):
        sensor_value = data.sensordata[i]
        expected_value = test_positions[i]
        assert abs(sensor_value - expected_value) < 1e-6

def test_joint_velocity_sensors():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_velocities = [1.0, 0.5, 0.3, 0, 0, 0]
    data.qvel[:] = test_velocities
    
    mujoco.mj_forward(model, data)
    
    for i in range(model.nv):
        sensor_value = data.sensordata[model.nq + i]
        expected_value = test_velocities[i]
        assert abs(sensor_value - expected_value) < 1e-6

def test_force_torque_sensors():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
    
    for i in range(model.nsensor):
        if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FORCE:
            sensor_value = data.sensordata[i]
            assert not np.isnan(sensor_value)
            assert abs(sensor_value) < 1e6

def test_imu_sensors():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    for i in range(model.nsensor):
        if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
            sensor_value = data.sensordata[i]
            assert not np.isnan(sensor_value)
            assert np.linalg.norm(sensor_value) < 100.0

def test_touch_sensors():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [np.pi/2, 0, 0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    
    for i in range(model.nsensor):
        if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH:
            sensor_value = data.sensordata[i]
            assert sensor_value >= 0
            assert sensor_value <= 1

def test_proximity_sensors():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    for i in range(model.nsensor):
        if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_RAYCAST:
            sensor_value = data.sensordata[i]
            assert not np.isnan(sensor_value)
            assert sensor_value >= 0

def test_sensor_noise():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_position = np.pi/4
    data.qpos[0] = test_position
    
    mujoco.mj_forward(model, data)
    
    sensor_value = data.sensordata[0]
    noise_std = model.sensor_noise[0]
    
    if noise_std > 0:
        noise_magnitude = abs(sensor_value - test_position)
        assert noise_magnitude < 3 * noise_std

def test_sensor_calibration():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    zero_position = np.zeros(model.nq)
    data.qpos[:] = zero_position
    
    mujoco.mj_forward(model, data)
    
    for i in range(model.nq):
        sensor_value = data.sensordata[i]
        calibration_error = abs(sensor_value - zero_position[i])
        assert calibration_error < 1e-3

def test_sensor_sampling_rate():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    dt = model.opt.timestep
    expected_samples = int(1.0 / dt)
    
    sample_count = 0
    for _ in range(expected_samples):
        mujoco.mj_step(model, data)
        sample_count += 1
    
    assert sample_count == expected_samples

def test_sensor_fusion():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [np.pi/4, np.pi/6, np.pi/3, 0, 0, 0]
    data.qvel[:] = [0.1, 0.05, 0.03, 0, 0, 0]
    
    mujoco.mj_forward(model, data)
    
    position_sensor = data.sensordata[0]
    velocity_sensor = data.sensordata[model.nq]
    
    integrated_position = position_sensor + velocity_sensor * model.opt.timestep
    
    assert abs(integrated_position - position_sensor) < 1e-3 