import mujoco
import numpy as np
import pytest

def test_mujoco_model_loading():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    assert model is not None
    assert data is not None
    assert model.nq > 0
    assert model.nv > 0

def test_robot_dof_validation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    
    assert model.nq == 6
    assert model.nv == 6
    assert model.nu == 6

def test_joint_limits():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    
    for i in range(model.nq):
        assert model.jnt_range[i, 0] < model.jnt_range[i, 1]

def test_initial_state():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    assert np.allclose(data.qpos, np.zeros(model.nq))
    assert np.allclose(data.qvel, np.zeros(model.nv))

def test_forward_kinematics():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    end_effector_pos = data.xpos[model.body("end_effector").id]
    assert end_effector_pos.shape == (3,)
    assert not np.any(np.isnan(end_effector_pos)) 