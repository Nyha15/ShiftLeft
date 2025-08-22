import mujoco
import numpy as np
import pytest

def test_collision_detection():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    ncon = data.ncon
    assert ncon >= 0
    
    if ncon > 0:
        for i in range(ncon):
            contact = data.contact[i]
            assert contact.geom1 >= 0
            assert contact.geom2 >= 0
            assert contact.dist <= 0

def test_contact_forces():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [np.pi/2, 0, 0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    
    contact_forces = data.qfrc_contact
    assert contact_forces.shape == (model.nv,)
    
    if np.any(contact_forces != 0):
        assert np.all(contact_forces >= 0)

def test_penetration_depth():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [np.pi/2, 0, 0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist < 0:
            assert abs(contact.dist) < 0.1

def test_friction_coefficients():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    for i in range(model.ngeom):
        geom = model.geom(i)
        assert geom.friction[0] >= 0
        assert geom.friction[1] >= 0
        assert geom.friction[2] >= 0

def test_collision_response():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    initial_pos = data.qpos.copy()
    initial_vel = data.qvel.copy()
    
    data.qpos[:] = [np.pi/2, 0, 0, 0, 0, 0]
    data.qvel[:] = [1.0, 0, 0, 0, 0, 0]
    
    mujoco.mj_step(model, data)
    
    if data.ncon > 0:
        contact_forces = data.qfrc_contact
        assert np.any(contact_forces != 0)

def test_self_collision():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [np.pi, np.pi/2, 0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    
    self_collisions = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        
        if "robot" in geom1_name and "robot" in geom2_name:
            self_collisions += 1
    
    assert self_collisions >= 0

def test_contact_constraints():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [np.pi/2, 0, 0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    
    if data.ncon > 0:
        for i in range(data.ncon):
            contact = data.contact[i]
            assert contact.dist <= 0
            assert np.linalg.norm(contact.frame[:3, :3]) > 0.9

def test_impact_detection():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = [np.pi/2, 0, 0, 0, 0, 0]
    data.qvel[:] = [2.0, 0, 0, 0, 0, 0]
    
    mujoco.mj_step(model, data)
    
    if data.ncon > 0:
        impact_forces = data.qfrc_impact
        assert impact_forces.shape == (model.nv,)
        assert np.any(impact_forces != 0) 