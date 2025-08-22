import mujoco
import numpy as np
import pytest

def test_mass_matrix_properties():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
    
    mass_matrix = data.qM
    assert mass_matrix.shape == (model.nv, model.nv)
    assert np.all(np.linalg.eigvals(mass_matrix) > 0)
    assert np.allclose(mass_matrix, mass_matrix.T)

def test_coriolis_matrix():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = np.random.uniform(-np.pi/2, np.pi/2, model.nq)
    data.qvel[:] = np.random.uniform(-1.0, 1.0, model.nv)
    
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
    
    coriolis = data.qfrc_bias
    assert coriolis.shape == (model.nv,)
    assert not np.any(np.isnan(coriolis))

def test_gravity_compensation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = np.zeros(model.nq)
    data.qvel[:] = np.zeros(model.nv)
    data.qacc[:] = np.zeros(model.nv)
    
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
    
    gravity_torques = data.qfrc_bias
    assert gravity_torques.shape == (model.nv,)
    
    for i in range(model.nv):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            assert abs(gravity_torques[i]) < 100.0

def test_inverse_dynamics():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    test_qpos = np.random.uniform(-np.pi/2, np.pi/2, model.nq)
    test_qvel = np.random.uniform(-1.0, 1.0, model.nv)
    test_qacc = np.random.uniform(-2.0, 2.0, model.nv)
    
    data.qpos[:] = test_qpos
    data.qvel[:] = test_qvel
    data.qacc[:] = test_qacc
    
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
    
    required_torques = data.qfrc_inverse
    assert required_torques.shape == (model.nv,)
    assert not np.any(np.isnan(required_torques))

def test_energy_conservation():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    initial_energy = 0.0
    
    for t in range(100):
        data.qpos[:] = np.sin(t * 0.1) * np.ones(model.nq) * 0.1
        data.qvel[:] = np.cos(t * 0.1) * np.ones(model.nv) * 0.1
        
        mujoco.mj_forward(model, data)
        mujoco.mj_inverse(model, data)
        
        kinetic_energy = 0.5 * data.qvel @ data.qM @ data.qvel
        potential_energy = -data.qfrc_bias @ data.qpos
        
        total_energy = kinetic_energy + potential_energy
        
        if t == 0:
            initial_energy = total_energy
        else:
            energy_change = abs(total_energy - initial_energy)
            assert energy_change < 1e-3

def test_joint_limits_enforcement():
    model = mujoco.MjModel.from_xml_path("robot_arm.xml")
    data = mujoco.MjData(model)
    
    for i in range(model.nq):
        lower_limit = model.jnt_range[i, 0]
        upper_limit = model.jnt_range[i, 1]
        
        data.qpos[i] = lower_limit - 0.1
        mujoco.mj_forward(model, data)
        assert data.qpos[i] >= lower_limit
        
        data.qpos[i] = upper_limit + 0.1
        mujoco.mj_forward(model, data)
        assert data.qpos[i] <= upper_limit 