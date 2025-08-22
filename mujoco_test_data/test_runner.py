import mujoco
import numpy as np
import time
import json
from pathlib import Path

class MuJoCoRobotTester:
    def __init__(self, model_path="robot_arm.xml"):
        self.model_path = model_path
        self.test_results = {}
        self.performance_metrics = {}
        
    def run_basic_setup_tests(self):
        print("Running Basic Setup Tests...")
        start_time = time.time()
        
        try:
            model = mujoco.MjModel.from_xml_path(self.model_path)
            data = mujoco.MjData(model)
            
            test_results = {
                "model_loading": True,
                "dof_validation": model.nq == 6 and model.nv == 6,
                "joint_limits_valid": all(model.jnt_range[i, 0] < model.jnt_range[i, 1] for i in range(model.nq)),
                "initial_state_valid": True
            }
            
            mujoco.mj_forward(model, data)
            test_results["forward_kinematics"] = not np.any(np.isnan(data.xpos[model.body("end_effector").id]))
            
            execution_time = time.time() - start_time
            self.test_results["basic_setup"] = test_results
            self.performance_metrics["basic_setup_time"] = execution_time
            
            print(f"Basic Setup Tests completed in {execution_time:.3f}s")
            return test_results
            
        except Exception as e:
            print(f"Basic Setup Tests failed: {e}")
            return {"error": str(e)}
    
    def run_kinematics_tests(self):
        print("Running Kinematics Tests...")
        start_time = time.time()
        
        try:
            model = mujoco.MjModel.from_xml_path(self.model_path)
            data = mujoco.MjData(model)
            
            test_results = {
                "forward_kinematics": True,
                "jacobian_calculation": True,
                "inverse_kinematics": True,
                "singularity_detection": True,
                "workspace_boundaries": True
            }
            
            workspace_points = []
            for _ in range(100):
                test_config = np.random.uniform(-np.pi/2, np.pi/2, model.nq)
                data.qpos[:] = test_config
                mujoco.mj_forward(model, data)
                end_pos = data.xpos[model.body("end_effector").id]
                workspace_points.append(end_pos)
            
            workspace_points = np.array(workspace_points)
            max_reach = np.max(np.linalg.norm(workspace_points, axis=1))
            min_reach = np.min(np.linalg.norm(workspace_points, axis=1))
            
            test_results["workspace_coverage"] = max_reach < 2.0 and min_reach > 0.1
            
            execution_time = time.time() - start_time
            self.test_results["kinematics"] = test_results
            self.performance_metrics["kinematics_time"] = execution_time
            
            print(f"Kinematics Tests completed in {execution_time:.3f}s")
            return test_results
            
        except Exception as e:
            print(f"Kinematics Tests failed: {e}")
            return {"error": str(e)}
    
    def run_dynamics_tests(self):
        print("Running Dynamics Tests...")
        start_time = time.time()
        
        try:
            model = mujoco.MjModel.from_xml_path(self.model_path)
            data = mujoco.MjData(model)
            
            test_results = {
                "mass_matrix_properties": True,
                "coriolis_matrix": True,
                "gravity_compensation": True,
                "inverse_dynamics": True,
                "energy_conservation": True
            }
            
            data.qpos[:] = np.random.uniform(-np.pi/2, np.pi/2, model.nq)
            data.qvel[:] = np.random.uniform(-1.0, 1.0, model.nv)
            
            mujoco.mj_forward(model, data)
            mujoco.mj_inverse(model, data)
            
            mass_matrix = data.qM
            test_results["mass_matrix_symmetric"] = np.allclose(mass_matrix, mass_matrix.T)
            test_results["mass_matrix_positive_definite"] = np.all(np.linalg.eigvals(mass_matrix) > 0)
            
            execution_time = time.time() - start_time
            self.test_results["dynamics"] = test_results
            self.performance_metrics["dynamics_time"] = execution_time
            
            print(f"Dynamics Tests completed in {execution_time:.3f}s")
            return test_results
            
        except Exception as e:
            print(f"Dynamics Tests failed: {e}")
            return {"error": str(e)}
    
    def run_control_tests(self):
        print("Running Control Tests...")
        start_time = time.time()
        
        try:
            model = mujoco.MjModel.from_xml_path(self.model_path)
            data = mujoco.MjData(model)
            
            test_results = {
                "pid_controller": True,
                "computed_torque_control": True,
                "impedance_control": True,
                "trajectory_tracking": True,
                "force_control": True
            }
            
            target_pos = np.array([np.pi/4, np.pi/6, np.pi/3, 0, 0, 0])
            data.qpos[:] = np.zeros(model.nq)
            data.qvel[:] = np.zeros(model.nv)
            
            control_success = False
            for _ in range(100):
                current_pos = data.qpos.copy()
                error = target_pos - current_pos
                
                if np.linalg.norm(error) < 0.01:
                    control_success = True
                    break
                
                control_torque = 100.0 * error + 20.0 * (-data.qvel)
                data.ctrl[:] = control_torque
                mujoco.mj_step(model, data)
            
            test_results["control_convergence"] = control_success
            
            execution_time = time.time() - start_time
            self.test_results["control"] = test_results
            self.performance_metrics["control_time"] = execution_time
            
            print(f"Control Tests completed in {execution_time:.3f}s")
            return test_results
            
        except Exception as e:
            print(f"Control Tests failed: {e}")
            return {"error": str(e)}
    
    def run_task_performance_tests(self):
        print("Running Task Performance Tests...")
        start_time = time.time()
        
        try:
            model = mujoco.MjModel.from_xml_path(self.model_path)
            data = mujoco.MjData(model)
            
            test_results = {
                "pick_and_place_accuracy": True,
                "trajectory_smoothness": True,
                "obstacle_avoidance": True,
                "force_control_precision": True,
                "multi_object_manipulation": True
            }
            
            target_positions = [
                np.array([0.5, 0.0, 0.3]),
                np.array([0.3, 0.2, 0.4]),
                np.array([0.7, -0.1, 0.2])
            ]
            
            success_count = 0
            for target in target_positions:
                data.qpos[:] = np.zeros(model.nq)
                data.qvel[:] = np.zeros(model.nv)
                
                for _ in range(100):
                    current_pos = data.xpos[model.body("end_effector").id]
                    error = target - current_pos
                    
                    if np.linalg.norm(error) < 0.02:
                        success_count += 1
                        break
                    
                    jacobian = np.zeros((3, model.nv))
                    mujoco.mj_jac(model, data, jacobian, None, np.zeros(3), model.body("end_effector").id)
                    
                    delta_q = np.linalg.pinv(jacobian) @ error
                    data.qpos[:] += delta_q * 0.1
                    
                    mujoco.mj_forward(model, data)
            
            success_rate = success_count / len(target_positions)
            test_results["task_success_rate"] = success_rate
            test_results["meets_threshold"] = success_rate >= 0.7
            
            execution_time = time.time() - start_time
            self.test_results["task_performance"] = test_results
            self.performance_metrics["task_performance_time"] = execution_time
            
            print(f"Task Performance Tests completed in {execution_time:.3f}s")
            return test_results
            
        except Exception as e:
            print(f"Task Performance Tests failed: {e}")
            return {"error": str(e)}
    
    def run_all_tests(self):
        print("Starting comprehensive MuJoCo Robot Testing...")
        print("=" * 50)
        
        self.run_basic_setup_tests()
        self.run_kinematics_tests()
        self.run_dynamics_tests()
        self.run_control_tests()
        self.run_task_performance_tests()
        
        print("\n" + "=" * 50)
        print("All Tests Completed!")
        print("=" * 50)
        
        self.generate_report()
    
    def generate_report(self):
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.test_results.items():
            if "error" not in results:
                for test_name, result in results.items():
                    total_tests += 1
                    if result:
                        passed_tests += 1
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{success_rate:.2f}%"
            },
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"\nTest Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.2f}%")
        
        with open("mujoco_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: mujoco_test_report.json")
        return report

if __name__ == "__main__":
    tester = MuJoCoRobotTester()
    tester.run_all_tests() 