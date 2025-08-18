#!/usr/bin/env python3
"""
Simulation Runner
=================

Runs robotics simulations for parameter evaluation using various physics engines.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import json

from data_models import TaskParameters, SimulationResult

logger = logging.getLogger(__name__)

class SimulationRunner:
    """Runs robotics simulations for parameter evaluation"""
    
    def __init__(self, engine: str):
        self.engine = engine.lower()
        self.logger = logging.getLogger(f"{__name__}.SimulationRunner")
        
        # Initialize physics engine
        self.sim = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the specified physics engine"""
        try:
            if self.engine == 'mujoco':
                self._init_mujoco()
            elif self.engine == 'pybullet':
                self._init_pybullet()
            elif self.engine == 'gazebo':
                self._init_gazebo()
            else:
                self.logger.warning(f"Unknown engine {self.engine}, using mock simulation")
                self.sim = MockSimulation()
        
        except Exception as e:
            self.logger.warning(f"Failed to initialize {self.engine}: {e}")
            self.logger.info("Falling back to mock simulation")
            self.sim = MockSimulation()
    
    def _init_mujoco(self):
        """Initialize MuJoCo simulation"""
        try:
            import mujoco
            self.sim = MuJoCoSimulation()
            self.logger.info("MuJoCo simulation initialized")
        except ImportError:
            self.logger.warning("MuJoCo not available")
            raise
    
    def _init_pybullet(self):
        """Initialize PyBullet simulation"""
        try:
            import pybullet as p
            self.sim = PyBulletSimulation()
            self.logger.info("PyBullet simulation initialized")
        except ImportError:
            self.logger.warning("PyBullet not available")
            raise
    
    def _init_gazebo(self):
        """Initialize Gazebo simulation"""
        try:
            # Gazebo integration would require ROS 2 and gazebo_ros
            # For now, we'll use a mock implementation
            self.logger.warning("Gazebo integration not fully implemented, using mock")
            self.sim = MockSimulation()
        except Exception as e:
            self.logger.warning(f"Gazebo initialization failed: {e}")
            self.sim = MockSimulation()
    
    def run_simulation(self, parameters: Dict[str, float], 
                      task_config: Dict[str, Any], 
                      seed: int = None) -> SimulationResult:
        """Run a single simulation with given parameters"""
        if seed is not None:
            np.random.seed(seed)
        
        start_time = time.time()
        
        try:
            # Configure simulation with parameters
            self.sim.configure(parameters, task_config)
            
            # Run simulation
            success, metrics, failure_reason = self.sim.run()
            
            # Calculate run time
            run_time = time.time() - start_time
            
            result = SimulationResult(
                seed=seed or int(time.time()),
                parameters=parameters,
                success=success,
                metrics=metrics,
                failure_reason=failure_reason
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            
            return SimulationResult(
                seed=seed or int(time.time()),
                parameters=parameters,
                success=False,
                metrics={},
                failure_reason=str(e)
            )
    
    def run_parameter_sweep(self, parameters: Dict[str, TaskParameters], 
                           sweep_config: Dict[str, Any]) -> List[SimulationResult]:
        """Run multiple simulations for parameter sweep"""
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameters, sweep_config)
        
        self.logger.info(f"Running {len(param_combinations)} parameter combinations")
        
        for i, param_set in enumerate(param_combinations):
            self.logger.info(f"Running simulation {i+1}/{len(param_combinations)}")
            
            # Run simulation
            result = self.run_simulation(param_set, sweep_config, seed=i)
            results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0:
                success_rate = sum(1 for r in results if r.success) / len(results)
                self.logger.info(f"Progress: {i+1}/{len(param_combinations)}, "
                               f"Success rate: {success_rate:.2f}")
        
        return results
    
    def _generate_parameter_combinations(self, parameters: Dict[str, TaskParameters], 
                                       sweep_config: Dict[str, Any]) -> List[Dict[str, float]]:
        """Generate parameter combinations for sweep"""
        combinations = []
        
        # Get sweep method
        method = sweep_config.get('method', 'grid')
        max_combinations = sweep_config.get('max_combinations', 100)
        
        if method == 'grid':
            combinations = self._generate_grid_combinations(parameters, max_combinations)
        elif method == 'random':
            combinations = self._generate_random_combinations(parameters, max_combinations)
        elif method == 'latin_hypercube':
            combinations = self._generate_latin_hypercube_combinations(parameters, max_combinations)
        else:
            combinations = self._generate_random_combinations(parameters, max_combinations)
        
        return combinations
    
    def _generate_grid_combinations(self, parameters: Dict[str, TaskParameters], 
                                   max_combinations: int) -> List[Dict[str, float]]:
        """Generate grid-based parameter combinations"""
        combinations = []
        
        # Calculate grid size per parameter
        num_params = len(parameters)
        if num_params == 0:
            return combinations
        
        grid_size = int(max_combinations ** (1.0 / num_params))
        grid_size = max(2, min(grid_size, 5))  # Limit grid size
        
        # Generate grid points
        param_names = list(parameters.keys())
        param_ranges = [parameters[name].sweep.range for name in param_names]
        
        # Create grid
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(param_ranges) > 1:
                    for k in range(grid_size):
                        if len(param_ranges) > 2:
                            # 3+ parameters
                            point = []
                            for param_range in param_ranges:
                                idx = np.random.randint(0, grid_size)
                                value = param_range[0] + (param_range[1] - param_range[0]) * idx / (grid_size - 1)
                                point.append(value)
                            grid_points.append(point)
                        else:
                            # 2 parameters
                            point = [
                                param_ranges[0][0] + (param_ranges[0][1] - param_ranges[0][0]) * i / (grid_size - 1),
                                param_ranges[1][0] + (param_ranges[1][1] - param_ranges[1][0]) * j / (grid_size - 1)
                            ]
                            grid_points.append(point)
                else:
                    # Single parameter
                    point = [param_ranges[0][0] + (param_ranges[0][1] - param_ranges[0][0]) * i / (grid_size - 1)]
                    grid_points.append(point)
        
        # Convert to parameter dictionaries
        for point in grid_points:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = point[i]
            combinations.append(param_dict)
        
        return combinations[:max_combinations]
    
    def _generate_random_combinations(self, parameters: Dict[str, TaskParameters], 
                                     max_combinations: int) -> List[Dict[str, float]]:
        """Generate random parameter combinations"""
        combinations = []
        
        param_names = list(parameters.keys())
        param_ranges = [parameters[name].sweep.range for name in param_names]
        
        for _ in range(max_combinations):
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_range = param_ranges[i]
                value = np.random.uniform(param_range[0], param_range[1])
                param_dict[param_name] = value
            combinations.append(param_dict)
        
        return combinations
    
    def _generate_latin_hypercube_combinations(self, parameters: Dict[str, TaskParameters], 
                                              max_combinations: int) -> List[Dict[str, float]]:
        """Generate Latin Hypercube parameter combinations"""
        try:
            from SALib.sample import latin
            
            param_names = list(parameters.keys())
            param_ranges = [parameters[name].sweep.range for name in param_names]
            
            problem = {
                'num_vars': len(param_names),
                'names': param_names,
                'bounds': param_ranges
            }
            
            # Generate samples
            samples = latin.sample(problem, max_combinations)
            
            # Convert to parameter dictionaries
            combinations = []
            for sample in samples:
                param_dict = {}
                for i, param_name in enumerate(param_names):
                    param_dict[param_name] = float(sample[i])
                combinations.append(param_dict)
            
            return combinations
            
        except ImportError:
            self.logger.warning("SALib not available, falling back to random sampling")
            return self._generate_random_combinations(parameters, max_combinations)
    
    def cleanup(self):
        """Clean up simulation resources"""
        if self.sim:
            self.sim.cleanup()


class MockSimulation:
    """Mock simulation for testing and fallback"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MockSimulation")
        self.configured = False
    
    def configure(self, parameters: Dict[str, float], task_config: Dict[str, Any]):
        """Configure mock simulation"""
        self.parameters = parameters
        self.task_config = task_config
        self.configured = True
        self.logger.info(f"Mock simulation configured with {len(parameters)} parameters")
    
    def run(self) -> Tuple[bool, Dict[str, float], Optional[str]]:
        """Run mock simulation"""
        if not self.configured:
            return False, {}, "Simulation not configured"
        
        # Simulate computation time
        time.sleep(0.1)
        
        # Generate mock metrics based on parameters
        metrics = self._calculate_mock_metrics()
        
        # Determine success based on parameter values
        success = self._determine_success(metrics)
        failure_reason = None if success else "Mock failure condition met"
        
        return success, metrics, failure_reason
    
    def _calculate_mock_metrics(self) -> Dict[str, float]:
        """Calculate mock performance metrics"""
        metrics = {}
        
        # Calculate composite performance score
        performance = 0.0
        
        for param_name, value in self.parameters.items():
            # Simple heuristic-based performance calculation
            if 'friction' in param_name.lower():
                if 0.1 <= value <= 0.8:
                    performance += 0.2
                else:
                    performance -= 0.1
            
            elif 'mass' in param_name.lower():
                if 0.5 <= value <= 2.0:
                    performance += 0.15
                else:
                    performance -= 0.1
            
            elif 'gain' in param_name.lower():
                if 0.5 <= value <= 2.0:
                    performance += 0.2
                else:
                    performance -= 0.15
        
        # Add some randomness
        performance += np.random.normal(0, 0.05)
        performance = max(0.0, min(1.0, performance))
        
        metrics['performance'] = performance
        metrics['stability'] = 1.0 - abs(performance - 0.5) * 2
        metrics['efficiency'] = performance * 0.8 + 0.2
        
        return metrics
    
    def _determine_success(self, metrics: Dict[str, float]) -> bool:
        """Determine if simulation was successful"""
        # Success if performance is above threshold
        return metrics.get('performance', 0.0) > 0.3
    
    def cleanup(self):
        """Clean up mock simulation"""
        pass


class MuJoCoSimulation:
    """MuJoCo-based simulation runner"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MuJoCoSimulation")
        self.model = None
        self.data = None
        self.configured = False
    
    def configure(self, parameters: Dict[str, float], task_config: Dict[str, Any]):
        """Configure MuJoCo simulation"""
        try:
            import mujoco
            
            # Load model from task config
            model_path = task_config.get('model_path')
            if model_path and Path(model_path).exists():
                self.model = mujoco.MjModel.from_xml_path(model_path)
                self.data = mujoco.MjData(self.model)
                
                # Apply parameters to model
                self._apply_parameters(parameters)
                
                self.configured = True
                self.logger.info("MuJoCo simulation configured")
            else:
                self.logger.warning("No valid model path provided for MuJoCo")
                
        except Exception as e:
            self.logger.error(f"MuJoCo configuration failed: {e}")
            raise
    
    def _apply_parameters(self, parameters: Dict[str, float]):
        """Apply parameters to MuJoCo model"""
        if not self.model:
            return
        
        for param_name, value in parameters.items():
            # Apply to appropriate model properties
            if 'mass' in param_name.lower():
                # Apply to body masses
                for i in range(self.model.nbody):
                    if param_name.lower() in self.model.body_names[i].lower():
                        self.model.body_mass[i] = value
            
            elif 'friction' in param_name.lower():
                # Apply to geom friction
                for i in range(self.model.ngeom):
                    if param_name.lower() in self.model.geom_names[i].lower():
                        self.model.geom_friction[i, 0] = value
            
            elif 'damping' in param_name.lower():
                # Apply to joint damping
                for i in range(self.model.njnt):
                    if param_name.lower() in self.model.joint_names[i].lower():
                        self.model.dof_damping[i] = value
    
    def run(self) -> Tuple[bool, Dict[str, float], Optional[str]]:
        """Run MuJoCo simulation"""
        if not self.configured:
            return False, {}, "Simulation not configured"
        
        try:
            # Run simulation for specified duration
            duration = 5.0  # seconds
            dt = self.model.opt.timestep
            steps = int(duration / dt)
            
            # Reset simulation
            mujoco.mj_resetData(self.model, self.data)
            
            # Run simulation steps
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            # Determine success
            success = metrics.get('stability', 0.0) > 0.5
            failure_reason = None if success else "Low stability"
            
            return success, metrics, failure_reason
            
        except Exception as e:
            self.logger.error(f"MuJoCo simulation failed: {e}")
            return False, {}, str(e)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate simulation metrics"""
        metrics = {}
        
        if not self.data:
            return metrics
        
        # Calculate stability (based on joint velocities)
        joint_velocities = self.data.qvel
        stability = 1.0 / (1.0 + np.std(joint_velocities))
        
        # Calculate energy efficiency
        kinetic_energy = 0.5 * np.sum(self.data.qvel ** 2)
        efficiency = 1.0 / (1.0 + kinetic_energy)
        
        # Calculate task completion (simplified)
        completion = min(1.0, stability * efficiency)
        
        metrics['stability'] = float(stability)
        metrics['efficiency'] = float(efficiency)
        metrics['completion'] = float(completion)
        
        return metrics
    
    def cleanup(self):
        """Clean up MuJoCo simulation"""
        self.model = None
        self.data = None


class PyBulletSimulation:
    """PyBullet-based simulation runner"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PyBulletSimulation")
        self.physics_client = None
        self.robot_id = None
        self.configured = False
    
    def configure(self, parameters: Dict[str, float], task_config: Dict[str, Any]):
        """Configure PyBullet simulation"""
        try:
            import pybullet as p
            
            # Connect to physics server
            self.physics_client = p.connect(p.DIRECT)
            
            # Load robot model
            model_path = task_config.get('model_path')
            if model_path and Path(model_path).exists():
                self.robot_id = p.loadURDF(model_path)
                
                # Apply parameters
                self._apply_parameters(parameters)
                
                self.configured = True
                self.logger.info("PyBullet simulation configured")
            else:
                self.logger.warning("No valid model path provided for PyBullet")
                
        except Exception as e:
            self.logger.error(f"PyBullet configuration failed: {e}")
            raise
    
    def _apply_parameters(self, parameters: Dict[str, float]):
        """Apply parameters to PyBullet simulation"""
        if not self.robot_id:
            return
        
        import pybullet as p
        
        for param_name, value in parameters.items():
            # Apply to appropriate simulation properties
            if 'friction' in param_name.lower():
                # Apply to all links
                for i in range(p.getNumJoints(self.robot_id)):
                    p.changeDynamics(self.robot_id, i, lateralFriction=value)
            
            elif 'mass' in param_name.lower():
                # Apply to specific links
                for i in range(p.getNumJoints(self.robot_id)):
                    if param_name.lower() in p.getJointInfo(self.robot_id, i)[12].lower():
                        p.changeDynamics(self.robot_id, i, mass=value)
    
    def run(self) -> Tuple[bool, Dict[str, float], Optional[str]]:
        """Run PyBullet simulation"""
        if not self.configured:
            return False, {}, "Simulation not configured"
        
        try:
            import pybullet as p
            
            # Run simulation
            duration = 5.0  # seconds
            dt = 1.0 / 240.0  # 240 Hz
            steps = int(duration / dt)
            
            # Reset simulation
            p.resetSimulation()
            self.robot_id = p.loadURDF(self.task_config.get('model_path', ''))
            
            # Run simulation steps
            for _ in range(steps):
                p.stepSimulation()
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            # Determine success
            success = metrics.get('stability', 0.0) > 0.5
            failure_reason = None if success else "Low stability"
            
            return success, metrics, failure_reason
            
        except Exception as e:
            self.logger.error(f"PyBullet simulation failed: {e}")
            return False, {}, str(e)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate simulation metrics"""
        metrics = {}
        
        if not self.robot_id:
            return metrics
        
        try:
            import pybullet as p
            
            # Get joint states
            joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
            
            # Calculate stability (based on joint velocities)
            velocities = [state[1] for state in joint_states]
            stability = 1.0 / (1.0 + np.std(velocities))
            
            # Calculate energy
            kinetic_energy = 0.5 * sum(v * v for v in velocities)
            efficiency = 1.0 / (1.0 + kinetic_energy)
            
            metrics['stability'] = float(stability)
            metrics['efficiency'] = float(efficiency)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate metrics: {e}")
            metrics['stability'] = 0.5
            metrics['efficiency'] = 0.5
        
        return metrics
    
    def cleanup(self):
        """Clean up PyBullet simulation"""
        if self.physics_client:
            import pybullet as p
            p.disconnect(self.physics_client)
            self.physics_client = None
        self.robot_id = None 