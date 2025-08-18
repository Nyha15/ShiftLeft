#!/usr/bin/env python3
"""
Sensitivity Analyzer
====================

Performs parameter sensitivity analysis using SALib and simulation.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

try:
    from SALib.sample import sobol, latin, morris
    from SALib.analyze import sobol as sobol_analyze
    from SALib.analyze import morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logging.warning("SALib not available. Sensitivity analysis will be limited.")

from data_models import (
    TaskParameters, ParameterSweep, SensitivityResult, 
    SweepMethod, ParameterPriority, SimulationResult
)

logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """Performs parameter sensitivity analysis for robotics parameters"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SensitivityAnalyzer")
        
        if not SALIB_AVAILABLE:
            self.logger.warning("SALib not available. Using simplified analysis.")
    
    def analyze_parameters(self, parameters: Dict[str, TaskParameters], 
                          method: SweepMethod, max_samples: int, 
                          random_seed: int) -> List[SensitivityResult]:
        """Analyze parameter sensitivity using specified method"""
        if not parameters:
            return []
        
        self.logger.info(f"Starting sensitivity analysis with {method.value} method")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        if method == SweepMethod.SOBOL:
            return self._sobol_analysis(parameters, max_samples)
        elif method == SweepMethod.OAT:
            return self._oat_analysis(parameters, max_samples)
        elif method == SweepMethod.LATIN_HYPERCUBE:
            return self._latin_hypercube_analysis(parameters, max_samples)
        else:
            return self._random_analysis(parameters, max_samples)
    
    def _sobol_analysis(self, parameters: Dict[str, TaskParameters], 
                        max_samples: int) -> List[SensitivityResult]:
        """Perform Sobol sensitivity analysis"""
        if not SALIB_AVAILABLE:
            return self._fallback_analysis(parameters, "sobol")
        
        try:
            # Prepare problem definition for SALib
            problem = self._create_salib_problem(parameters)
            
            # Generate samples
            param_values = sobol.sample(problem, max_samples, calc_second_order=False)
            
            # Run simulations to get outputs
            outputs = self._evaluate_parameters(param_values, parameters)
            
            # Analyze results
            results = sobol_analyze.analyze(problem, outputs, calc_second_order=False)
            
            # Convert to our format
            sensitivity_results = []
            for i, param_name in enumerate(problem['names']):
                if param_name in parameters:
                    param = parameters[param_name]
                    
                    result = SensitivityResult(
                        parameter_name=param_name,
                        sobol_index=float(results['S1'][i]),
                        variance_contribution=float(results['ST'][i]),
                        priority=param.sweep.priority,
                        confidence=min(0.9, 1.0 - np.std(outputs) / np.mean(outputs))
                    )
                    sensitivity_results.append(result)
            
            self.logger.info(f"Sobol analysis completed for {len(sensitivity_results)} parameters")
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"Sobol analysis failed: {e}")
            return self._fallback_analysis(parameters, "sobol")
    
    def _oat_analysis(self, parameters: Dict[str, TaskParameters], 
                      max_samples: int) -> List[SensitivityResult]:
        """Perform One-At-a-Time sensitivity analysis"""
        try:
            sensitivity_results = []
            
            for param_name, param in parameters.items():
                # Get nominal value
                nominal_value = param.nominal_value
                param_range = param.sweep.range
                
                # Test values around nominal
                test_values = np.linspace(param_range[0], param_range[1], 
                                        min(max_samples // len(parameters), 10))
                
                outputs = []
                for test_val in test_values:
                    # Create parameter set with this value
                    test_params = {name: p.nominal_value for name, p in parameters.items()}
                    test_params[param_name] = test_val
                    
                    # Evaluate
                    output = self._evaluate_single_parameter_set(test_params, parameters)
                    outputs.append(output)
                
                # Calculate sensitivity (variance in output)
                if len(outputs) > 1:
                    variance_contribution = np.var(outputs) / (np.mean(outputs) + 1e-8)
                    
                    # Normalize to 0-1 range
                    normalized_sensitivity = min(1.0, variance_contribution / 10.0)
                    
                    result = SensitivityResult(
                        parameter_name=param_name,
                        sobol_index=normalized_sensitivity,
                        variance_contribution=float(variance_contribution),
                        priority=param.sweep.priority,
                        confidence=0.7
                    )
                    sensitivity_results.append(result)
            
            self.logger.info(f"OAT analysis completed for {len(sensitivity_results)} parameters")
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"OAT analysis failed: {e}")
            return self._fallback_analysis(parameters, "oat")
    
    def _latin_hypercube_analysis(self, parameters: Dict[str, TaskParameters], 
                                  max_samples: int) -> List[SensitivityResult]:
        """Perform Latin Hypercube sampling analysis"""
        if not SALIB_AVAILABLE:
            return self._fallback_analysis(parameters, "latin_hypercube")
        
        try:
            # Prepare problem definition
            problem = self._create_salib_problem(parameters)
            
            # Generate samples
            param_values = latin.sample(problem, max_samples)
            
            # Run simulations
            outputs = self._evaluate_parameters(param_values, parameters)
            
            # Calculate variance contribution for each parameter
            sensitivity_results = []
            for i, param_name in enumerate(problem['names']):
                if param_name in parameters:
                    param = parameters[param_name]
                    
                    # Calculate correlation between parameter and output
                    param_col = param_values[:, i]
                    correlation = np.corrcoef(param_col, outputs)[0, 1]
                    
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    # Convert correlation to sensitivity index
                    sensitivity_index = abs(correlation)
                    
                    result = SensitivityResult(
                        parameter_name=param_name,
                        sobol_index=sensitivity_index,
                        variance_contribution=sensitivity_index,
                        priority=param.sweep.priority,
                        confidence=0.8
                    )
                    sensitivity_results.append(result)
            
            self.logger.info(f"Latin Hypercube analysis completed for {len(sensitivity_results)} parameters")
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"Latin Hypercube analysis failed: {e}")
            return self._fallback_analysis(parameters, "latin_hypercube")
    
    def _random_analysis(self, parameters: Dict[str, TaskParameters], 
                         max_samples: int) -> List[SensitivityResult]:
        """Perform random sampling analysis"""
        try:
            sensitivity_results = []
            
            # Generate random parameter combinations
            num_combinations = min(max_samples, 50)
            
            for param_name, param in parameters.items():
                # Generate random values within range
                random_values = np.random.uniform(
                    param.sweep.range[0], 
                    param.sweep.range[1], 
                    num_combinations
                )
                
                outputs = []
                for random_val in random_values:
                    # Create parameter set
                    test_params = {name: p.nominal_value for name, p in parameters.items()}
                    test_params[param_name] = random_val
                    
                    # Evaluate
                    output = self._evaluate_single_parameter_set(test_params, parameters)
                    outputs.append(output)
                
                # Calculate sensitivity
                if len(outputs) > 1:
                    variance_contribution = np.var(outputs) / (np.mean(outputs) + 1e-8)
                    normalized_sensitivity = min(1.0, variance_contribution / 5.0)
                    
                    result = SensitivityResult(
                        parameter_name=param_name,
                        sobol_index=normalized_sensitivity,
                        variance_contribution=float(variance_contribution),
                        priority=param.sweep.priority,
                        confidence=0.6
                    )
                    sensitivity_results.append(result)
            
            self.logger.info(f"Random analysis completed for {len(sensitivity_results)} parameters")
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"Random analysis failed: {e}")
            return self._fallback_analysis(parameters, "random")
    
    def _fallback_analysis(self, parameters: Dict[str, TaskParameters], 
                           method_name: str) -> List[SensitivityResult]:
        """Fallback analysis when primary method fails"""
        self.logger.warning(f"Using fallback analysis for {method_name}")
        
        sensitivity_results = []
        
        for param_name, param in parameters.items():
            # Simple heuristic-based sensitivity
            priority_to_sensitivity = {
                ParameterPriority.HIGH: 0.8,
                ParameterPriority.MEDIUM: 0.5,
                ParameterPriority.LOW: 0.2
            }
            
            # Adjust based on parameter type
            param_type = self._infer_parameter_type(param_name)
            type_multiplier = self._get_type_sensitivity_multiplier(param_type)
            
            base_sensitivity = priority_to_sensitivity[param.sweep.priority]
            adjusted_sensitivity = base_sensitivity * type_multiplier
            
            result = SensitivityResult(
                parameter_name=param_name,
                sobol_index=adjusted_sensitivity,
                variance_contribution=adjusted_sensitivity,
                priority=param.sweep.priority,
                confidence=0.5
            )
            sensitivity_results.append(result)
        
        return sensitivity_results
    
    def _create_salib_problem(self, parameters: Dict[str, TaskParameters]) -> Dict:
        """Create SALib problem definition"""
        names = []
        bounds = []
        
        for param_name, param in parameters.items():
            names.append(param_name)
            bounds.append(param.sweep.range)
        
        return {
            'num_vars': len(names),
            'names': names,
            'bounds': bounds
        }
    
    def _evaluate_parameters(self, param_values: np.ndarray, 
                           parameters: Dict[str, TaskParameters]) -> np.ndarray:
        """Evaluate multiple parameter sets"""
        outputs = []
        
        for i in range(param_values.shape[0]):
            # Create parameter dictionary for this sample
            param_dict = {}
            for j, param_name in enumerate(parameters.keys()):
                param_dict[param_name] = param_values[i, j]
            
            # Evaluate single parameter set
            output = self._evaluate_single_parameter_set(param_dict, parameters)
            outputs.append(output)
        
        return np.array(outputs)
    
    def _evaluate_single_parameter_set(self, param_dict: Dict[str, float], 
                                     parameters: Dict[str, TaskParameters]) -> float:
        """Evaluate a single parameter set and return performance metric"""
        try:
            # This is a simplified evaluation - in practice, this would run
            # the actual robotics simulation or task
            
            # Calculate a composite performance metric based on parameter values
            performance = 0.0
            
            for param_name, value in param_dict.items():
                if param_name in parameters:
                    param = parameters[param_name]
                    nominal = param.nominal_value
                    
                    # Normalize parameter value
                    normalized = (value - nominal) / (nominal + 1e-8)
                    
                    # Penalize deviation from nominal
                    deviation_penalty = abs(normalized) * 0.1
                    
                    # Add parameter-specific effects
                    if 'friction' in param_name.lower():
                        # Friction affects stability
                        if value < 0.1:
                            deviation_penalty += 0.5  # Too low friction
                        elif value > 0.8:
                            deviation_penalty += 0.2  # High friction
                    
                    elif 'mass' in param_name.lower():
                        # Mass affects dynamics
                        if abs(normalized) > 0.3:
                            deviation_penalty += 0.3
                    
                    elif 'gain' in param_name.lower() or 'stiffness' in param_name.lower():
                        # Control gains affect performance
                        if value < nominal * 0.5:
                            deviation_penalty += 0.4  # Too low gain
                        elif value > nominal * 2.0:
                            deviation_penalty += 0.4  # Too high gain
                    
                    performance -= deviation_penalty
            
            # Add some randomness to simulate real simulation
            noise = np.random.normal(0, 0.01)
            performance += noise
            
            # Ensure performance is in reasonable range
            return max(0.0, min(1.0, performance + 0.5))
            
        except Exception as e:
            self.logger.warning(f"Parameter evaluation failed: {e}")
            return 0.5  # Default neutral performance
    
    def _infer_parameter_type(self, param_name: str) -> str:
        """Infer parameter type from name"""
        name_lower = param_name.lower()
        
        if any(word in name_lower for word in ['mass', 'inertia']):
            return 'physical'
        elif any(word in name_lower for word in ['friction', 'damping']):
            return 'contact'
        elif any(word in name_lower for word in ['gain', 'stiffness', 'pid']):
            return 'control'
        elif any(word in name_lower for word in ['noise', 'threshold']):
            return 'sensor'
        elif any(word in name_lower for word in ['limit', 'velocity', 'acceleration']):
            return 'constraint'
        else:
            return 'general'
    
    def _get_type_sensitivity_multiplier(self, param_type: str) -> float:
        """Get sensitivity multiplier based on parameter type"""
        type_multipliers = {
            'physical': 1.2,      # Physical parameters are important
            'contact': 1.5,       # Contact parameters are very important
            'control': 1.3,       # Control parameters are important
            'sensor': 1.1,        # Sensor parameters are moderately important
            'constraint': 1.0,    # Constraint parameters are standard
            'general': 0.8        # General parameters are less important
        }
        
        return type_multipliers.get(param_type, 1.0) 