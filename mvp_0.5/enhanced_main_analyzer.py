#!/usr/bin/env python3
"""
Enhanced Main Robotics Analyzer
===============================

Produces production-ready YAMLs with semantic descriptions, parameters (with Sobol ranges),
KPIs, and function names for robotics tasks.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from enhanced_task_extractor import EnhancedTaskExtractor, EnhancedTaskInfo

logger = logging.getLogger(__name__)

class EnhancedRoboticsAnalyzer:
    """Enhanced analyzer that produces production-ready task YAMLs"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedRoboticsAnalyzer")
        self.task_extractor = EnhancedTaskExtractor()
    
    def analyze_repository(self, repo_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Analyze a robotics repository and generate production-ready YAMLs"""
        start_time = datetime.now()
        self.logger.info(f"Starting enhanced analysis of {repo_path}")
        
        try:
            repo_path = Path(repo_path).resolve()
            
            if not repo_path.exists():
                raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
            
            if not repo_path.is_dir():
                raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")
            
            if output_dir is None:
                output_dir = repo_path / "task_analysis_output"
            else:
                output_dir = Path(output_dir)
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract enhanced tasks
            self.logger.info("Extracting enhanced task information...")
            tasks = self.task_extractor.extract_enhanced_tasks(repo_path)
            self.logger.info(f"Found {len(tasks)} enhanced tasks")
            
            if not tasks:
                self.logger.warning("No tasks found. This might indicate:")
                self.logger.warning("- Repository contains no Python files")
                self.logger.warning("- Python files don't contain robotics code")
                self.logger.warning("- Confidence threshold is too high")
            
            # Generate task YAMLs
            self.logger.info("Generating task YAMLs...")
            generated_files = self.task_extractor.generate_task_yamls(tasks, output_dir)
            
            # Generate summary
            summary = self._generate_enhanced_summary(tasks, generated_files)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(tasks)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "repository_path": str(repo_path),
                "analysis_time": analysis_time,
                "total_tasks": len(tasks),
                "overall_confidence": overall_confidence,
                "generated_files": generated_files,
                "summary": summary,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Save analysis result
            result_file = output_dir / "analysis_summary.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            self.logger.info(f"Enhanced analysis complete in {analysis_time:.2f}s")
            self.logger.info(f"Generated {len(generated_files)} task YAMLs")
            self.logger.info(f"Results saved to: {output_dir}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            # Return partial results if possible
            partial_result = {
                "repository_path": str(repo_path) if 'repo_path' in locals() else str(repo_path),
                "analysis_time": (datetime.now() - start_time).total_seconds(),
                "total_tasks": 0,
                "overall_confidence": 0.0,
                "generated_files": [],
                "summary": {},
                "analysis_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            
            # Try to save partial results
            try:
                if 'output_dir' in locals():
                    result_file = output_dir / "analysis_summary.json"
                    with open(result_file, 'w') as f:
                        json.dump(partial_result, f, indent=2, default=str)
            except:
                pass
            
            return partial_result
    
    def _generate_enhanced_summary(self, tasks: List[EnhancedTaskInfo], 
                                 generated_files: List[str]) -> Dict[str, Any]:
        """Generate enhanced summary of analysis results"""
        summary = {
            "task_categories": {},
            "parameter_statistics": {},
            "kpi_statistics": {},
            "confidence_distribution": {},
            "file_coverage": {}
        }
        
        # Task categories
        for task in tasks:
            # Determine task category from semantic description
            category = self._categorize_task(task.semantic_description)
            if category not in summary["task_categories"]:
                summary["task_categories"][category] = 0
            summary["task_categories"][category] += 1
        
        # Parameter statistics
        total_params = 0
        param_types = {}
        for task in tasks:
            total_params += len(task.parameters)
            for param in task.parameters:
                param_type = param.source
                if param_type not in param_types:
                    param_types[param_type] = 0
                param_types[param_type] += 1
        
        summary["parameter_statistics"] = {
            "total_parameters": total_params,
            "average_per_task": total_params / len(tasks) if tasks else 0,
            "by_source": param_types
        }
        
        # KPI statistics
        total_kpis = 0
        kpi_types = {}
        for task in tasks:
            total_kpis += len(task.kpis)
            for kpi in task.kpis:
                data_type = kpi.data_type
                if data_type not in kpi_types:
                    kpi_types[data_type] = 0
                kpi_types[data_type] += 1
        
        summary["kpi_statistics"] = {
            "total_kpis": total_kpis,
            "average_per_task": total_kpis / len(tasks) if tasks else 0,
            "by_data_type": kpi_types
        }
        
        # Confidence distribution
        confidence_ranges = {
            "high": 0,      # 0.7-1.0
            "medium": 0,    # 0.4-0.7
            "low": 0        # 0.0-0.4
        }
        
        for task in tasks:
            if task.confidence >= 0.7:
                confidence_ranges["high"] += 1
            elif task.confidence >= 0.4:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1
        
        summary["confidence_distribution"] = confidence_ranges
        
        # File coverage
        file_extensions = {}
        for task in tasks:
            ext = Path(task.file_path).suffix
            if ext not in file_extensions:
                file_extensions[ext] = 0
            file_extensions[ext] += 1
        
        summary["file_coverage"] = {
            "total_files": len(set(task.file_path for task in tasks)),
            "by_extension": file_extensions
        }
        
        return summary
    
    def _categorize_task(self, semantic_description: str) -> str:
        """Categorize task based on semantic description"""
        description_lower = semantic_description.lower()
        
        if any(keyword in description_lower for keyword in ['pick', 'place', 'grasp', 'grip']):
            return "manipulation"
        elif any(keyword in description_lower for keyword in ['move', 'navigate', 'path']):
            return "navigation"
        elif any(keyword in description_lower for keyword in ['detect', 'recognize', 'vision']):
            return "perception"
        elif any(keyword in description_lower for keyword in ['control', 'servo', 'pid']):
            return "control"
        elif any(keyword in description_lower for keyword in ['kinematics', 'dynamics', 'jacobian']):
            return "kinematics"
        elif any(keyword in description_lower for keyword in ['plan', 'trajectory', 'motion', 'rrt', 'apf', 'potential field']):
            return "motion_planning"
        elif any(keyword in description_lower for keyword in ['torque', 'force', 'inertia', 'mass', 'gravity']):
            return "dynamics"
        else:
            return "general"
    
    def _calculate_overall_confidence(self, tasks: List[EnhancedTaskInfo]) -> float:
        """Calculate overall confidence of the analysis"""
        if not tasks:
            return 0.0
        
        total_confidence = sum(task.confidence for task in tasks)
        return total_confidence / len(tasks)
    
    def generate_master_yaml(self, tasks: List[EnhancedTaskInfo], output_dir: Path) -> str:
        """Generate a master YAML file containing all tasks"""
        master_data = {
            "repository_analysis": {
                "total_tasks": len(tasks),
                "analysis_timestamp": datetime.now().isoformat(),
                "format_version": "1.0"
            },
            "tasks": []
        }
        
        for task in tasks:
            task_dict = {
                "task_name": task.task_name,
                "semantic_description": task.semantic_description,
                "parameters": [
                    {
                        "variable_name": param.variable_name,
                        "current_value": param.current_value,
                        "range": [param.range_min, param.range_max],
                        "unit": param.unit,
                        "description": param.description,
                        "source": param.source
                    }
                    for param in task.parameters
                ],
                "kpis": [
                    {
                        "name": kpi.name,
                        "data_type": kpi.data_type,
                        "description": kpi.description,
                        "success_criteria": kpi.success_criteria,
                        "monitoring_frequency": kpi.monitoring_frequency
                    }
                    for kpi in task.kpis
                ],
                "function_names": task.function_names,
                "file_path": task.file_path,
                "confidence": task.confidence
            }
            master_data["tasks"].append(task_dict)
        
        # Save master YAML
        master_file = output_dir / "all_tasks_master.yaml"
        with open(master_file, 'w') as f:
            yaml.dump(master_data, f, default_flow_style=False, 
                     sort_keys=False, indent=2, width=80)
        
        self.logger.info(f"Generated master YAML: {master_file}")
        return str(master_file) 