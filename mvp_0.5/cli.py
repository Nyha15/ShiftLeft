#!/usr/bin/env python3
"""
Command Line Interface
======================

CLI for the Robotics Repository Analyzer MVP and CI Pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from main_analyzer import RoboticsAnalyzer
from ci_pipeline import RoboticsCIPipeline
from data_models import CIConfig, SweepMethod

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Robotics Repository Analyzer & CI Pipeline - Extract robot specs, tasks, and perform parameter analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing local repository
  python3 cli.py analyze /path/to/robot/repo
  python3 cli.py analyze /path/to/repo --output analysis.yaml --verbose
  
  # Run CI pipeline on remote repository
  python3 cli.py ci https://github.com/user/robot-repo.git
  python3 cli.py ci https://github.com/user/robot-repo.git --engine mujoco --method sobol --samples 200
  
  # Run CI pipeline with custom configuration
  python3 cli.py ci https://github.com/user/robot-repo.git --output ci_results --engine pybullet --method oat
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command (original functionality)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing local repository')
    analyze_parser.add_argument(
        'repository',
        help='Path to robotics repository (local path)'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        default='robotics_analysis_results',
        help='Output directory path (default: robotics_analysis_results)'
    )
    analyze_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # CI pipeline command (new functionality)
    ci_parser = subparsers.add_parser('ci', help='Run CI pipeline on remote repository')
    ci_parser.add_argument(
        'repo_url',
        help='Git repository URL to clone and analyze'
    )
    ci_parser.add_argument(
        '--output', '-o',
        default='ci_pipeline_results',
        help='Output directory path (default: ci_pipeline_results)'
    )
    ci_parser.add_argument(
        '--engine', '-e',
        choices=['mujoco', 'pybullet', 'gazebo', 'mock'],
        default='mock',
        help='Simulation engine to use (default: mock)'
    )
    ci_parser.add_argument(
        '--method', '-m',
        choices=['sobol', 'oat', 'latin_hypercube', 'random'],
        default='sobol',
        help='Sensitivity analysis method (default: sobol)'
    )
    ci_parser.add_argument(
        '--samples', '-s',
        type=int,
        default=100,
        help='Maximum number of samples for sensitivity analysis (default: 100)'
    )
    ci_parser.add_argument(
        '--seed', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    ci_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    if args.command == 'analyze':
        run_analysis(args, logger)
    elif args.command == 'ci':
        run_ci_pipeline(args, logger)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

def run_analysis(args, logger):
    """Run repository analysis (original functionality)"""
    # Validate repository path
    repo_path = Path(args.repository)
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    if not repo_path.is_dir():
        logger.error(f"Repository path is not a directory: {repo_path}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = RoboticsAnalyzer()
    
    try:
        # Run analysis
        logger.info("Starting robotics repository analysis...")
        result = analyzer.analyze_repository(str(repo_path))
        
        # Save results
        output_dir = Path(args.output)
        analyzer.save_results(result, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("ROBOTICS REPOSITORY ANALYSIS COMPLETE")
        print("="*60)
        print(f"Repository: {result.repository_path}")
        print(f"Analysis Time: {result.analysis_time}")
        print(f"Overall Confidence: {result.confidence:.2f}")
        print()
        print("SUMMARY:")
        print(f"  • Robots Found: {result.summary['total_robots']}")
        print(f"  • Tasks Found: {result.summary['total_tasks']}")
        print(f"  • Config Files: {result.summary['total_config_files']}")
        print(f"  • Total DOF: {result.summary['total_dof']}")
        print(f"  • Avg Task Complexity: {result.summary['avg_task_complexity']}")
        
        if result.robots:
            print("\nROBOTS:")
            for robot in result.robots:
                print(f"  • {robot.name}: {robot.dof} DOF")
                if robot.end_effector:
                    print(f"    End Effector: {robot.end_effector}")
        
        if result.summary['task_types']:
            print("\nTASK TYPES:")
            for task_type, count in result.summary['task_types'].items():
                print(f"  • {task_type}: {count}")
        
        print(f"\nDetailed results saved to: {output_dir}/")
        print(f"  • Individual task files: {output_dir}/tasks/")
        print(f"  • Summary document: {output_dir}/ANALYSIS_SUMMARY.md")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def run_ci_pipeline(args, logger):
    """Run CI pipeline on remote repository"""
    try:
        # Create CI configuration
        config = CIConfig(
            repo_url=args.repo_url,
            local_path=f"cloned_repo_{hash(args.repo_url) % 10000}",
            simulation_engine=args.engine,
            sensitivity_method=SweepMethod(args.method),
            max_samples=args.samples,
            random_seed=args.seed,
            output_dir=args.output
        )
        
        # Initialize CI pipeline
        logger.info("Initializing CI pipeline...")
        pipeline = RoboticsCIPipeline(config)
        
        # Run pipeline
        logger.info("Starting CI pipeline execution...")
        result = pipeline.run_pipeline()
        
        if result.success:
            # Print success summary
            print("\n" + "="*60)
            print("ROBOTICS CI PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Repository: {result.config.repo_url}")
            print(f"Simulation Engine: {result.config.simulation_engine}")
            print(f"Sensitivity Method: {result.config.sensitivity_method.value}")
            print(f"Run Time: {result.run_time:.2f}s")
            print()
            print("RESULTS:")
            print(f"  • Parameters Extracted: {len(result.extracted_parameters)}")
            print(f"  • Parameters Analyzed: {len(result.sensitivity_results)}")
            print(f"  • YAML Files Updated: {len(result.updated_yamls)}")
            print(f"  • Artifacts Generated: {result.artifacts_dir}")
            
            if result.sensitivity_results:
                print("\nHIGH IMPACT PARAMETERS:")
                high_impact = [r for r in result.sensitivity_results if r.priority.value == 'high']
                for param in high_impact[:5]:  # Show top 5
                    print(f"  • {param.parameter_name}: Sobol Index = {param.sobol_index:.3f}")
            
            print(f"\nDetailed results saved to: {result.config.output_dir}/")
            print(f"  • Parameter summary: {result.config.output_dir}/artifacts/parameter_summary.md")
            print(f"  • CI report: {result.config.output_dir}/artifacts/ci_report.md")
            if result.sensitivity_results:
                print(f"  • Sensitivity plots: {result.config.output_dir}/artifacts/sensitivity_plot.png")
            print("="*60)
            
        else:
            # Print failure summary
            print("\n" + "="*60)
            print("ROBOTICS CI PIPELINE FAILED")
            print("="*60)
            print(f"Repository: {result.config.repo_url}")
            print(f"Run Time: {result.run_time:.2f}s")
            print(f"Error: Pipeline execution failed")
            print("="*60)
            sys.exit(1)
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"CI pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
