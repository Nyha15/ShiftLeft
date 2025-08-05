#!/usr/bin/env python3
"""
Command Line Interface
======================

CLI for the Robotics Repository Analyzer MVP.
"""

import argparse
import logging
import sys
from pathlib import Path

from main_analyzer import RoboticsAnalyzer

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
        description='Robotics Repository Analyzer - Extract robot specs and tasks from codebases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py /path/to/robot/repo
  python cli.py /path/to/repo --output analysis.yaml --verbose
        """
    )
    
    parser.add_argument(
        'repository',
        help='Path to robotics repository (local path)'
    )
    parser.add_argument(
        '--output', '-o',
        default='robotics_analysis_results',
        help='Output directory path (default: robotics_analysis_results)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
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

if __name__ == '__main__':
    main()
