#!/usr/bin/env python3
"""
Enhanced Command Line Interface
==============================

CLI for the Enhanced Robotics Repository Analyzer that produces production-ready YAMLs
with semantic descriptions, parameters (with Sobol ranges), KPIs, and function names.
"""

import argparse
import logging
import sys
from pathlib import Path

from enhanced_main_analyzer import EnhancedRoboticsAnalyzer

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI interface for enhanced analysis"""
    parser = argparse.ArgumentParser(
        description='Enhanced Robotics Repository Analyzer - Produces production-ready YAMLs with semantic descriptions, parameters (with Sobol ranges), KPIs, and function names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze repository and generate task YAMLs
  python3 enhanced_cli.py analyze /path/to/robot/repo
  python3 enhanced_cli.py analyze /path/to/repo --output task_yamls --verbose
  
  # Analyze GitHub repository (will be cloned automatically)
  python3 enhanced_cli.py analyze https://github.com/user/robot-repo.git
  
  # Generate master YAML with all tasks
  python3 enhanced_cli.py analyze /path/to/repo --master-yaml
  
  # Analyze with custom output directory
  python3 enhanced_cli.py analyze /path/to/repo --output /custom/output/dir
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze repository and generate task YAMLs')
    analyze_parser.add_argument(
        'repository',
        help='Path to robotics repository (local path) or GitHub URL'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory path (default: repository/task_analysis_output)'
    )
    analyze_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    analyze_parser.add_argument(
        '--master-yaml',
        action='store_true',
        help='Generate master YAML with all tasks'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'analyze':
            run_enhanced_analysis(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def run_enhanced_analysis(args, logger):
    """Run enhanced analysis on repository"""
    repo_path = args.repository
    
    # Check if it's a URL and clone if needed
    if repo_path.startswith('http'):
        logger.info(f"Detected repository URL: {repo_path}")
        logger.info("Cloning repository...")
        
        import tempfile
        import subprocess
        
        # Create temporary directory for cloning
        temp_dir = tempfile.mkdtemp(prefix="repo_analysis_")
        repo_path = Path(temp_dir) / "cloned_repo"
        
        try:
            # Clone the repository
            result = subprocess.run(
                ['git', 'clone', args.repository, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to clone repository: {result.stderr}")
                sys.exit(1)
            
            logger.info(f"Repository cloned to: {repo_path}")
            
        except subprocess.TimeoutExpired:
            logger.error("Repository cloning timed out")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            sys.exit(1)
    else:
        repo_path = Path(repo_path)
        
        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            sys.exit(1)
        
        if not repo_path.is_dir():
            logger.error(f"Repository path is not a directory: {repo_path}")
            sys.exit(1)
    
    logger.info(f"Starting enhanced analysis of: {repo_path}")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedRoboticsAnalyzer()
    
    # Run analysis
    result = analyzer.analyze_repository(str(repo_path), args.output)
    
    # Generate master YAML if requested
    if args.master_yaml:
        logger.info("Generating master YAML...")
        tasks = analyzer.task_extractor.extract_enhanced_tasks(repo_path)
        output_dir = Path(args.output) if args.output else Path("analysis_output")
        master_file = analyzer.generate_master_yaml(tasks, output_dir)
        logger.info(f"Master YAML generated: {master_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED ANALYSIS COMPLETE")
    print("="*60)
    print(f"Repository: {result['repository_path']}")
    print(f"Total Tasks: {result['total_tasks']}")
    print(f"Overall Confidence: {result['overall_confidence']:.2f}")
    print(f"Analysis Time: {result['analysis_time']:.2f}s")
    print(f"Generated Files: {len(result['generated_files'])}")
    print(f"Output Directory: {result['generated_files'][0] if result['generated_files'] else 'None'}")
    
    # Print task categories
    if result['summary']['task_categories']:
        print("\nTask Categories:")
        for category, count in result['summary']['task_categories'].items():
            print(f"  {category}: {count}")
    
    # Print confidence distribution
    confidence_dist = result['summary']['confidence_distribution']
    print(f"\nConfidence Distribution:")
    print(f"  High (≥0.7): {confidence_dist['high']}")
    print(f"  Medium (0.4-0.7): {confidence_dist['medium']}")
    print(f"  Low (<0.4): {confidence_dist['low']}")
    
    # Print parameter and KPI statistics
    param_stats = result['summary']['parameter_statistics']
    kpi_stats = result['summary']['kpi_statistics']
    print(f"\nParameters: {param_stats['total_parameters']} total, {param_stats['average_per_task']:.1f} avg/task")
    print(f"KPIs: {kpi_stats['total_kpis']} total, {kpi_stats['average_per_task']:.1f} avg/task")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the output directory for generated YAMLs.")
    print("="*60)
    
    # Cleanup cloned repository if it was cloned
    if args.repository.startswith('http'):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up cloned repository")
        except Exception as e:
            logger.warning(f"Failed to cleanup cloned repository: {e}")

if __name__ == '__main__':
    main() 