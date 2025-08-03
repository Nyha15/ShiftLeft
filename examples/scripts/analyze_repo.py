#!/usr/bin/env python3
"""
Example script to analyze a robotics repository.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from task_definition.robotics_repo_analyzer.scanner import RepositoryScanner
from task_definition.robotics_repo_analyzer.fusion import InformationFusion
from task_definition.robotics_repo_analyzer.output import OutputGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze a robotics repository and extract robot specifications and task sequences.'
    )
    parser.add_argument(
        'repository',
        help='Local repository path'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='task.yaml',
        help='Output file path (default: task.yaml)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Scan the repository
    logger.info(f"Scanning repository: {args.repository}")
    scanner = RepositoryScanner(repo_path=args.repository)
    scan_results = scanner.scan()
    
    # Fuse information
    logger.info("Fusing information...")
    fusion = InformationFusion(scan_results)
    fused_data = fusion.fuse()
    
    # Generate output
    logger.info(f"Generating output: {output_path}")
    generator = OutputGenerator(fused_data)
    generator.generate(output_path)
    
    logger.info(f"Analysis complete. Output written to: {output_path}")

if __name__ == '__main__':
    main()
