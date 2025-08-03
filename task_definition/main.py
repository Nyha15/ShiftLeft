#!/usr/bin/env python3
"""
ShiftLeft Task Definition - Main Entry Point

This module provides the command-line interface for the ShiftLeft Task Definition module,
which analyzes robotics Git repositories and extracts robot specifications and task
sequences to generate a task.yaml file.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import tempfile
import shutil
import git

from task_definition.robotics_repo_analyzer.scanner import RepositoryScanner
from task_definition.robotics_repo_analyzer.fusion import InformationFusion
from task_definition.robotics_repo_analyzer.output import OutputGenerator
from task_definition.utils.cleanup import cleanup_temp_dir, cleanup_output_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze robotics repositories and extract robot specifications and task sequences.'
    )
    parser.add_argument(
        'repository',
        help='Git repository URL or local repository path'
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
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM for complex code analysis'
    )
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'anthropic', 'llama'],
        default='openai',
        help='LLM provider to use (default: openai)'
    )
    parser.add_argument(
        '--llm-model',
        default='gpt-4',
        help='LLM model to use (default: gpt-4)'
    )
    parser.add_argument(
        '--llm-api-key',
        help='API key for the LLM provider'
    )
    parser.add_argument(
        '--llm-model-path',
        help='Path to local LLM model (for llama provider)'
    )
    parser.add_argument(
        '--complexity-threshold',
        type=float,
        default=0.7,
        help='Complexity threshold for using LLM (0.0-1.0, default: 0.7)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up temporary files after analysis'
    )
    
    return parser.parse_args()

def clone_repository(repo_url, target_dir):
    """Clone a Git repository to a target directory."""
    logger.info(f"Cloning repository: {repo_url}")
    try:
        git.Repo.clone_from(repo_url, target_dir)
        logger.info(f"Repository cloned to: {target_dir}")
        return True
    except git.GitCommandError as e:
        logger.error(f"Error cloning repository: {e}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM client if needed
    llm_client = None
    if args.use_llm:
        try:
            if args.llm_provider == 'openai':
                from task_definition.robotics_repo_analyzer.llm.client import OpenAIClient
                llm_client = OpenAIClient(api_key=args.llm_api_key, model=args.llm_model)
            elif args.llm_provider == 'anthropic':
                from task_definition.robotics_repo_analyzer.llm.client import AnthropicClient
                llm_client = AnthropicClient(api_key=args.llm_api_key, model=args.llm_model)
            elif args.llm_provider == 'llama':
                from task_definition.robotics_repo_analyzer.llm.llama_client import LlamaClient
                llm_client = LlamaClient(model_path=args.llm_model_path)
            
            logger.info(f"Using LLM provider: {args.llm_provider}, model: {args.llm_model}")
        except ImportError as e:
            logger.error(f"Error initializing LLM client: {e}")
            logger.error(f"Make sure you have installed the required dependencies: pip install -e '.[llm]'")
            args.use_llm = False
    
    # Determine if the repository is a URL or local path
    repo_path = args.repository
    temp_dir = None
    
    if repo_path.startswith(('http://', 'https://', 'git://', 'ssh://')):
        # Clone the repository to a temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix='robotics_repo_analyzer_'))
        if not clone_repository(repo_path, temp_dir):
            logger.error("Failed to clone repository. Exiting.")
            sys.exit(1)
        repo_path = temp_dir
    
    try:
        # Scan the repository
        logger.info(f"Scanning repository: {repo_path}")
        scanner = RepositoryScanner(
            repo_path=repo_path,
            use_llm=args.use_llm,
            llm_client=llm_client,
            complexity_threshold=args.complexity_threshold
        )
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
        
        # Log LLM usage statistics
        if args.use_llm and llm_client:
            logger.info(f"LLM Usage: {llm_client.request_count} requests")
    
    finally:
        # Clean up temporary directory
        if temp_dir and args.cleanup:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            cleanup_temp_dir(temp_dir)

if __name__ == '__main__':
    main()