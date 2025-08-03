#!/usr/bin/env python3
"""
Robotics Repository Analyzer - Main Entry Point

This module provides the command-line interface for the Robotics Repository Analyzer,
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

from robotics_repo_analyzer.scanner import RepositoryScanner
from robotics_repo_analyzer.fusion import InformationFusion
from robotics_repo_analyzer.output import OutputGenerator

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
    
    # Add LLM-related arguments
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Enable LLM-based analysis'
    )
    parser.add_argument(
        '--llm-provider',
        default='local',
        choices=['openai', 'anthropic', 'local'],
        help='LLM provider to use (default: local)'
    )
    parser.add_argument(
        '--llm-model',
        help='Path to LLM model (for local provider) or model name (for API providers)'
    )
    parser.add_argument(
        '--llm-api-key',
        help='API key for LLM provider (not needed for local provider)'
    )
    parser.add_argument(
        '--complexity-threshold',
        type=float,
        default=0.7,
        help='Complexity threshold for LLM analysis (0.0-1.0)'
    )
    
    return parser.parse_args()

def clone_repository(repo_url):
    """Clone a Git repository to a temporary directory."""
    logger.info(f"Cloning repository: {repo_url}")
    temp_dir = tempfile.mkdtemp(prefix="robotics_repo_analyzer_")
    try:
        git.Repo.clone_from(repo_url, temp_dir)
        logger.info(f"Repository cloned to: {temp_dir}")
        return temp_dir
    except git.GitCommandError as e:
        logger.error(f"Failed to clone repository: {e}")
        shutil.rmtree(temp_dir)
        sys.exit(1)

def main():
    """Main entry point for the Robotics Repository Analyzer."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize LLM client if requested
    llm_client = None
    if args.use_llm:
        logger.info("Initializing LLM client...")
        if args.llm_provider == 'local':
            from robotics_repo_analyzer.llm.llama_client import LlamaClient
            llm_client = LlamaClient(model_path=args.llm_model)
            if not llm_client.available:
                logger.warning("Local Llama model not available. Continuing without LLM.")
                llm_client = None
        else:
            from robotics_repo_analyzer.llm.client import LLMClient
            llm_client = LLMClient(
                provider=args.llm_provider,
                api_key=args.llm_api_key,
                model=args.llm_model
            )
            if not llm_client.available:
                logger.warning(f"{args.llm_provider} client not available. Continuing without LLM.")
                llm_client = None
    
    # Determine if the repository is a URL or local path
    repo_path = args.repository
    temp_dir = None
    
    if repo_path.startswith(('http://', 'https://', 'git@')):
        temp_dir = clone_repository(repo_path)
        repo_path = temp_dir
    elif not os.path.isdir(repo_path):
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    try:
        # Scan the repository
        logger.info(f"Analyzing repository: {repo_path}")
        scanner = RepositoryScanner(
            repo_path, 
            use_llm=args.use_llm,
            llm_client=llm_client,
            complexity_threshold=args.complexity_threshold
        )
        scan_results = scanner.scan()
        
        # Fuse information from multiple sources
        fusion = InformationFusion(scan_results)
        fused_data = fusion.fuse()
        
        # Add LLM usage information if used
        if args.use_llm and llm_client:
            fused_data['llm_used'] = True
            fused_data['llm_stats'] = llm_client.get_usage_stats()
        else:
            fused_data['llm_used'] = False
        
        # Generate output
        output_generator = OutputGenerator(fused_data)
        output_path = Path(args.output)
        output_generator.generate(output_path)
        
        logger.info(f"Analysis complete. Output written to: {output_path}")
        
        # Print LLM usage stats if used
        if args.use_llm and llm_client:
            stats = llm_client.get_usage_stats()
            logger.info(f"LLM Usage: {stats['request_count']} requests")
            if 'total_tokens' in stats:
                logger.info(f"Total tokens: {stats['total_tokens']}")
            if 'estimated_cost' in stats:
                logger.info(f"Estimated cost: ${stats['estimated_cost']:.4f}")
    finally:
        # Clean up temporary directory if created
        if temp_dir and os.path.exists(temp_dir):
            logger.debug(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()