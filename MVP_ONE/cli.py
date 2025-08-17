#!/usr/bin/env python3
"""
CLI Interface for AST Generator
===============================

Command-line interface for generating and managing repository ASTs.
"""

import argparse
import logging
import sys
from pathlib import Path
from ast_generator import ASTGenerator
from github_handler import GitHubHandler

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def generate_ast_command(args):
    """Generate AST for a repository (local path or GitHub URL)"""
    repo_input = args.repo_path
    
    # Check if input is a GitHub URL
    if repo_input.startswith('https://github.com/') or repo_input.startswith('git@github.com:'):
        return generate_ast_from_github(args, repo_input)
    else:
        return generate_ast_from_local(args, repo_input)

def generate_ast_from_github(args, github_url):
    """Generate AST from GitHub repository"""
    github_handler = GitHubHandler()
    
    # Check if git is available
    if not github_handler.is_git_available():
        print("Error: Git is not installed or not available in PATH")
        print("Please install Git to clone GitHub repositories")
        return 1
    
    print(f"Cloning GitHub repository: {github_url}")
    
    # Clone repository
    repo_path = github_handler.clone_repository(github_url)
    if not repo_path:
        print("Failed to clone repository")
        return 1
    
    try:
        # Get repository info
        repo_info = github_handler.get_repository_info(repo_path)
        print(f"Repository cloned: {repo_info['name']}")
        print(f"Size: {repo_info['size_mb']} MB")
        print(f"Python files: {repo_info['python_files']}")
        print(f"Total files: {repo_info['total_files']}")
        
        # Generate AST
        return generate_ast_from_local(args, str(repo_path), cleanup_after=True, github_handler=github_handler)
        
    except Exception as e:
        print(f"Error processing repository: {e}")
        github_handler.cleanup_repository(repo_path)
        return 1

def generate_ast_from_local(args, repo_path_str, cleanup_after=False, github_handler=None):
    """Generate AST from local repository path"""
    repo_path = Path(repo_path_str).resolve()
    
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        return 1
    
    if not repo_path.is_dir():
        print(f"Error: Path is not a directory: {repo_path}")
        return 1
    
    # Initialize generator with custom storage directory if provided
    storage_dir = Path(args.output_dir) if args.output_dir else None
    generator = ASTGenerator(storage_dir=storage_dir)
    
    print(f"Generating AST for repository: {repo_path}")
    
    try:
        # Generate AST
        repo_ast = generator.generate_repository_ast(repo_path)
        
        # Determine repository name
        repo_name = args.name or repo_path.name
        
        # Save to filesystem
        saved_files = generator.save_ast_to_filesystem(repo_ast, repo_name)
        
        print(f"\nAST generation complete!")
        print(f"Repository: {repo_path}")
        print(f"Files processed: {repo_ast['metadata']['successful_parses']}/{repo_ast['metadata']['total_files']}")
        print(f"Success rate: {repo_ast['metadata']['success_rate']:.1%}")
        
        print(f"\nSaved files:")
        for format_type, file_path in saved_files.items():
            print(f"  {format_type.upper()}: {file_path}")
        
        return 0
        
    finally:
        # Cleanup cloned repository if requested
        if cleanup_after and github_handler:
            github_handler.cleanup_repository(repo_path)

def list_asts_command(args):
    """List stored ASTs"""
    storage_dir = Path(args.storage_dir) if args.storage_dir else None
    generator = ASTGenerator(storage_dir=storage_dir)
    
    stored_asts = generator.list_stored_asts()
    
    if not stored_asts:
        print("No stored ASTs found.")
        return 0
    
    print(f"Found {len(stored_asts)} stored ASTs:")
    print()
    
    for i, ast_info in enumerate(stored_asts, 1):
        print(f"{i}. {ast_info['repo_name']}")
        print(f"   Generated: {ast_info['generated_at']}")
        print(f"   Files: {ast_info['successful_parses']}/{ast_info['total_files']} ({ast_info['success_rate']:.1%} success)")
        print(f"   Path: {ast_info['repo_path']}")
        print(f"   JSON: {ast_info['json_file']}")
        print()
    
    return 0

def inspect_ast_command(args):
    """Inspect a specific AST file"""
    ast_file = Path(args.ast_file)
    
    if not ast_file.exists():
        print(f"Error: AST file does not exist: {ast_file}")
        return 1
    
    generator = ASTGenerator()
    
    # Determine format from file extension
    format_type = 'pickle' if ast_file.suffix == '.pkl' else 'json'
    
    print(f"Loading AST from: {ast_file}")
    repo_ast = generator.load_ast_from_filesystem(ast_file, format_type)
    
    if not repo_ast:
        print("Failed to load AST file.")
        return 1
    
    # Display metadata
    metadata = repo_ast['metadata']
    print(f"\nRepository: {metadata['repo_path']}")
    print(f"Generated: {metadata['generated_at']}")
    print(f"Files: {metadata['successful_parses']}/{metadata['total_files']} ({metadata['success_rate']:.1%} success)")
    
    # List files
    print(f"\nPython files ({len(repo_ast['files'])}):")
    for file_path in sorted(repo_ast['files'].keys()):
        file_info = repo_ast['files'][file_path]
        print(f"  {file_path} ({file_info['line_count']} lines)")
    
    # Show functions if requested
    if args.show_functions:
        print(f"\nFunctions by file:")
        for file_path in sorted(repo_ast['files'].keys()):
            file_ast = repo_ast['files'][file_path]
            functions = generator.extract_functions_from_ast(file_ast)
            
            if functions:
                print(f"\n  {file_path}:")
                for func in functions:
                    args_str = ', '.join([arg['name'] for arg in func['args']])
                    print(f"    {func['name']}({args_str}) - line {func['lineno']}")
                    if func['docstring']:
                        print(f"      \"{func['docstring'][:60]}{'...' if len(func['docstring']) > 60 else ''}\"")
    
    return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate and manage repository ASTs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate AST for a local repository
  python cli.py generate /path/to/repo --name my_project
  
  # Generate AST from GitHub repository
  python cli.py generate https://github.com/user/repo --name github_project

  # List all stored ASTs
  python cli.py list

  # Inspect a specific AST file
  python cli.py inspect ast_storage/json/my_project_20240817_094530.json --show-functions
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate AST for a repository')
    generate_parser.add_argument('repo_path', help='Path to the repository or GitHub URL')
    generate_parser.add_argument('--name', help='Name for the generated AST (defaults to directory/repo name)')
    generate_parser.add_argument('--output-dir', help='Output directory for AST files (defaults to ast_storage)')
    generate_parser.set_defaults(func=generate_ast_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List stored ASTs')
    list_parser.add_argument('--storage-dir', help='AST storage directory (defaults to ast_storage)')
    list_parser.set_defaults(func=list_asts_command)
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a specific AST file')
    inspect_parser.add_argument('ast_file', help='Path to the AST file')
    inspect_parser.add_argument('--show-functions', action='store_true', help='Show functions in each file')
    inspect_parser.set_defaults(func=inspect_ast_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
