#!/usr/bin/env python3
"""
AST Generator for MVP_ONE
=========================

Generates and stores Abstract Syntax Trees (ASTs) from repository code for later analysis.
This module scans repositories, parses Python files into ASTs, and persists them to the filesystem.
"""

import ast
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class ASTNode:
    """Serializable AST node representation"""
    
    def __init__(self, node_type: str, attributes: Dict[str, Any], children: List['ASTNode'] = None):
        self.node_type = node_type
        self.attributes = attributes
        self.children = children or []
        self.lineno = attributes.get('lineno')
        self.col_offset = attributes.get('col_offset')

class ASTGenerator:
    """Generate and store ASTs from repository code"""
    
    def __init__(self, storage_dir: Path = None):
        self.logger = logging.getLogger(f"{__name__}.ASTGenerator")
        self.storage_dir = storage_dir or Path("ast_storage")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different storage formats
        (self.storage_dir / "json").mkdir(exist_ok=True)
        (self.storage_dir / "pickle").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
    
    def generate_repository_ast(self, repo_path: Path) -> Dict[str, Any]:
        """Generate ASTs for all Python files in a repository"""
        repo_path = Path(repo_path).resolve()
        self.logger.info(f"Generating ASTs for repository: {repo_path}")
        
        # Find all Python files
        python_files = list(repo_path.rglob("*.py"))
        self.logger.info(f"Found {len(python_files)} Python files")
        
        repository_ast = {
            'metadata': {
                'repo_path': str(repo_path),
                'generated_at': datetime.now().isoformat(),
                'total_files': len(python_files),
                'generator_version': '1.0.0'
            },
            'files': {}
        }
        
        successful_parses = 0
        failed_parses = 0
        
        for file_path in python_files:
            try:
                relative_path = file_path.relative_to(repo_path)
                file_ast = self._parse_file_to_ast(file_path)
                
                if file_ast:
                    repository_ast['files'][str(relative_path)] = file_ast
                    successful_parses += 1
                    self.logger.debug(f"Successfully parsed: {relative_path}")
                else:
                    failed_parses += 1
                    self.logger.warning(f"Failed to parse: {relative_path}")
                    
            except Exception as e:
                failed_parses += 1
                self.logger.error(f"Error parsing {file_path}: {e}")
        
        repository_ast['metadata']['successful_parses'] = successful_parses
        repository_ast['metadata']['failed_parses'] = failed_parses
        repository_ast['metadata']['success_rate'] = successful_parses / len(python_files) if python_files else 0
        
        self.logger.info(f"AST generation complete: {successful_parses}/{len(python_files)} files parsed successfully")
        return repository_ast
    
    def _parse_file_to_ast(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single Python file to AST"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content, filename=str(file_path))
            
            # Convert to serializable format
            serializable_ast = self._ast_to_dict(tree)
            
            return {
                'file_path': str(file_path),
                'file_size': len(content),
                'line_count': len(content.split('\n')),
                'content_hash': hashlib.md5(content.encode()).hexdigest(),
                'ast': serializable_ast,
                'parsed_at': datetime.now().isoformat()
            }
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _ast_to_dict(self, node) -> Dict[str, Any]:
        """Convert AST node to dictionary representation"""
        if not isinstance(node, ast.AST):
            return node
        
        result = {
            'node_type': type(node).__name__,
            'attributes': {}
        }
        
        # Extract node attributes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                result['attributes'][field] = [self._ast_to_dict(item) for item in value]
            elif isinstance(value, ast.AST):
                result['attributes'][field] = self._ast_to_dict(value)
            else:
                result['attributes'][field] = value
        
        # Add position information if available
        if hasattr(node, 'lineno'):
            result['lineno'] = node.lineno
        if hasattr(node, 'col_offset'):
            result['col_offset'] = node.col_offset
        if hasattr(node, 'end_lineno'):
            result['end_lineno'] = node.end_lineno
        if hasattr(node, 'end_col_offset'):
            result['end_col_offset'] = node.end_col_offset
        
        return result
    
    def save_ast_to_filesystem(self, repository_ast: Dict[str, Any], repo_name: str) -> Dict[str, Path]:
        """Save repository AST to filesystem in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{repo_name}_{timestamp}"
        
        saved_files = {}
        
        # Save as JSON (human-readable)
        json_path = self.storage_dir / "json" / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(repository_ast, f, indent=2, default=str)
        saved_files['json'] = json_path
        self.logger.info(f"Saved JSON AST to: {json_path}")
        
        # Save as Pickle (efficient for Python)
        pickle_path = self.storage_dir / "pickle" / f"{base_filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(repository_ast, f)
        saved_files['pickle'] = pickle_path
        self.logger.info(f"Saved Pickle AST to: {pickle_path}")
        
        # Save metadata summary
        metadata = {
            'repo_name': repo_name,
            'generated_at': repository_ast['metadata']['generated_at'],
            'repo_path': repository_ast['metadata']['repo_path'],
            'total_files': repository_ast['metadata']['total_files'],
            'successful_parses': repository_ast['metadata']['successful_parses'],
            'failed_parses': repository_ast['metadata']['failed_parses'],
            'success_rate': repository_ast['metadata']['success_rate'],
            'json_file': str(json_path),
            'pickle_file': str(pickle_path),
            'file_list': list(repository_ast['files'].keys())
        }
        
        metadata_path = self.storage_dir / "metadata" / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path
        self.logger.info(f"Saved metadata to: {metadata_path}")
        
        return saved_files
    
    def load_ast_from_filesystem(self, ast_file_path: Path, format_type: str = 'json') -> Dict[str, Any]:
        """Load repository AST from filesystem"""
        try:
            if format_type == 'json':
                with open(ast_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif format_type == 'pickle':
                with open(ast_file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to load AST from {ast_file_path}: {e}")
            return None
    
    def list_stored_asts(self) -> List[Dict[str, Any]]:
        """List all stored ASTs with metadata"""
        metadata_dir = self.storage_dir / "metadata"
        stored_asts = []
        
        for metadata_file in metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    stored_asts.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        return sorted(stored_asts, key=lambda x: x['generated_at'], reverse=True)
    
    def get_file_ast(self, repository_ast: Dict[str, Any], file_path: str) -> Optional[Dict[str, Any]]:
        """Extract AST for a specific file from repository AST"""
        return repository_ast['files'].get(file_path)
    
    def extract_functions_from_ast(self, file_ast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all function definitions from a file AST"""
        functions = []
        
        def walk_ast(node):
            if isinstance(node, dict) and node.get('node_type') == 'FunctionDef':
                func_info = {
                    'name': node['attributes'].get('name'),
                    'lineno': node.get('lineno'),
                    'args': self._extract_function_args(node),
                    'docstring': self._extract_docstring(node),
                    'decorators': self._extract_decorators(node),
                    'returns': node['attributes'].get('returns'),
                    'full_node': node
                }
                functions.append(func_info)
            
            # Recursively walk children
            if isinstance(node, dict) and 'attributes' in node:
                for value in node['attributes'].values():
                    if isinstance(value, list):
                        for item in value:
                            walk_ast(item)
                    elif isinstance(value, dict):
                        walk_ast(value)
        
        if file_ast and 'ast' in file_ast:
            walk_ast(file_ast['ast'])
        
        return functions
    
    def _extract_function_args(self, func_node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract function arguments from AST node"""
        args = []
        args_node = func_node['attributes'].get('args', {})
        
        if isinstance(args_node, dict) and 'attributes' in args_node:
            arg_list = args_node['attributes'].get('args', [])
            for arg in arg_list:
                if isinstance(arg, dict) and arg.get('node_type') == 'arg':
                    args.append({
                        'name': arg['attributes'].get('arg'),
                        'annotation': arg['attributes'].get('annotation')
                    })
        
        return args
    
    def _extract_docstring(self, func_node: Dict[str, Any]) -> Optional[str]:
        """Extract docstring from function AST node"""
        body = func_node['attributes'].get('body', [])
        if body and isinstance(body[0], dict):
            first_stmt = body[0]
            if (first_stmt.get('node_type') == 'Expr' and 
                isinstance(first_stmt['attributes'].get('value'), dict) and
                first_stmt['attributes']['value'].get('node_type') == 'Constant'):
                return first_stmt['attributes']['value']['attributes'].get('value')
        return None
    
    def _extract_decorators(self, func_node: Dict[str, Any]) -> List[str]:
        """Extract decorators from function AST node"""
        decorators = []
        decorator_list = func_node['attributes'].get('decorator_list', [])
        
        for decorator in decorator_list:
            if isinstance(decorator, dict):
                if decorator.get('node_type') == 'Name':
                    decorators.append(decorator['attributes'].get('id'))
                elif decorator.get('node_type') == 'Attribute':
                    # Handle @obj.decorator syntax
                    decorators.append('attribute_decorator')
        
        return decorators

def main():
    """Example usage of AST Generator"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = ASTGenerator()
    
    # Example: Generate AST for MVP_Zero directory
    mvp_zero_path = Path("MVP_Zero/MVP_Zero")
    if mvp_zero_path.exists():
        # Generate AST
        repo_ast = generator.generate_repository_ast(mvp_zero_path)
        
        # Save to filesystem
        saved_files = generator.save_ast_to_filesystem(repo_ast, "MVP_Zero")
        
        print(f"AST generated and saved:")
        for format_type, file_path in saved_files.items():
            print(f"  {format_type}: {file_path}")
        
        # Example: Extract functions from a specific file
        if 'task_extractor.py' in repo_ast['files']:
            file_ast = repo_ast['files']['task_extractor.py']
            functions = generator.extract_functions_from_ast(file_ast)
            print(f"\nFound {len(functions)} functions in task_extractor.py:")
            for func in functions[:3]:  # Show first 3
                print(f"  - {func['name']} (line {func['lineno']})")

if __name__ == "__main__":
    main()
