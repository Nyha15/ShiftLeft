# MVP_ONE: AST Generator

Generate and store Abstract Syntax Trees (ASTs) from repository code for later analysis.

## Overview

MVP_ONE provides a comprehensive AST generation system that:
- Scans repositories for Python files
- Parses code into structured AST representations
- Stores ASTs in multiple formats (JSON, Pickle)
- Provides CLI interface for AST management
- Enables efficient code analysis without re-parsing

## Features

- **Repository Scanning**: Automatically finds all Python files in a repository
- **AST Generation**: Converts Python code to structured AST format
- **Multiple Storage Formats**: JSON (human-readable) and Pickle (efficient)
- **Metadata Tracking**: Stores parsing statistics and file information
- **Function Extraction**: Identifies and extracts function definitions
- **CLI Interface**: Command-line tools for generation and inspection

## Usage

### Generate AST for a Repository

```bash
python cli.py generate /path/to/repository --name my_project

Eg: python cli.py generate https://github.com/omar-a-aman/MIRMI_APF_FrankaEmikaPanda_Control_Task --name franka_control
```

### List Stored ASTs

```bash
python cli.py list
```

### Inspect AST File

```bash
python cli.py inspect ast_storage/json/my_project_20240817_094530.json --show-functions
```

## Storage Structure

```
ast_storage/
├── json/           # Human-readable JSON format
├── pickle/         # Binary pickle format (efficient)
└── metadata/       # Summary metadata files
```

## API Usage

```python
from ast_generator import ASTGenerator

# Initialize generator
generator = ASTGenerator()

# Generate AST for repository
repo_ast = generator.generate_repository_ast("/path/to/repo")

# Save to filesystem
saved_files = generator.save_ast_to_filesystem(repo_ast, "project_name")

# Load existing AST
loaded_ast = generator.load_ast_from_filesystem("ast_file.json")

# Extract functions from file
functions = generator.extract_functions_from_ast(file_ast)
```

## AST Format

The generated AST includes:

- **Metadata**: Repository info, parsing statistics, timestamps
- **File-level ASTs**: Complete AST for each Python file
- **Function extraction**: Structured function information
- **Position tracking**: Line numbers and column offsets

## Benefits

1. **Performance**: Parse once, analyze many times
2. **Persistence**: Store ASTs for future reference
3. **Structure**: Work with structured data instead of raw code
4. **Flexibility**: Multiple storage formats for different use cases
5. **Metadata**: Rich information about parsing success and file statistics

## Integration

MVP_ONE generates ASTs that can be consumed by other analysis tools, enabling:
- Static code analysis
- Robotics task extraction
- Code pattern detection
- Dependency analysis
- Code metrics calculation
