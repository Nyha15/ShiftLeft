# ShiftLeft Task Definition

A production-ready Python application that analyzes messy, real-world robotics Git repositories and extracts robot specifications and task sequences to generate an MVP1-ready task.yaml file.

## Features

- **Discovery-First Information Extraction**: Extracts robot specifications from various sources
- **Multi-Source Information Fusion**: Combines information from multiple sources
- **Framework-Agnostic Pattern Detection**: Works with MuJoCo, PyBullet, ROS, and more
- **Task Organization**: Organizes action sequences into meaningful tasks

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m task_definition.main <repository_url_or_path> --output task.yaml
```

## Examples

Example output files can be found in the `examples/output` directory.

## Project Structure

- `task_definition/`: Main package
  - `main.py`: Entry point
  - `robotics_repo_analyzer/`: Core analyzer module
    - `analyzers/`: File-specific analyzers
    - `detectors/`: Feature detectors
    - `frameworks/`: Framework-specific analyzers
    - `llm/`: LLM integration
    - `utils/`: Utility functions
  - `utils/`: General utilities
