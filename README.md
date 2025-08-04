# ShiftLeft

A Python application that analyzes messy, real-world robotics Git repositories and extracts robot specifications and task sequences to generate a task.yaml file.

## Installation

```bash
pip install requirements.txt
```

## Usage

```bash
python -m task_definition.robotics_repo_analyzer.main <repository_url_or_path> --output <task_yaml_name> --filter-tasks
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
