"""
Prompt templates for LLM analysis.
"""

# Prompt for analyzing a function
FUNCTION_ANALYSIS_PROMPT = """
Analyze this robotics function and extract:
1. What actions it performs on the robot
2. What parameters it uses
3. The sequence of operations
4. Any conditional logic that affects the robot's behavior

Function name: {function_name}
File: {file_path}

Code:
```python
{code}
```

Provide your analysis in JSON format with the following structure:
{
    "robot_config": {
        "joints": [],
        "dof": 0,
        "limits": []
    },
    "tasks": {
        "task_name": {
            "steps": [
                {"action": "action_name", "parameters": {}}
            ]
        }
    },
    "parameters": {
        "param_name": "value"
    },
    "purpose": "Description of function purpose"
}
"""

# Prompt for analyzing a class
CLASS_ANALYSIS_PROMPT = """
Analyze this robotics class and extract:
1. What robot components it represents or controls
2. What actions/methods it provides
3. What parameters and state it maintains
4. How it interacts with the robot

Class name: {class_name}
File: {file_path}

Code:
```python
{code}
```

Provide your analysis in JSON format with the following structure:
{
    "robot_config": {
        "joints": [],
        "dof": 0,
        "limits": []
    },
    "components": {
        "component_name": "description"
    },
    "methods": {
        "method_name": "purpose"
    },
    "parameters": {
        "param_name": "value"
    }
}
"""

# Prompt for analyzing a batch of code sections
BATCH_ANALYSIS_PROMPT = """
Analyze the following {num_sections} robotics code sections. For each section, extract:
1. Robot configuration (joints, limits, etc.)
2. Task sequences or actions being performed
3. Important parameters and their values
4. The purpose of this function in the robotics system

{sections_text}

Provide your analysis in JSON format with the following structure:
{{
    "sections": [
        {{
            "section_id": 1,
            "robot_config": {{ 
                "joints": [],
                "dof": 0,
                "limits": []
            }},
            "tasks": {{ 
                "task_name": {
                    "steps": [
                        {"action": "action_name", "parameters": {}}
                    ]
                }
            }},
            "parameters": {{ 
                "param_name": "value"
            }},
            "purpose": "description"
        }}
    ]
}}
"""

# Prompt for analyzing a file
FILE_ANALYSIS_PROMPT = """
Analyze this robotics file and extract:
1. What robot is being controlled
2. What actions are being performed
3. What parameters are being used
4. The overall purpose of this file

File: {file_path}

Code:
```python
{code}
```

Provide your analysis in JSON format with the following structure:
{
    "robot_config": {
        "joints": [],
        "dof": 0,
        "limits": []
    },
    "actions": [
        {"name": "action_name", "description": "what it does"}
    ],
    "parameters": {
        "param_name": "value"
    },
    "purpose": "Description of file purpose"
}
"""