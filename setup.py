from setuptools import setup, find_packages

setup(
    name="shiftleft-task-definition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
        "gitpython",
    ],
    extras_require={
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "task-definition=task_definition.main:main",
        ],
    },
    python_requires=">=3.8",
    description="Analyzes robotics repositories to extract robot specifications and task sequences",
    author="ShiftLeft",
    author_email="info@shiftleft.dev",
    url="https://github.com/shiftleft/task-definition",
)
