# Robotics Repository Analyzer MVP

A production-ready tool that analyzes robotics codebases to extract robot specifications and task sequences. This MVP combines the best features from ShiftLeft and the Kinematic Parser to provide comprehensive robotics repository insights.

## 🚀 Features

### Advanced URDF Analysis
- **Deep kinematic parsing** with `urdfpy` integration and XML fallback
- **Structured robot data** extraction (joints, links, DOF, geometry, inertial properties)
- **Automatic end-effector detection** using naming patterns and topology analysis
- **Base link identification** for proper kinematic chain understanding
- **Joint limits and safety constraints** extraction

### Intelligent Task Extraction
- **AST-based Python analysis** for accurate function parsing
- **Robotics relevance filtering** to avoid false positives
- **Rich task metadata** including parameters, dependencies, duration estimates
- **Task complexity scoring** (simple/medium/complex)
- **Action sequence identification** (move, pick, place, rotate, etc.)
- **Confidence scoring** for reliability assessment

### Production-Ready Features
- **Modular architecture** with clear separation of concerns
- **Comprehensive error handling** with graceful degradation
- **Deduplication logic** to prevent repetitive results
- **Structured output** in YAML format
- **Detailed logging** with configurable verbosity
- **CLI interface** for easy integration

## 📁 Project Structure

```
MVP_Zero/
├── README.md              # This comprehensive documentation
├── requirements.txt       # Python dependencies
├── cli.py                # Command-line interface
├── data_models.py        # Structured data classes
├── kinematic_analyzer.py # URDF parsing and robot analysis
├── task_extractor.py     # Python code task extraction
└── main_analyzer.py      # Main orchestration logic
```

## 🛠 Installation

1. **Clone or navigate to the MVP_Zero directory:**
   ```bash
   cd /home/abhishek/llm_robot/MVP_Zero
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install urdfpy for advanced URDF parsing:**
   ```bash
   pip install urdfpy
   ```
   *Note: If urdfpy is not available, the analyzer will use XML fallback parsing.*

## 🎯 Usage

### Basic Usage
```bash
python cli.py /path/to/robotics/repository
```

### Advanced Usage
```bash
python cli.py /path/to/repo --output my_analysis.yaml --verbose
```

### Command Line Options
- `repository`: Path to the robotics repository (required)
- `--output, -o`: Output file path (default: robotics_analysis.yaml)
- `--verbose, -v`: Enable detailed logging

## 📊 Output Format

The analyzer generates a comprehensive YAML file with the following structure:

```yaml
metadata:
  repository_path: "/path/to/repository"
  analysis_time: "2024-01-01T12:00:00"
  confidence: 0.85
  analyzer_version: "1.0.0"

summary:
  total_robots: 2
  total_tasks: 15
  total_config_files: 8
  task_types:
    manipulation: 6
    navigation: 4
    control: 5
  robot_names: ["franka_panda", "ur5"]
  total_dof: 13
  avg_task_complexity: "medium"

robots:
  - name: "franka_panda"
    dof: 7
    base_link: "panda_link0"
    end_effector: "panda_hand"
    joints:
      - name: "panda_joint1"
        joint_type: "revolute"
        limits:
          lower: -2.8973
          upper: 2.8973
        # ... more joint details
    # ... more robot details

tasks:
  - name: "pick_object"
    description: "Pick up an object from the table"
    task_type: "manipulation"
    required_actions: ["move", "pick"]
    complexity: "medium"
    confidence: 0.92
    estimated_duration: 5.2
    # ... more task details

config_files:
  - "/path/to/config.yaml"
  - "/path/to/robot_params.json"
```

## 🏗 Architecture

### Core Components

#### 1. Data Models (`data_models.py`)
Structured dataclasses for:
- `JointInfo`: Joint specifications with limits and constraints
- `LinkInfo`: Link geometry and inertial properties  
- `RobotKinematics`: Complete robot description
- `TaskInfo`: Rich task metadata with confidence scoring
- `AnalysisResult`: Comprehensive analysis output

#### 2. Kinematic Analyzer (`kinematic_analyzer.py`)
- **URDF Discovery**: Finds `.urdf` and `.xacro` files recursively
- **Advanced Parsing**: Uses `urdfpy` with XML fallback
- **Topology Analysis**: Identifies end-effectors and base links
- **Data Extraction**: Geometry, inertial, and constraint information

#### 3. Task Extractor (`task_extractor.py`)
- **AST Parsing**: Proper Python syntax tree analysis
- **Relevance Filtering**: Robotics keyword matching with scoring
- **Pattern Recognition**: Task type classification (manipulation, navigation, etc.)
- **Metadata Extraction**: Parameters, dependencies, complexity estimation
- **Deduplication**: Removes duplicate tasks by confidence

#### 4. Main Analyzer (`main_analyzer.py`)
- **Orchestration**: Coordinates kinematic and task analysis
- **Configuration Discovery**: Finds relevant config files
- **Summary Generation**: Creates comprehensive analysis summary
- **Confidence Calculation**: Weighted confidence scoring
- **Output Management**: YAML serialization and file handling

#### 5. CLI Interface (`cli.py`)
- **Argument Parsing**: User-friendly command-line interface
- **Logging Setup**: Configurable verbosity levels
- **Error Handling**: Graceful failure with informative messages
- **Results Display**: Pretty-printed summary output

## 🔧 Key Improvements Over Original Systems

### vs. ShiftLeft
- ✅ **Advanced URDF parsing** with `urdfpy` integration
- ✅ **Structured data models** instead of loose dictionaries
- ✅ **End-effector detection** for manipulation tasks
- ✅ **Geometry and inertial data** extraction
- ✅ **AST-based task extraction** vs. simple regex
- ✅ **Rich task metadata** with parameters and dependencies
- ✅ **Proper deduplication** to prevent repetitive results
- ✅ **Robotics relevance filtering** to reduce false positives

### vs. Kinematic Parser
- ✅ **Modular architecture** with clear separation
- ✅ **Production-ready CLI** interface
- ✅ **Comprehensive error handling** and logging
- ✅ **Structured output format** for integration
- ✅ **Configuration file discovery** 
- ✅ **Summary generation** with confidence scoring

## 🎯 Use Cases

### Research & Development
- **Repository Analysis**: Understand existing robotics codebases
- **Task Identification**: Discover available robot capabilities
- **Integration Planning**: Assess compatibility and requirements

### System Integration
- **Robot Discovery**: Automatically identify robot specifications
- **Task Mapping**: Map available tasks to system requirements
- **Configuration Analysis**: Find and understand config files

### Documentation & Maintenance
- **Automated Documentation**: Generate robot and task specifications
- **Code Analysis**: Understand complex robotics repositories
- **Migration Planning**: Assess effort for system updates

## 🔍 Example Analysis

For a typical robotics repository, the analyzer might find:

**Robots:**
- Franka Panda (7-DOF manipulator)
- UR5 (6-DOF industrial arm)

**Tasks:**
- `pick_and_place`: Manipulation task with gripper control
- `navigate_to_target`: Navigation with obstacle avoidance
- `visual_servoing`: Perception-based control task

**Configurations:**
- Joint limits and safety parameters
- Controller configurations
- Task-specific parameters

## 🚨 Error Handling

The analyzer includes robust error handling:

- **Missing Dependencies**: Graceful fallback when `urdfpy` unavailable
- **Malformed Files**: Continues analysis despite parsing errors
- **Empty Repositories**: Provides meaningful feedback
- **Permission Issues**: Clear error messages with suggestions

## 🔮 Future Enhancements

Potential areas for expansion:
- **Git Repository Cloning**: Support for remote repositories
- **Multi-language Support**: C++, ROS launch files, etc.
- **Interactive GUI**: Visual analysis interface
- **Database Integration**: Store and query analysis results
- **Comparative Analysis**: Compare multiple repositories

## 🤝 Contributing

This MVP provides a solid foundation for robotics repository analysis. Key areas for contribution:

1. **Additional Parsers**: Support for more file formats
2. **Enhanced Task Detection**: More sophisticated pattern recognition
3. **Visualization**: Graphical representation of results
4. **Performance Optimization**: Faster analysis for large repositories

## 📝 License

This project is part of the LLM Robot research initiative.

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Maintainer:** AI Assistant
