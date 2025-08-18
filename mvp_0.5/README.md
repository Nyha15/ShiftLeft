# Robotics Repository Analyzer & CI Pipeline

A comprehensive tool for analyzing robotics repositories and automatically extracting physical and control parameters for CI/CD testing.

## Features

### 🚀 Repository Analysis
- **Robot Kinematics**: Parse URDF, SDF, MJCF, and USD files
- **Task Extraction**: Automatically detect robotics tasks from source code
- **Configuration Discovery**: Find ROS configs, launch files, and parameters
- **Comprehensive Reports**: Generate industry-ready analysis documents

### 🔧 CI Pipeline
- **Repository Cloning**: Automatically clone remote repositories
- **Parameter Extraction**: Extract parameters from all sources (URDF, configs, code)
- **Sensitivity Analysis**: Use SALib for parameter impact assessment
- **Simulation Integration**: Support for MuJoCo, PyBullet, and Gazebo
- **YAML Updates**: Automatically update task YAMLs with sweep definitions
- **Artifact Generation**: Create comprehensive reports and visualizations

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MVP_Zero

# Install dependencies
pip3 install -r requirements.txt
```

## Quick Start

### 1. Analyze Existing Repository

```bash
# Basic analysis
python3 cli.py analyze /path/to/robotics/repo

# With custom output and verbose logging
python3 cli.py analyze /path/to/repo --output my_analysis --verbose
```

### 2. Run CI Pipeline on Remote Repository

```bash
# Basic CI pipeline with default settings
python3 cli.py ci https://github.com/user/robot-repo.git

# Advanced CI pipeline with custom configuration
python3 cli.py ci https://github.com/user/robot-repo.git \
    --engine mujoco \
    --method sobol \
    --samples 200 \
    --output ci_results \
    --verbose
```

## CLI Commands

### Analyze Command
```bash
python3 cli.py analyze <repository_path> [options]

Options:
  --output, -o    Output directory (default: robotics_analysis_results)
  --verbose, -v   Enable verbose logging
```

### CI Pipeline Command
```bash
python3 cli.py ci <repo_url> [options]

Options:
  --output, -o    Output directory (default: ci_pipeline_results)
  --engine, -e    Simulation engine: mujoco, pybullet, gazebo, mock (default: mock)
  --method, -m    Sensitivity method: sobol, oat, latin_hypercube, random (default: sobol)
  --samples, -s   Max samples for analysis (default: 100)
  --seed, -r      Random seed for reproducibility (default: 42)
  --verbose, -v   Enable verbose logging
```

## CI Pipeline Workflow

The CI pipeline automatically performs the following steps:

1. **Repository Cloning**: Clones the target repository locally
2. **Parameter Extraction**: 
   - URDF/SDF/MJCF files → Physical parameters (mass, inertia, friction)
   - ROS configs → Control parameters (gains, thresholds, limits)
   - Source code → Constants and configuration values
   - Task YAMLs → Existing parameter definitions
3. **Sensitivity Analysis**: 
   - Uses SALib for robust parameter impact assessment
   - Supports Sobol, OAT, Latin Hypercube, and random sampling
   - Assigns priority levels (high/medium/low) based on impact
4. **YAML Updates**: 
   - Automatically updates task YAMLs with sweep definitions
   - Preserves existing user configurations
   - Adds sensitivity analysis results
5. **Artifact Generation**: 
   - Parameter summary reports
   - Sensitivity analysis plots
   - Comprehensive CI reports

## Parameter Types Extracted

### Physical Parameters
- **Mass & Inertia**: Link masses, moments of inertia
- **Contact Properties**: Friction coefficients, damping values
- **Joint Limits**: Velocity, acceleration, effort limits
- **Geometric Properties**: Dimensions, offsets, transformations

### Control Parameters
- **PID Gains**: Proportional, integral, derivative gains
- **Thresholds**: Success criteria, failure conditions
- **Timeouts**: Operation time limits, safety thresholds
- **Scaling Factors**: Coordinate transformations, unit conversions

### Sensor Parameters
- **Noise Characteristics**: Standard deviations, bias values
- **Calibration**: Offset values, scaling factors
- **Filtering**: Cutoff frequencies, smoothing parameters

## Sensitivity Analysis Methods

### Sobol Analysis (Recommended)
- **Purpose**: Global sensitivity analysis with variance decomposition
- **Advantage**: Captures parameter interactions and non-linear effects
- **Use Case**: Comprehensive parameter ranking and prioritization

### One-At-a-Time (OAT)
- **Purpose**: Local sensitivity around nominal values
- **Advantage**: Fast execution, easy interpretation
- **Use Case**: Quick parameter screening and validation

### Latin Hypercube
- **Purpose**: Space-filling sampling for parameter exploration
- **Advantage**: Efficient coverage of parameter space
- **Use Case**: Parameter space exploration and visualization

### Random Sampling
- **Purpose**: Monte Carlo-based parameter testing
- **Advantage**: Simple implementation, statistical robustness
- **Use Case**: Baseline testing and validation

## Simulation Engines

### MuJoCo
- **Capabilities**: High-fidelity physics simulation
- **Advantages**: Fast, accurate, industry standard
- **Requirements**: MuJoCo license (free for research)

### PyBullet
- **Capabilities**: Open-source physics simulation
- **Advantages**: Free, Python-native, good performance
- **Requirements**: PyBullet package

### Gazebo
- **Capabilities**: ROS-integrated simulation environment
- **Advantages**: ROS ecosystem integration, extensive models
- **Requirements**: ROS 2 and gazebo_ros packages

### Mock Simulation (Default)
- **Capabilities**: Simplified parameter evaluation
- **Advantages**: No external dependencies, fast execution
- **Use Case**: Development, testing, and CI environments

## Output Structure

```
output_directory/
├── artifacts/
│   ├── parameter_summary.md      # Extracted parameters
│   ├── sensitivity_plot.png      # Sensitivity visualization
│   └── ci_report.md             # Comprehensive CI report
├── updated_yamls/               # YAML files with sweep definitions
└── simulation_results/          # Simulation outputs and logs
```

## Example YAML Output

After running the CI pipeline, your task YAMLs will be automatically updated:

```yaml
parameters:
  joint_friction:
    value: 0.5
    sweep:
      method: sobol
      range: [0.05, 1.0]
      priority: high
      samples: 100
      nominal_value: 0.5
      unit: dimensionless
      description: "Extracted from robot.urdf"
  
  control_gain:
    value: 1.0
    sweep:
      method: oat
      range: [0.75, 1.25]
      priority: medium
      samples: 50
      nominal_value: 1.0
      unit: unknown
      description: "Extracted from control.yaml"
```

## Configuration

### Environment Variables
```bash
# MuJoCo license (if using MuJoCo engine)
export MUJOCO_LICENSE_PATH=/path/to/mjkey.txt

# ROS environment (if using Gazebo engine)
source /opt/ros/humble/setup.bash
```

### Custom Parameter Ranges
The system automatically determines parameter sweep ranges based on:
- **Physical plausibility**: Mass ±20%, friction [0.05, 1.0]
- **Control stability**: Gains ±25%, damping [0.5, 2.0]
- **Sensor characteristics**: Noise 1× to 3× nominal
- **Literature values**: Based on robotics research and testing

## Best Practices

### For Repository Analysis
1. **Organize files logically**: Keep URDFs in `assets/`, tasks in `tasks/`
2. **Use descriptive names**: Parameter names should indicate their purpose
3. **Document units**: Include units in parameter definitions
4. **Version control**: Track parameter changes over time

### For CI Pipeline
1. **Start with mock engine**: Use for development and testing
2. **Graduate to physics engines**: Use MuJoCo/PyBullet for production
3. **Use appropriate methods**: Sobol for comprehensive analysis, OAT for quick checks
4. **Set reasonable sample counts**: Balance accuracy vs. execution time
5. **Review generated ranges**: Ensure physical plausibility

## Troubleshooting

### Common Issues

**SALib not available**
```bash
pip3 install SALib
```

**MuJoCo initialization failed**
```bash
# Check license file
export MUJOCO_LICENSE_PATH=/path/to/mjkey.txt
```

**Repository cloning failed**
```bash
# Check network connectivity and repository access
git ls-remote <repo_url>
```

**Simulation engine not found**
```bash
# Install required packages
pip3 install mujoco pybullet
```

### Debug Mode
```bash
# Enable verbose logging for detailed debugging
python3 cli.py ci <repo_url> --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs with `--verbose` flag
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This tool is designed for robotics research and development. Always validate generated parameter ranges against physical constraints and safety requirements before using in production systems.
