"""
Hybrid code analyzer using traditional parsing and LLM.
"""

import logging
import re
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from task_definition.robotics_repo_analyzer.llm.prompts import (
    FUNCTION_ANALYSIS_PROMPT,
    CLASS_ANALYSIS_PROMPT,
    FILE_ANALYSIS_PROMPT
)

logger = logging.getLogger(__name__)

class ComplexityAnalyzer:
    """
    Analyzer for code complexity to determine when to use LLM.
    """
    
    def __init__(self, threshold=0.7):
        """
        Initialize the complexity analyzer.
        
        Args:
            threshold: Complexity threshold for using LLM (0.0-1.0)
        """
        self.threshold = threshold
    
    def calculate_complexity(self, code: str) -> float:
        """
        Calculate complexity score for a code section.
        
        Args:
            code: The code to analyze
            
        Returns:
            Complexity score (0.0-1.0)
        """
        # Factors that indicate complexity:
        # 1. Length of code
        length_score = min(len(code.split('\n')) / 50, 1.0)
        
        # 2. Number of conditionals
        conditionals = len(re.findall(r'\bif\b|\belse\b|\belif\b|\bfor\b|\bwhile\b', code))
        conditional_score = min(conditionals / 10, 1.0)
        
        # 3. Number of function calls
        function_calls = len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\(', code))
        call_score = min(function_calls / 15, 1.0)
        
        # 4. Presence of complex patterns
        complex_patterns = [
            r'lambda', r'\[.+?\s+for\s+.+?\s+in\s+.+?\]', r'for\s+.+?\s+in\s+.+?:\s+for\s+',
            r'try\s*:', r'with\s+', r'yield', r'async', r'await'
        ]
        pattern_score = sum(0.1 for p in complex_patterns if re.search(p, code)) 
        
        # Combine scores
        return (length_score * 0.3 + conditional_score * 0.3 + 
                call_score * 0.3 + pattern_score * 0.1)
    
    def is_complex(self, code: str) -> bool:
        """
        Check if code is complex enough to warrant LLM analysis.
        
        Args:
            code: The code to analyze
            
        Returns:
            True if complex, False otherwise
        """
        return self.calculate_complexity(code) > self.threshold
    
    def check_robotics_relevance(self, code: str) -> float:
        """
        Check how relevant a code section is to robotics.
        
        Args:
            code: The code to analyze
            
        Returns:
            Relevance score (0.0-1.0)
        """
        robotics_keywords = [
            # Movement and control
            r'move', r'position', r'joint', r'angle', r'velocity', r'acceleration',
            r'trajectory', r'path', r'inverse kinematics', r'forward kinematics',
            
            # Hardware
            r'robot', r'arm', r'gripper', r'motor', r'actuator', r'sensor',
            
            # Frameworks
            r'mujoco', r'pybullet', r'ros', r'moveit', r'gazebo',
            
            # Tasks
            r'pick', r'place', r'grasp', r'release', r'manipulation'
        ]
        
        # Count matches
        matches = sum(1 for kw in robotics_keywords if re.search(r'\b' + kw + r'\b', code, re.IGNORECASE))
        
        # Normalize score
        return min(matches / 10, 1.0)


class HybridCodeAnalyzer:
    """
    Hybrid code analyzer using traditional parsing and LLM.
    """
    
    def __init__(self, llm_client=None, complexity_threshold=0.7):
        """
        Initialize the hybrid code analyzer.
        
        Args:
            llm_client: LLM client for complex code analysis
            complexity_threshold: Threshold for using LLM (0.0-1.0)
        """
        self.llm_client = llm_client
        self.complexity_analyzer = ComplexityAnalyzer(threshold=complexity_threshold)
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a file using the hybrid approach.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Analysis results
        """
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {'error': str(e)}
        
        # First use traditional parsing
        traditional_results = self._analyze_with_ast(code, file_path)
        
        # If LLM client is not available, return traditional results
        if not self.llm_client:
            return traditional_results
        
        # Check if the whole file is complex
        file_complexity = self.complexity_analyzer.calculate_complexity(code)
        file_relevance = self.complexity_analyzer.check_robotics_relevance(code)
        
        # If the file is complex and relevant, use LLM for the whole file
        if file_complexity > self.complexity_analyzer.threshold and file_relevance > 0.5:
            try:
                llm_file_results = self._analyze_file_with_llm(code, file_path)
                # Merge with traditional results
                traditional_results['llm_analysis'] = llm_file_results
            except Exception as e:
                logger.error(f"Error in LLM file analysis for {file_path}: {e}")
                traditional_results['llm_error'] = str(e)
        
        # Identify complex functions that need LLM analysis
        complex_functions = self._identify_complex_functions(code, file_path, traditional_results)
        
        # Use LLM for complex functions
        try:
            llm_results = self._analyze_with_llm(file_path, complex_functions)
        except Exception as e:
            logger.error(f"Error in LLM function analysis for {file_path}: {e}")
            llm_results = {'error': str(e)}
        
        # Merge results
        try:
            merged_results = self._merge_results(traditional_results, llm_results)
        except Exception as e:
            logger.error(f"Error merging results for {file_path}: {e}")
            merged_results = traditional_results
            merged_results['merge_error'] = str(e)
        
        return merged_results
    
    def _analyze_with_ast(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Analyze code using AST.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Analysis results
        """
        try:
            tree = ast.parse(code)
            visitor = RoboticsASTVisitor()
            visitor.visit(tree)
            
            return {
                'functions': visitor.functions,
                'classes': visitor.classes,
                'imports': visitor.imports,
                'constants': visitor.constants,
                'function_calls': visitor.function_calls,
                'has_main': visitor.has_main,
                'file_path': str(file_path)
            }
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return {
                'error': f"Syntax error: {e}",
                'file_path': str(file_path)
            }
        except Exception as e:
            logger.error(f"Error analyzing {file_path} with AST: {e}")
            return {
                'error': str(e),
                'file_path': str(file_path)
            }
    
    def _identify_complex_functions(self, code: str, file_path: Path, 
                                   ast_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify complex functions that need LLM analysis.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            ast_results: Results from AST analysis
            
        Returns:
            List of complex functions
        """
        complex_functions = []
        
        # Get functions from AST analysis
        functions = ast_results.get('functions', [])
        
        for func in functions:
            # Extract function code
            func_code = self._extract_function_code(code, func)
            
            if not func_code:
                continue
            
            # Check complexity and relevance
            complexity = self.complexity_analyzer.calculate_complexity(func_code)
            relevance = self.complexity_analyzer.check_robotics_relevance(func_code)
            
            # If complex and relevant, add to list
            if complexity > self.complexity_analyzer.threshold and relevance > 0.5:
                complex_functions.append({
                    'name': func.get('name'),
                    'code': func_code,
                    'lineno': func.get('lineno'),
                    'complexity': complexity,
                    'relevance': relevance
                })
        
        # Get classes from AST analysis
        classes = ast_results.get('classes', [])
        
        for cls in classes:
            # Check if class has robotics-related name
            class_name = cls.get('name', '')
            if not any(kw in class_name.lower() for kw in ['robot', 'arm', 'gripper', 'controller']):
                continue
            
            # Extract class code
            class_code = self._extract_class_code(code, cls)
            
            if not class_code:
                continue
            
            # Check complexity and relevance
            complexity = self.complexity_analyzer.calculate_complexity(class_code)
            relevance = self.complexity_analyzer.check_robotics_relevance(class_code)
            
            # If complex and relevant, add to list
            if complexity > self.complexity_analyzer.threshold and relevance > 0.5:
                complex_functions.append({
                    'name': class_name,
                    'code': class_code,
                    'lineno': cls.get('lineno'),
                    'complexity': complexity,
                    'relevance': relevance,
                    'type': 'class'
                })
        
        # Sort by complexity * relevance
        complex_functions.sort(key=lambda x: x['complexity'] * x['relevance'], reverse=True)
        
        # Limit to top 5 to control costs
        return complex_functions[:5]
    
    def _extract_function_code(self, code: str, func: Dict[str, Any]) -> Optional[str]:
        """
        Extract function code from source.
        
        Args:
            code: The source code
            func: Function information from AST analysis
            
        Returns:
            Function code or None if extraction fails
        """
        try:
            lineno = func.get('lineno')
            if not lineno:
                return None
            
            lines = code.split('\n')
            if lineno > len(lines):
                return None
            
            # Find the function definition line
            func_def_line = lines[lineno - 1]
            if not func_def_line.strip().startswith('def '):
                return None
            
            # Get the indentation of the function definition
            indent_match = re.match(r'^(\s*)', func_def_line)
            if not indent_match:
                return None
            
            base_indent = indent_match.group(1)
            
            # Find the end of the function
            end_line = lineno
            for i in range(lineno, len(lines)):
                # Skip empty lines
                if not lines[i].strip():
                    end_line = i
                    continue
                
                # Check if the line has less indentation than the function
                line_indent_match = re.match(r'^(\s*)', lines[i])
                if line_indent_match and line_indent_match.group(1) < base_indent:
                    end_line = i - 1
                    break
                
                end_line = i
            
            # Extract the function code
            return '\n'.join(lines[lineno - 1:end_line + 1])
        except Exception as e:
            logger.error(f"Error extracting function code: {e}")
            return None
    
    def _extract_class_code(self, code: str, cls: Dict[str, Any]) -> Optional[str]:
        """
        Extract class code from source.
        
        Args:
            code: The source code
            cls: Class information from AST analysis
            
        Returns:
            Class code or None if extraction fails
        """
        try:
            lineno = cls.get('lineno')
            if not lineno:
                return None
            
            lines = code.split('\n')
            if lineno > len(lines):
                return None
            
            # Find the class definition line
            class_def_line = lines[lineno - 1]
            if not class_def_line.strip().startswith('class '):
                return None
            
            # Get the indentation of the class definition
            indent_match = re.match(r'^(\s*)', class_def_line)
            if not indent_match:
                return None
            
            base_indent = indent_match.group(1)
            
            # Find the end of the class
            end_line = lineno
            for i in range(lineno, len(lines)):
                # Skip empty lines
                if not lines[i].strip():
                    end_line = i
                    continue
                
                # Check if the line has less indentation than the class
                line_indent_match = re.match(r'^(\s*)', lines[i])
                if line_indent_match and line_indent_match.group(1) < base_indent:
                    end_line = i - 1
                    break
                
                end_line = i
            
            # Extract the class code
            return '\n'.join(lines[lineno - 1:end_line + 1])
        except Exception as e:
            logger.error(f"Error extracting class code: {e}")
            return None
    
    def _analyze_with_llm(self, file_path: Path, complex_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze complex code sections with LLM.
        
        Args:
            file_path: Path to the file being analyzed
            complex_sections: List of complex code sections
            
        Returns:
            LLM analysis results
        """
        llm_results = {}
        
        for section in complex_sections:
            # Create prompt based on section type
            if section.get('type') == 'class':
                prompt = CLASS_ANALYSIS_PROMPT.format(
                    class_name=section['name'],
                    file_path=file_path,
                    code=section['code']
                )
            else:
                prompt = FUNCTION_ANALYSIS_PROMPT.format(
                    function_name=section['name'],
                    file_path=file_path,
                    code=section['code']
                )
            
            # Create cache key
            cache_key = hashlib.md5(section['code'].encode()).hexdigest()
            
            try:
                # Get LLM response
                response = self.llm_client.analyze_code(prompt, cache_key=cache_key)
                
                # Log the raw response for debugging
                logger.debug(f"Raw LLM response for {section['name']}: {response[:200]}...")
                
                # Parse response
                parsed_response = self._parse_llm_response(response)
                
                # Add metadata
                parsed_response['_meta'] = {
                    'complexity': section['complexity'],
                    'relevance': section['relevance'],
                    'type': section.get('type', 'function'),
                    'lineno': section['lineno']
                }
                
                # Add to results
                llm_results[section['name']] = parsed_response
            except Exception as e:
                logger.error(f"Error in LLM analysis for section {section.get('name', 'unknown')}: {e}")
                llm_results[section.get('name', f"unknown_{len(llm_results)}")] = {
                    'error': str(e),
                    'robot_config': {
                        'joints': [],
                        'dof': 0,
                        'limits': []
                    },
                    'parameters': {},
                    'purpose': f"Error in LLM analysis: {str(e)}",
                    '_meta': {
                        'complexity': section.get('complexity', 0.0),
                        'relevance': section.get('relevance', 0.0),
                        'type': section.get('type', 'function'),
                        'lineno': section.get('lineno', 0)
                    }
                }
        
        return llm_results
    
    def _analyze_file_with_llm(self, code: str, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a whole file with LLM.
        
        Args:
            code: The code to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            LLM analysis results
        """
        try:
            # Truncate code if too long
            max_length = 8000  # Adjust based on LLM context window
            if len(code) > max_length:
                code = code[:max_length] + "\n# ... [code truncated] ..."
            
            # Create prompt
            prompt = FILE_ANALYSIS_PROMPT.format(
                file_path=file_path,
                code=code
            )
            
            # Create cache key
            cache_key = hashlib.md5(code.encode()).hexdigest()
            
            # Get LLM response
            response = self.llm_client.analyze_code(prompt, cache_key=cache_key)
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response for file {file_path}: {response[:200]}...")
            
            # Parse response
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"Error in LLM file analysis for {file_path}: {e}")
            return {
                'error': str(e),
                'robot_config': {
                    'joints': [],
                    'dof': 0,
                    'limits': []
                },
                'actions': [],
                'parameters': {},
                'purpose': f"Error in LLM analysis: {str(e)}"
            }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured data.
        
        Args:
            response: LLM response
            
        Returns:
            Parsed response
        """
        try:
            # Extract JSON from response
            logger.debug(f"Searching for JSON in response: {response[:200]}...")
            
            # First try to find JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.debug(f"Found JSON in code block: {json_str[:200]}...")
            else:
                # Try to find JSON without code blocks - more aggressive pattern
                json_match = re.search(r'(\{[\s\S]*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.debug(f"Found JSON without code block: {json_str[:200]}...")
                else:
                    # If no JSON found, return a default structure
                    logger.warning(f"No JSON found in LLM response: {response[:100]}...")
                    return {
                        'robot_config': {
                            'joints': [],
                            'dof': 0,
                            'limits': []
                        },
                        'actions': [],
                        'parameters': {},
                        'purpose': 'No JSON found in response'
                    }
            
            # Clean up the JSON string
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)  # Remove comments
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # Remove block comments
            
            # Fix common JSON issues
            json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            # Handle unquoted keys (a common LLM error)
            json_str = re.sub(r'(\s*)([a-zA-Z0-9_]+)(\s*):(\s*)', r'\1"\2"\3:\4', json_str)
            
            # Parse JSON
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                
                # Try to fix the JSON
                try:
                    # Remove any trailing commas in the JSON
                    json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                    
                    # Try again
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    # Try a more lenient approach with ast.literal_eval
                    try:
                        # Convert to Python dict syntax
                        py_str = json_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
                        data = ast.literal_eval(py_str)
                        return data
                    except (SyntaxError, ValueError):
                        # If all else fails, try to extract partial information
                        logger.error(f"Failed to parse JSON: {json_str[:100]}...")
                        
                        # Try to extract robot_config if present
                        robot_config_match = re.search(r'"robot_config"\s*:\s*(\{[^}]*\})', json_str)
                        robot_config = {}
                        if robot_config_match:
                            try:
                                robot_config = json.loads(robot_config_match.group(1))
                            except:
                                pass
                        
                        # Try to extract parameters if present
                        parameters_match = re.search(r'"parameters"\s*:\s*(\{[^}]*\})', json_str)
                        parameters = {}
                        if parameters_match:
                            try:
                                parameters = json.loads(parameters_match.group(1))
                            except:
                                pass
                        
                        return {
                            'robot_config': robot_config or {
                                'joints': [],
                                'dof': 0,
                                'limits': []
                            },
                            'actions': [],
                            'parameters': parameters or {},
                            'purpose': 'Partial JSON parsing'
                        }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                'robot_config': {
                    'joints': [],
                    'dof': 0,
                    'limits': []
                },
                'actions': [],
                'parameters': {},
                'purpose': f'Error: {str(e)}'
            }
    
    def _merge_results(self, traditional_results: Dict[str, Any], 
                      llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge traditional and LLM results.
        
        Args:
            traditional_results: Results from traditional parsing
            llm_results: Results from LLM analysis
            
        Returns:
            Merged results
        """
        merged = dict(traditional_results)
        
        # Add LLM analysis results
        merged['llm_analysis'] = llm_results
        
        # Enhance function information with LLM insights
        for func_name, llm_result in llm_results.items():
            # Find the function in traditional results
            for i, func in enumerate(merged.get('functions', [])):
                if func.get('name') == func_name:
                    # Add LLM insights
                    merged['functions'][i]['llm_purpose'] = llm_result.get('purpose')
                    merged['functions'][i]['llm_parameters'] = llm_result.get('parameters')
                    merged['functions'][i]['llm_tasks'] = llm_result.get('tasks')
                    break
            
            # Find the class in traditional results
            for i, cls in enumerate(merged.get('classes', [])):
                if cls.get('name') == func_name:
                    # Add LLM insights
                    merged['classes'][i]['llm_purpose'] = llm_result.get('purpose')
                    merged['classes'][i]['llm_components'] = llm_result.get('components')
                    merged['classes'][i]['llm_methods'] = llm_result.get('methods')
                    merged['classes'][i]['llm_parameters'] = llm_result.get('parameters')
                    break
        
        # Extract robot configuration from LLM results
        robot_configs = []
        for func_name, llm_result in llm_results.items():
            robot_config = llm_result.get('robot_config')
            if robot_config:
                robot_configs.append({
                    'source': func_name,
                    'config': robot_config,
                    'confidence': 0.6  # Lower confidence for LLM-derived config
                })
        
        if robot_configs:
            merged['llm_robot_configs'] = robot_configs
        
        return merged


class RoboticsASTVisitor(ast.NodeVisitor):
    """AST visitor for extracting robotics-related information from Python code."""
    
    def __init__(self):
        self.constants = []
        self.functions = []
        self.classes = []
        self.imports = []
        self.function_calls = []
        self.assignments = []
        self.has_main = False
        self.has_main_function = False
        self.current_function = None
        self.current_class = None
        
    def visit_Import(self, node):
        """Visit an Import node."""
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit an ImportFrom node."""
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit a ClassDef node."""
        class_info = {
            'name': node.name,
            'bases': [base.id if isinstance(base, ast.Name) else '' for base in node.bases],
            'methods': [],
            'lineno': node.lineno
        }
        
        old_class = self.current_class
        self.current_class = class_info
        self.classes.append(class_info)
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        """Visit a FunctionDef node."""
        function_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'calls': [],
            'assignments': [],
            'lineno': node.lineno,
            'class': self.current_class['name'] if self.current_class else None
        }
        
        if node.name == 'main':
            self.has_main_function = True
        
        old_function = self.current_function
        self.current_function = function_info
        
        if self.current_class:
            self.current_class['methods'].append(function_info)
        else:
            self.functions.append(function_info)
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Assign(self, node):
        """Visit an Assign node."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Check if it's a constant (uppercase name)
                if target.id.isupper():
                    value = self._extract_value(node.value)
                    if value is not None:
                        constant_info = {
                            'name': target.id,
                            'value': value,
                            'lineno': node.lineno
                        }
                        self.constants.append(constant_info)
                
                # Record all assignments
                assignment_info = {
                    'target': target.id,
                    'value': self._extract_value(node.value),
                    'lineno': node.lineno,
                    'function': self.current_function['name'] if self.current_function else None,
                    'class': self.current_class['name'] if self.current_class else None
                }
                self.assignments.append(assignment_info)
                
                if self.current_function:
                    self.current_function['assignments'].append(assignment_info)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit a Call node."""
        func_name = self._get_call_name(node.func)
        args = [self._extract_value(arg) for arg in node.args]
        keywords = {kw.arg: self._extract_value(kw.value) for kw in node.keywords if kw.arg}
        
        call_info = {
            'name': func_name,
            'args': args,
            'keywords': keywords,
            'lineno': node.lineno,
            'function': self.current_function['name'] if self.current_function else None,
            'class': self.current_class['name'] if self.current_class else None
        }
        
        self.function_calls.append(call_info)
        
        if self.current_function:
            self.current_function['calls'].append(call_info)
        
        self.generic_visit(node)
    
    def visit_If(self, node):
        """Visit an If node."""
        # Check for if __name__ == "__main__"
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__' and
            len(node.test.ops) == 1 and
            isinstance(node.test.ops[0], ast.Eq) and
            len(node.test.comparators) == 1 and
            isinstance(node.test.comparators[0], ast.Constant) and
            node.test.comparators[0].value == '__main__'):
            self.has_main = True
        
        self.generic_visit(node)
    
    def _get_call_name(self, node):
        """Get the name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        return "unknown"
    
    def _extract_value(self, node):
        """Extract a Python value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._extract_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return {self._extract_value(k): self._extract_value(v) 
                   for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Name):
            return node.id  # Return variable name as string
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers
            value = self._extract_value(node.operand)
            if isinstance(value, (int, float)):
                return -value
        elif isinstance(node, ast.BinOp):
            # Handle simple binary operations
            left = self._extract_value(node.left)
            right = self._extract_value(node.right)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
        return None