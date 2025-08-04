"""
LLM-based Task Filter

This module provides functionality to filter tasks using LLM to identify genuine robotics tasks.
"""

import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class LLMTaskFilter:
    """
    Filter tasks using LLM to identify genuine robotics tasks.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the LLM task filter.

        Args:
            llm_client: LLM client for task analysis
        """
        self.llm_client = llm_client

    def filter_tasks(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter tasks using LLM to identify genuine robotics tasks.

        Args:
            tasks: Dictionary of tasks to filter

        Returns:
            Dictionary containing only genuine robotics tasks
        """
        if not self.llm_client:
            logger.warning("No LLM client provided. Skipping task filtering.")
            return tasks

        filtered_tasks = {}

        logger.info(f"Filtering {len(tasks)} tasks using LLM...")

        for task_name, task_data in tasks.items():
            # Skip tasks with no steps
            if not task_data.get('steps'):
                continue

            # Prepare task information for LLM
            task_info = {
                'name': task_name,
                'source': task_data.get('source', 'unknown'),
                'steps': [
                    {
                        'step': step.get('step', i+1),
                        'action': step.get('action', 'unknown'),
                        'parameters': step.get('parameters', {})
                    }
                    for i, step in enumerate(task_data.get('steps', []))
                ]
            }

            # Create LLM prompt
            prompt = self._create_task_filter_prompt(task_info)

            try:
                # Get LLM response - use analyze_code instead of analyze
                response = self.llm_client.analyze_code(prompt)

                # Parse LLM response
                classification, explanation = self._parse_task_classification(response)

                logger.debug(f"Task '{task_name}' classified as {classification}: {explanation}")

                # Keep only genuine tasks
                if classification == 'GENUINE_TASK':
                    filtered_tasks[task_name] = task_data.copy()
                    # Add LLM verification info
                    filtered_tasks[task_name]['llm_verified'] = True
                    filtered_tasks[task_name]['llm_explanation'] = explanation
                    logger.debug(f"Kept task '{task_name}' with {len(task_data.get('steps', []))} steps")
            except Exception as e:
                logger.warning(f"Error filtering task '{task_name}': {e}")
                # Keep the task if LLM filtering fails
                filtered_tasks[task_name] = task_data

        logger.info(f"LLM filtering complete. Kept {len(filtered_tasks)} out of {len(tasks)} tasks.")

        # If no tasks were kept, return the original tasks
        if not filtered_tasks:
            logger.warning("No tasks passed LLM filtering. Returning original tasks.")
            return tasks

        return filtered_tasks

    def _create_task_filter_prompt(self, task_info: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM to classify a task.

        Args:
            task_info: Dictionary containing task information

        Returns:
            Prompt string for the LLM
        """
        steps_text = "\n".join([
            f"- Step {step['step']}: {step['action']} with parameters: {step['parameters']}"
            for step in task_info['steps']
        ])

        prompt = f"""
        You are a robotics expert tasked with identifying genuine robotics tasks from code analysis results.

        Task Information:
        - Name: {task_info['name']}
        - Source: {task_info['source']}
        - Steps:
        {steps_text}

        Based on this information, classify this task as one of the following:
        1. GENUINE_TASK: A meaningful robotics task (e.g., pick_and_place, navigation)
        2. TEST_FUNCTION: A test function that happens to call robotics actions
        3. UTILITY_FUNCTION: A utility function that's part of the framework
        4. UNRELATED: Not related to robotics tasks at all

        Provide your classification and a brief explanation in the following format:
        Classification: [GENUINE_TASK/TEST_FUNCTION/UTILITY_FUNCTION/UNRELATED]
        Explanation: [Your explanation]
        """

        return prompt

    def _parse_task_classification(self, response: str) -> tuple:
        """
        Parse the LLM response to get the task classification.

        Args:
            response: LLM response string

        Returns:
            Tuple of (classification, explanation)
        """
        # Default values
        classification = 'UNRELATED'
        explanation = 'No clear classification provided by LLM.'

        # Extract classification
        if 'Classification:' in response:
            classification_line = response.split('Classification:')[1].split('\n')[0].strip()
            if 'GENUINE_TASK' in classification_line:
                classification = 'GENUINE_TASK'
            elif 'TEST_FUNCTION' in classification_line:
                classification = 'TEST_FUNCTION'
            elif 'UTILITY_FUNCTION' in classification_line:
                classification = 'UTILITY_FUNCTION'
            elif 'UNRELATED' in classification_line:
                classification = 'UNRELATED'

        # Extract explanation
        if 'Explanation:' in response:
            explanation = response.split('Explanation:')[1].strip()

        return classification, explanation