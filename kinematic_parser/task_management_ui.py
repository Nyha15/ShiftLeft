#!/usr/bin/env python3
"""
LLM Robot Task Management UI
============================

A comprehensive graphical user interface for managing, visualizing, and executing
the extracted robot manipulation tasks.

Author: AI Assistant
Date: 2024
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import yaml
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import time

class TaskManagementUI:
    """Main UI class for task management"""
    
    def __init__(self, tasks_dir: str = "organized_tasks"):
        self.tasks_dir = Path(tasks_dir)
        self.tasks = {}
        self.robot_kinematics = {}
        self.task_summary = {}
        
        # Initialize UI
        self.root = tk.Tk()
        self.root.title("LLM Robot Task Management System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Load data
        self.load_data()
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar()
        
        # Create UI components
        self.create_widgets()
        
    def load_data(self):
        """Load task and robot data from YAML files"""
        try:
            # Load task summary
            summary_file = self.tasks_dir / "task_summary.yaml"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.task_summary = yaml.safe_load(f)
            
            # Load robot kinematics
            kinematics_file = self.tasks_dir / "robot_kinematics.yaml"
            if kinematics_file.exists():
                with open(kinematics_file, 'r') as f:
                    self.robot_kinematics = yaml.safe_load(f)
            
            # Load individual tasks
            for task_file in self.tasks_dir.glob("*.yaml"):
                if task_file.name not in ["task_summary.yaml", "robot_kinematics.yaml"]:
                    with open(task_file, 'r') as f:
                        task_data = yaml.safe_load(f)
                        task_name = task_file.stem
                        self.tasks[task_name] = task_data
            
            print(f"✅ Loaded {len(self.tasks)} tasks and robot kinematics")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def create_widgets(self):
        """Create all UI widgets"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_overview_tab()
        self.create_task_browser_tab()
        self.create_robot_info_tab()
        self.create_execution_tab()
        
        # Create status bar
        self.create_status_bar()
    
    def create_overview_tab(self):
        """Create overview tab with summary information"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="📊 Overview")
        
        # Title
        title_label = tk.Label(overview_frame, text="LLM Robot Manipulation System", 
                              font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Summary statistics
        stats_frame = ttk.LabelFrame(overview_frame, text="System Statistics")
        stats_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if self.task_summary:
            summary = self.task_summary.get('summary', {})
            
            stats_text = f"""
            Total Tasks: {summary.get('total_tasks', 0)}
            Robot: {summary.get('robot_name', 'Unknown')} ({summary.get('robot_dof', 0)} DOF)
            Analysis Date: {summary.get('analysis_date', 'Unknown')}
            """
            
            stats_label = tk.Label(stats_frame, text=stats_text, justify=tk.LEFT, bg='#f0f0f0')
            stats_label.pack(padx=10, pady=10)
    
    def create_task_browser_tab(self):
        """Create task browser tab"""
        browser_frame = ttk.Frame(self.notebook)
        self.notebook.add(browser_frame, text="📋 Task Browser")
        
        # Create paned window
        paned = ttk.PanedWindow(browser_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - task list
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Task list
        list_label = ttk.Label(left_frame, text="Available Tasks")
        list_label.pack(pady=5)
        
        # Task listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.task_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.task_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.task_listbox.yview)
        
        # Populate task list
        for task_name in sorted(self.tasks.keys()):
            self.task_listbox.insert(tk.END, task_name)
        
        self.task_listbox.bind('<<ListboxSelect>>', self.on_task_select)
        
        # Right panel - task details
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        details_label = ttk.Label(right_frame, text="Task Details")
        details_label.pack(pady=5)
        
        self.task_details = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=20)
        self.task_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Execute Task", 
                  command=self.execute_selected_task).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Task", 
                  command=self.export_selected_task).pack(side=tk.LEFT, padx=5)
    
    def create_robot_info_tab(self):
        """Create robot information tab"""
        robot_frame = ttk.Frame(self.notebook)
        self.notebook.add(robot_frame, text="🤖 Robot Info")
        
        if not self.robot_kinematics:
            tk.Label(robot_frame, text="No robot kinematics data available", 
                    font=("Arial", 12), bg='#f0f0f0').pack(pady=50)
            return
        
        robot_info = self.robot_kinematics.get('robot_kinematics', {})
        
        # Robot name and description
        title = f"{robot_info.get('name', 'Unknown Robot')}"
        title_label = tk.Label(robot_frame, text=title, font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        desc = robot_info.get('description', 'No description available')
        desc_label = tk.Label(robot_frame, text=desc, wraplength=600, bg='#f0f0f0')
        desc_label.pack(pady=5)
        
        # Specifications
        specs_frame = ttk.LabelFrame(robot_frame, text="Specifications")
        specs_frame.pack(fill=tk.X, padx=20, pady=10)
        
        specs = robot_info.get('specifications', {})
        specs_text = f"""
        Degrees of Freedom: {specs.get('degrees_of_freedom', 'Unknown')}
        Base Link: {specs.get('base_link', 'Unknown')}
        End Effector: {specs.get('end_effector', 'Unknown')}
        """
        
        specs_label = tk.Label(specs_frame, text=specs_text, justify=tk.LEFT, bg='#f0f0f0')
        specs_label.pack(padx=20, pady=20)
    
    def create_execution_tab(self):
        """Create task execution tab"""
        exec_frame = ttk.Frame(self.notebook)
        self.notebook.add(exec_frame, text="▶️ Execution")
        
        # Execution controls
        control_frame = ttk.LabelFrame(exec_frame, text="Execution Controls")
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Task selection for execution
        ttk.Label(control_frame, text="Select Task:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.exec_task_var = tk.StringVar()
        task_combo = ttk.Combobox(control_frame, textvariable=self.exec_task_var, 
                                 values=list(self.tasks.keys()), state="readonly")
        task_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Start Execution", 
                  command=self.start_execution).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Execution", 
                  command=self.stop_execution).pack(side=tk.LEFT, padx=5)
        
        control_frame.columnconfigure(1, weight=1)
        
        # Execution log
        log_frame = ttk.LabelFrame(exec_frame, text="Execution Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.exec_log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.exec_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(exec_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=5)
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=5)
    
    def on_task_select(self, event=None):
        """Handle task selection in listbox"""
        selection = self.task_listbox.curselection()
        if not selection:
            return
        
        task_name = self.task_listbox.get(selection[0])
        task_data = self.tasks.get(task_name, {})
        
        # Display task details
        self.task_details.delete(1.0, tk.END)
        
        # Format task information
        details = f"Task: {task_name}\n"
        details += "=" * 50 + "\n\n"
        
        if 'task_info' in task_data:
            info = task_data['task_info']
            details += f"Description: {info.get('description', 'N/A')}\n"
            details += f"Category: {info.get('category', 'N/A')}\n"
            details += f"Object Type: {info.get('object_type', 'N/A')}\n"
            details += f"Complexity: {info.get('complexity', 'N/A')}\n"
            details += f"Estimated Duration: {info.get('estimated_duration_seconds', 'N/A')} seconds\n\n"
        
        if 'execution' in task_data:
            exec_info = task_data['execution']
            details += "Required Actions:\n"
            for action in exec_info.get('required_actions', []):
                details += f"  • {action}\n"
            details += "\n"
        
        self.task_details.insert(1.0, details)
    
    def execute_selected_task(self):
        """Execute the selected task"""
        selection = self.task_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a task to execute")
            return
        
        task_name = self.task_listbox.get(selection[0])
        self.exec_task_var.set(task_name)
        self.notebook.select(3)  # Switch to execution tab
        self.start_execution()
    
    def export_selected_task(self):
        """Export the selected task"""
        selection = self.task_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a task to export")
            return
        
        task_name = self.task_listbox.get(selection[0])
        
        # Ask for export location
        filename = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialvalue=f"{task_name}.yaml"
        )
        
        if filename:
            try:
                task_file = self.tasks_dir / f"{task_name}.yaml"
                import shutil
                shutil.copy2(task_file, filename)
                messagebox.showinfo("Success", f"Task exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export task: {e}")
    
    def start_execution(self):
        """Start task execution"""
        task_name = self.exec_task_var.get()
        if not task_name:
            messagebox.showwarning("Warning", "Please select a task to execute")
            return
        
        self.log_execution(f"Starting execution of task: {task_name}")
        
        # Start execution in separate thread
        execution_thread = threading.Thread(target=self.execute_task_thread, 
                                           args=(task_name,))
        execution_thread.daemon = True
        execution_thread.start()
    
    def execute_task_thread(self, task_name: str):
        """Execute task in separate thread"""
        try:
            self.status_var.set(f"Executing {task_name}...")
            
            # Simulate task execution
            task_data = self.tasks.get(task_name, {})
            actions = task_data.get('execution', {}).get('required_actions', [])
            
            for i, action in enumerate(actions):
                self.log_execution(f"Executing action: {action}")
                self.progress_var.set((i + 1) / len(actions) * 100)
                time.sleep(1)  # Simulate action execution time
            
            self.log_execution(f"Task {task_name} completed successfully!")
            self.status_var.set("Task completed")
            
        except Exception as e:
            self.log_execution(f"Error executing task: {e}")
            self.status_var.set("Execution failed")
    
    def stop_execution(self):
        """Stop task execution"""
        self.log_execution("Execution stopped by user")
        self.status_var.set("Execution stopped")
        self.progress_var.set(0)
    
    def log_execution(self, message: str):
        """Log execution message"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.exec_log.insert(tk.END, log_message)
        self.exec_log.see(tk.END)
        self.root.update_idletasks()
    
    def run(self):
        """Run the UI"""
        self.root.mainloop()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Robot Task Management UI")
    parser.add_argument("--tasks-dir", "-t", default="organized_tasks", 
                       help="Directory containing organized task files")
    
    args = parser.parse_args()
    
    # Create and run UI
    app = TaskManagementUI(args.tasks_dir)
    app.run()

if __name__ == "__main__":
    main()
