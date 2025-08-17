#!/usr/bin/env python3
"""
GitHub Repository Handler
=========================

Handles cloning and processing GitHub repositories for AST generation.
"""

import os
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class GitHubHandler:
    """Handle GitHub repository operations"""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        self.logger = logging.getLogger(f"{__name__}.GitHubHandler")
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "mvp_one_repos"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def clone_repository(self, github_url: str, target_dir: Optional[Path] = None) -> Optional[Path]:
        """Clone a GitHub repository to local filesystem"""
        
        # Parse GitHub URL to extract repo info
        repo_info = self._parse_github_url(github_url)
        if not repo_info:
            self.logger.error(f"Invalid GitHub URL: {github_url}")
            return None
        
        # Determine target directory
        if target_dir is None:
            target_dir = self.temp_dir / repo_info['repo_name']
        
        # Remove existing directory if it exists
        if target_dir.exists():
            self.logger.info(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
        
        # Clone repository
        try:
            self.logger.info(f"Cloning {github_url} to {target_dir}")
            
            # Use git clone command
            cmd = ['git', 'clone', github_url, str(target_dir)]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"Git clone failed: {result.stderr}")
                return None
            
            self.logger.info(f"Successfully cloned repository to {target_dir}")
            return target_dir
            
        except subprocess.TimeoutExpired:
            self.logger.error("Git clone timed out after 5 minutes")
            return None
        except FileNotFoundError:
            self.logger.error("Git command not found. Please install Git.")
            return None
        except Exception as e:
            self.logger.error(f"Error cloning repository: {e}")
            return None
    
    def _parse_github_url(self, github_url: str) -> Optional[Dict[str, str]]:
        """Parse GitHub URL to extract owner and repository name"""
        try:
            parsed = urlparse(github_url)
            
            # Handle different GitHub URL formats
            if parsed.netloc == 'github.com':
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) >= 2:
                    owner = path_parts[0]
                    repo_name = path_parts[1]
                    
                    # Remove .git suffix if present
                    if repo_name.endswith('.git'):
                        repo_name = repo_name[:-4]
                    
                    return {
                        'owner': owner,
                        'repo_name': repo_name,
                        'full_name': f"{owner}/{repo_name}",
                        'clone_url': f"https://github.com/{owner}/{repo_name}.git"
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing GitHub URL: {e}")
            return None
    
    def get_repository_info(self, repo_path: Path) -> Dict[str, Any]:
        """Extract repository information from cloned repo"""
        info = {
            'path': str(repo_path),
            'name': repo_path.name,
            'size_mb': 0,
            'python_files': 0,
            'total_files': 0,
            'has_readme': False,
            'has_requirements': False,
            'has_setup_py': False,
            'directories': []
        }
        
        try:
            # Calculate repository size and file counts
            total_size = 0
            python_files = 0
            total_files = 0
            
            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    try:
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        
                        if file_path.suffix == '.py':
                            python_files += 1
                    except (OSError, PermissionError):
                        continue
            
            info['size_mb'] = round(total_size / (1024 * 1024), 2)
            info['python_files'] = python_files
            info['total_files'] = total_files
            
            # Check for common files
            info['has_readme'] = any(
                (repo_path / name).exists() 
                for name in ['README.md', 'README.rst', 'README.txt', 'readme.md']
            )
            
            info['has_requirements'] = (repo_path / 'requirements.txt').exists()
            info['has_setup_py'] = (repo_path / 'setup.py').exists()
            
            # Get top-level directories
            info['directories'] = [
                d.name for d in repo_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ]
            
        except Exception as e:
            self.logger.warning(f"Error getting repository info: {e}")
        
        return info
    
    def cleanup_repository(self, repo_path: Path) -> bool:
        """Clean up cloned repository with Windows-compatible file handling"""
        try:
            if repo_path.exists():
                # On Windows, Git files can be read-only, so we need to handle permissions
                self._make_writable_recursive(repo_path)
                shutil.rmtree(repo_path)
                self.logger.info(f"Cleaned up repository: {repo_path}")
                return True
        except Exception as e:
            self.logger.error(f"Error cleaning up repository: {e}")
            # Try alternative cleanup method
            return self._force_cleanup_windows(repo_path)
        
        return False
    
    def _make_writable_recursive(self, path: Path):
        """Make all files in directory tree writable (Windows fix)"""
        import stat
        
        try:
            for root, dirs, files in os.walk(path):
                # Make directories writable
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        dir_path.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                    except:
                        pass
                
                # Make files writable
                for file_name in files:
                    file_path = Path(root) / file_name
                    try:
                        file_path.chmod(stat.S_IWRITE | stat.S_IREAD)
                    except:
                        pass
        except Exception as e:
            self.logger.debug(f"Error making files writable: {e}")
    
    def _force_cleanup_windows(self, repo_path: Path) -> bool:
        """Force cleanup using Windows-specific methods"""
        try:
            # Try using rmdir command on Windows
            if os.name == 'nt':
                import subprocess
                result = subprocess.run(
                    ['rmdir', '/s', '/q', str(repo_path)], 
                    shell=True, 
                    capture_output=True
                )
                if result.returncode == 0:
                    self.logger.info(f"Force cleaned up repository using rmdir: {repo_path}")
                    return True
        except Exception as e:
            self.logger.debug(f"Force cleanup failed: {e}")
        
        # If all else fails, just log the issue but don't fail the operation
        self.logger.warning(f"Could not fully clean up repository: {repo_path}")
        self.logger.warning("You may need to manually delete the temporary directory")
        return False
    
    def list_cloned_repositories(self) -> list[Dict[str, Any]]:
        """List all cloned repositories in temp directory"""
        repos = []
        
        try:
            for repo_dir in self.temp_dir.iterdir():
                if repo_dir.is_dir():
                    info = self.get_repository_info(repo_dir)
                    repos.append(info)
        except Exception as e:
            self.logger.error(f"Error listing repositories: {e}")
        
        return repos
    
    def is_git_available(self) -> bool:
        """Check if git command is available"""
        try:
            result = subprocess.run(['git', '--version'], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
