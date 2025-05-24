"""
Temporary Directory Manager for cisTEM tests.

This module provides functionality to track, list, and clean up temporary directories
created during cisTEM test runs. It maintains a log file in the current working directory
that records information about each temporary directory including its path, creation time,
and a unique identifier.
"""

import os
import tempfile
import datetime
import json
import shutil
import argparse
from typing import List, Dict, Optional, Tuple, Union


class TempDirManager:
    """Class for managing temporary directories used in cisTEM tests."""
    
    def __init__(self):
        """Initialize the TempDirManager with a log file in the current working directory."""
        # The log file will be a hidden file in the current working directory
        self.log_file = os.path.join(os.getcwd(), '.cistem_temp_dirs.log')
        
        # Add temporary directory management arguments to an ArgumentParser
        self.temp_management_group = None
        
    def add_arguments(self, parser):
        """
        Add temporary directory management arguments to an ArgumentParser.
        
        Args:
            parser: argparse.ArgumentParser instance
        """
        self.temp_management_group = parser.add_argument_group('Temporary Directory Management')
        self.temp_management_group.add_argument('--list-temp-dirs', action='store_true',
                                          help='List all tracked temporary directories')
        self.temp_management_group.add_argument('--rm-temp-dir', type=int, metavar='INDEX',
                                          help='Remove a specific temporary directory by index')
        self.temp_management_group.add_argument('--rm-all-temp-dirs', action='store_true',
                                          help='Remove all tracked temporary directories')
    
    def create_temp_dir(self, prefix: str = "cistem_test_", dir: str = "/tmp") -> str:
        """
        Create a temporary directory and log its information.
    
        Args:
            prefix: Prefix for the temporary directory name
            dir: Parent directory where the temp directory will be created
    
        Returns:
            Path to the created temporary directory
        """
        # Create the temporary directory
        temp_dir = tempfile.mkdtemp(dir=dir, prefix=prefix)
        
        # Log the created directory
        log_entry = {
            'path': temp_dir,
            'created_at': datetime.datetime.now().isoformat(),
            'key': os.path.basename(temp_dir)  # Use the basename as the unique key
        }
        
        # Read existing log if it exists
        log_entries = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    log_entries = json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupted, start with an empty list
                log_entries = []
        
        # Add the new entry and write back to the log file
        log_entries.append(log_entry)
        with open(self.log_file, 'w') as f:
            json.dump(log_entries, f, indent=2)
        
        return temp_dir
    
    def list_temp_dirs(self) -> List[Dict]:
        """
        List all logged temporary directories.
    
        Returns:
            List of dictionaries containing information about each temp directory
        """
        if not os.path.exists(self.log_file):
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                log_entries = json.load(f)
            
            # Filter out directories that no longer exist
            valid_entries = []
            for entry in log_entries:
                if os.path.exists(entry['path']):
                    valid_entries.append(entry)
            
            # Update the log file if some directories were already removed
            if len(valid_entries) != len(log_entries):
                with open(self.log_file, 'w') as f:
                    json.dump(valid_entries, f, indent=2)
            
            return valid_entries
        except json.JSONDecodeError:
            # If the file is corrupted, return an empty list
            return []
    
    def remove_temp_dir(self, index: int) -> Tuple[bool, str]:
        """
        Remove a temporary directory by its index in the log.
    
        Args:
            index: The index of the directory to remove (0-based)
    
        Returns:
            Tuple of (success, message)
        """
        log_entries = self.list_temp_dirs()
        
        if not log_entries:
            return False, "No temporary directories found in the log."
        
        if index < 0 or index >= len(log_entries):
            return False, f"Invalid index: {index}. Valid range is 0-{len(log_entries)-1}."
        
        entry = log_entries[index]
        path = entry['path']
        
        # Check if the directory exists
        if not os.path.exists(path):
            # Remove from log and return
            log_entries.pop(index)
            with open(self.log_file, 'w') as f:
                json.dump(log_entries, f, indent=2)
            return False, f"Directory {path} no longer exists. Removed from log."
        
        # Remove the directory
        try:
            shutil.rmtree(path)
            # Update the log
            log_entries.pop(index)
            with open(self.log_file, 'w') as f:
                json.dump(log_entries, f, indent=2)
            return True, f"Successfully removed {path}"
        except Exception as e:
            return False, f"Failed to remove {path}: {str(e)}"
    
    def remove_all_temp_dirs(self) -> Tuple[int, int]:
        """
        Remove all temporary directories in the log.
    
        Returns:
            Tuple of (success_count, failure_count)
        """
        log_entries = self.list_temp_dirs()
        success_count = 0
        failure_count = 0
        
        for entry in log_entries:
            path = entry['path']
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    success_count += 1
                except:
                    failure_count += 1
        
        # Clear the log file
        with open(self.log_file, 'w') as f:
            json.dump([], f)
        
        return success_count, failure_count
    
    def print_temp_dirs(self) -> None:
        """
        Print information about all logged temporary directories.
        """
        log_entries = self.list_temp_dirs()
        
        if not log_entries:
            print("No temporary directories found.")
            return
        
        print(f"Found {len(log_entries)} temporary directories:")
        print("-" * 80)
        for i, entry in enumerate(log_entries):
            created_at = datetime.datetime.fromisoformat(entry['created_at'])
            formatted_time = created_at.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{i}] {entry['path']}")
            print(f"    Created: {formatted_time}")
            print(f"    Key: {entry['key']}")
            print("-" * 80)


# Create a singleton instance for backwards compatibility with existing code
_manager = TempDirManager()

# Provide functions that delegate to the instance methods for backwards compatibility
def create_temp_dir(prefix: str = "cistem_test_", dir: str = "/tmp") -> str:
    return _manager.create_temp_dir(prefix, dir)

def list_temp_dirs() -> List[Dict]:
    return _manager.list_temp_dirs()

def remove_temp_dir(index: int) -> Tuple[bool, str]:
    return _manager.remove_temp_dir(index)

def remove_all_temp_dirs() -> Tuple[int, int]:
    return _manager.remove_all_temp_dirs()

def print_temp_dirs() -> None:
    return _manager.print_temp_dirs()