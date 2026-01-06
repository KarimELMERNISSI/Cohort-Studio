import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path_utils import resolve_path

class TestPathResolution(unittest.TestCase):
    
    def test_absolute_path_exists(self):
        # Create a dummy file
        with open("test_file.txt", "w") as f:
            f.write("test")
        
        abs_path = os.path.abspath("test_file.txt")
        self.assertEqual(resolve_path(abs_path), abs_path)
        
        os.remove("test_file.txt")
    
    def test_path_not_exists_resolution(self):
        # Create dummy file in data/
        os.makedirs("data", exist_ok=True)
        with open("data/test_data.csv", "w") as f:
            f.write("data")
            
        # Simulate a Windows path that doesn't exist on this (Linux/Mock) env
        # but the file exists in data/
        fake_windows_path = "C:\\Users\\User\\Documents\\test_data.csv"
        
        # The resolution should find data/test_data.csv
        resolved = resolve_path(fake_windows_path)
        
        # Normalize paths for comparison
        expected = os.path.join("data", "test_data.csv")
        self.assertEqual(os.path.normpath(resolved), os.path.normpath(expected))
        
        # Clean up
        os.remove("data/test_data.csv")

    def test_path_not_found(self):
        self.assertIsNone(resolve_path("non_existent_file.xyz"))

if __name__ == '__main__':
    unittest.main()
