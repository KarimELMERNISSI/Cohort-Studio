import os
import sys

# Add the current directory to the path so we can import from manage
sys.path.append(os.getcwd())

from manage.file_handling import detect_encoding

# Create a dummy file with some content
test_file = "test_encoding.txt"
with open(test_file, "w", encoding="utf-8") as f:
    f.write("This is a test file with UTF-8 encoding. éàç")

print(f"Testing with {test_file}...")
encoding = detect_encoding(test_file)
print(f"Detected encoding: {encoding}")

# Clean up
os.remove(test_file)

# Test with ISO-8859-1
test_file_iso = "test_encoding_iso.txt"
with open(test_file_iso, "w", encoding="iso-8859-1") as f:
    f.write("Ceci est un fichier de test avec encodage ISO-8859-1. éàç")

print(f"Testing with {test_file_iso}...")
encoding_iso = detect_encoding(test_file_iso)
print(f"Detected encoding: {encoding_iso}")

os.remove(test_file_iso)
