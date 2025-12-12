import os

file_path = 'manage/rag_manager.py'
method_file = 'add_refinement_method.py'

with open(method_file, 'r', encoding='utf-8') as f:
    new_method_code = f.read()

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the last method in the class and append the new one
# We can just append it to the end of the file, assuming the class ends at the end of the file
# But we need to make sure indentation is correct (4 spaces)

# Check indentation of the new method code
# It seems I wrote it with 4 spaces indentation in the previous step
# Let's just append it

with open(file_path, 'a', encoding='utf-8') as f:
    f.write('\n\n')
    f.write(new_method_code)
