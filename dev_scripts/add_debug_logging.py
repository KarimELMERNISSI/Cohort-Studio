import os

file_path = 'manage/rag_manager.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_lines = 0

for i, line in enumerate(lines):
    if skip_lines > 0:
        skip_lines -= 1
        continue

    # Target 1: _generate_simple_taxonomy
    if 'taxonomy = json.loads(self._clean_json_response(response.content))' in line and '_generate_simple_taxonomy' in "".join(lines[max(0, i-20):i]):
        indent = line[:line.find('taxonomy')]
        new_lines.append(f'{indent}cleaned_response = self._clean_json_response(response.content)\n')
        new_lines.append(f'{indent}try:\n')
        new_lines.append(f'{indent}    taxonomy = json.loads(cleaned_response)\n')
        new_lines.append(f'{indent}except json.JSONDecodeError as e:\n')
        new_lines.append(f'{indent}    return None, f"JSON Parsing Error: {{str(e)}}. Raw output: {{response.content[:500]}}..."\n')
        continue

    # Target 2: _enrich_with_formulas
    if 'formulas_data = json.loads(self._clean_json_response(response.content))' in line:
        indent = line[:line.find('formulas_data')]
        new_lines.append(f'{indent}cleaned_response = self._clean_json_response(response.content)\n')
        new_lines.append(f'{indent}try:\n')
        new_lines.append(f'{indent}    formulas_data = json.loads(cleaned_response)\n')
        new_lines.append(f'{indent}except json.JSONDecodeError as e:\n')
        new_lines.append(f'{indent}    print(f"JSON Error in formulas: {{e}}. Raw: {{response.content[:200]}}...")\n')
        new_lines.append(f'{indent}    formulas_data = {{}}\n')
        continue

    # Target 3: _contextualize_variables
    if 'context_data = json.loads(self._clean_json_response(response.content))' in line:
        indent = line[:line.find('context_data')]
        new_lines.append(f'{indent}cleaned_response = self._clean_json_response(response.content)\n')
        new_lines.append(f'{indent}try:\n')
        new_lines.append(f'{indent}    context_data = json.loads(cleaned_response)\n')
        new_lines.append(f'{indent}except json.JSONDecodeError as e:\n')
        new_lines.append(f'{indent}    print(f"JSON Error in context: {{e}}. Raw: {{response.content[:200]}}...")\n')
        new_lines.append(f'{indent}    context_data = {{}}\n')
        continue
        
    # Improve generic exception handling in _generate_simple_taxonomy
    if 'return None, str(e)' in line and '_generate_simple_taxonomy' in "".join(lines[max(0, i-20):i]):
         new_lines.append(line.replace('str(e)', 'f"Generation Error: {str(e)}"'))
         continue

    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
