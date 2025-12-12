import os

file_path = 'manage/rag_manager.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_contextualize = False

for i, line in enumerate(lines):
    # Fix the call site
    if 'taxonomy = self._contextualize_variables(taxonomy)' in line:
        new_lines.append(line.replace('taxonomy)', 'taxonomy, columns_info)'))
        continue
        
    # Fix the method definition and body
    if 'def _contextualize_variables(self, taxonomy):' in line:
        new_lines.append('    def _contextualize_variables(self, taxonomy, columns_info):\n')
        in_contextualize = True
        continue
    
    if in_contextualize:
        # We are inside the method, we need to replace the body until the prompt starts or ends
        # Actually, let's just rewrite the whole method body until the next method or end of file
        # But we need to be careful not to delete subsequent methods if any
        
        # Heuristic: The method body seems to have been overwritten with formula logic.
        # We will replace the prompt part.
        
        if 'Role: Medical Statistician & Expert System.' in line:
             new_lines.append('        Role: Medical Expert.\n')
             continue
        if 'Task: Identify standard medical formulas' in line:
             new_lines.append('        Task: Provide clinical context and detailed descriptions for the provided variables.\n')
             continue
        if '1. Formulas where it is a PARAMETER' in line:
             new_lines.append('        Instructions:\n')
             new_lines.append('        For each variable, provide:\n')
             new_lines.append('        1. "description": A clear, medical definition.\n')
             new_lines.append('        2. "clinical_usage": How this variable is used in clinical practice (e.g., diagnosis, monitoring, prognosis).\n')
             new_lines.append('        3. "category": A broad category (e.g., Demographics, Vitals, Lab Test, Comorbidity).\n')
             new_lines.append('        4. "topic": A specific medical topic (e.g., "Cardiovascular Health", "Renal Function", "Diabetes Management").\n')
             new_lines.append('        5. "proxy_variables": A list of potential proxy variables or synonyms often used interchangeably or as surrogates.\n')
             new_lines.append('        \n')
             new_lines.append('        Return JSON:\n')
             new_lines.append('        {{\n')
             new_lines.append('            "original_var_name": {{\n')
             new_lines.append('                "description": "...",\n')
             new_lines.append('                "clinical_usage": "...",\n')
             new_lines.append('                "category": "...",\n')
             new_lines.append('                "topic": "...",\n')
             new_lines.append('                "proxy_variables": ["...", "..."]\n')
             new_lines.append('            }}\n')
             new_lines.append('        }}\n')
             # Skip lines until the end of the original prompt instructions
             continue
        
        # Skip the lines we are replacing
        if '2. Formulas where it is the RESULT' in line or \
           '3. **CRITICAL**: If the variable represents' in line or \
           'Example: "1 g/L = 100 mg/dL"' in line or \
           '"related_formulas": ["Formula 1", "Conversion: 1 g/L = ..."]' in line:
            continue
            
        new_lines.append(line)
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
