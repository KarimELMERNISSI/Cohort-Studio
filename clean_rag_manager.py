import os

file_path = 'manage/rag_manager.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_next = False

for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
        
    # Remove "Instructions:\n        For each variable, list:" if followed by another "Instructions:"
    if 'Instructions:' in line and 'For each variable, list:' in lines[i+1]:
        # Check if the line after that is "Instructions:"
        if i + 2 < len(lines) and 'Instructions:' in lines[i+2]:
            skip_next = True # Skip the next line too
            continue
            
    # Remove the extra "Return JSON:" block at the end
    if 'Return JSON:' in line and i + 1 < len(lines) and '{' in lines[i+1]:
         # Check if we already have a Return JSON block above
         # This is a bit risky, let's look for the specific empty one
         if i + 2 < len(lines) and '"original_var_name": {{' in lines[i+2]:
             if i + 3 < len(lines) and '}}' in lines[i+3]:
                 # This is the empty block we want to remove
                 # Skip 4 lines
                 # But we need to handle the loop
                 # Let's just not append these lines
                 # We need a way to skip multiple lines
                 pass # We will handle this by checking specific content
    
    # Specific checks for the duplicate content I saw
    if 'For each variable, list:' in line and 'Instructions:' in lines[i-1]:
         if 'Instructions:' in lines[i+1]:
             continue # Skip this line
             
    if 'Instructions:' in line and 'For each variable, list:' in lines[i+1]:
         if 'Instructions:' in lines[i+2]:
             continue # Skip this line

    # Remove the trailing empty JSON block
    if 'Return JSON:' in line:
        # Check if this is the second one
        # The first one has "description": "..."
        # The second one has just "original_var_name": {{ }}
        if i + 2 < len(lines) and '"original_var_name": {{' in lines[i+2]:
            if i + 3 < len(lines) and '            }}' in lines[i+3]:
                # This is likely the empty one
                # But wait, the first one also starts with "original_var_name": {{
                # The difference is the content inside
                if i + 3 < len(lines) and 'description' not in lines[i+3]:
                     # This is the empty one
                     # Skip this line and the next 3
                     # We can't easily skip in a for loop like this without an index variable we control
                     # So let's just mark lines to skip
                     pass

# Let's try a simpler approach: read the whole content and replace the bad blocks
content = "".join(lines)

# Replace the double instructions
bad_instructions = """        Instructions:
        For each variable, list:
        Instructions:
        For each variable, provide:"""
good_instructions = """        Instructions:
        For each variable, provide:"""

content = content.replace(bad_instructions, good_instructions)

# Replace the double JSON return
bad_json = """        Return JSON:
        {{
            "original_var_name": {{
                "description": "...",
                "clinical_usage": "...",
                "category": "...",
                "topic": "...",
                "proxy_variables": ["...", "..."]
            }}
        }}
        
        Return JSON:
        {{
            "original_var_name": {{
            }}"""
            
good_json = """        Return JSON:
        {{
            "original_var_name": {{
                "description": "...",
                "clinical_usage": "...",
                "category": "...",
                "topic": "...",
                "proxy_variables": ["...", "..."]
            }}
        }}"""

content = content.replace(bad_json, good_json)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
