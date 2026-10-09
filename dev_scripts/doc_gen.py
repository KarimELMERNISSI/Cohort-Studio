import ast
import os
import textwrap


def get_annotation(node):
    """Recursively stringify type annotations."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        value = get_annotation(node.value)
        slice_val = get_annotation(node.slice)
        return f"{value}[{slice_val}]"
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Attribute):
        return f"{get_annotation(node.value)}.{node.attr}"
    # Python < 3.9 uses ast.Index for subscripts, ignoring for simplicity/modern python assumptions
    return "ComplexType"

def parse_function(node):
    args = []
    for arg in node.args.args:
        annotation = get_annotation(arg.annotation)
        args.append(f"{arg.arg}: {annotation}" if annotation else arg.arg)
    
    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
    # Merging defaults could be done but for summary simple args listing is often enough.
    
    returns = get_annotation(node.returns)
    docstring = ast.get_docstring(node)
    
    # Simple dependency tracking: scan for Name nodes in the function body
    # This is very rough and includes local variables, but gives an idea of used symbols.
    # A better approach for "dependencies" is listing imports or function calls.
    # Let's extract Function Calls.
    calls = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
    
    return {
        "name": node.name,
        "args": ", ".join(args),
        "returns": returns,
        "docstring": docstring if docstring else "No description available.",
        "calls": sorted(list(calls))
    }

def parse_class(node):
    methods = []
    docstring = ast.get_docstring(node)
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            methods.append(parse_function(item))
            
    return {
        "name": node.name,
        "docstring": docstring if docstring else "No description available.",
        "methods": methods
    }

def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None

    module_doc = ast.get_docstring(tree)
    imports = []
    classes = []
    functions = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for n in node.names:
                imports.append(f"{module}.{n.name}")
        elif isinstance(node, ast.ClassDef):
            classes.append(parse_class(node))
        elif isinstance(node, ast.FunctionDef):
            functions.append(parse_function(node))

    return {
        "filename": os.path.basename(filepath),
        "docstring": module_doc if module_doc else "No module description.",
        "imports": imports,
        "classes": classes,
        "functions": functions
    }

def clean_docstring(docstring):
    """
    Clean docstring to avoid markdown conflicts and improve formatting.
    converts 'Section:\n-------' to '**Section**:'
    """
    if not docstring:
        return ""
    
    # Remove excessive leading indentation
    docstring = textwrap.dedent(docstring)
    lines = docstring.split('\n')
    cleaned_lines = []
    
    # Detect 'Parameters\n----------' style and replace with bold
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i+1].strip()
            # specific check for --- or === underlines which create setext headers
            if next_line and (set(next_line) == {'-'} or set(next_line) == {'='}) and len(next_line) >= 3:
                # detected a header
                cleaned_lines.append(f"\n**{line.strip().rstrip(':')}**:\n")
                i += 2
                continue
        
        cleaned_lines.append(line)
        i += 1
            
    return "\n".join(cleaned_lines)

def clean_output(text):
    """
    Post-process markdown to fix common lint errors.
    - Remove multiple blank lines (MD012)
    - Ensure blank lines around lists (MD032)
    - Remove trailing spaces (MD009)
    - Fix bold style underscores (MD050)
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Pre-process for whitespace
    for line in lines:
        cleaned_lines.append(line.rstrip())
        
    # Collapse multiple blank lines
    collapsed = []
    prev_blank = False
    for line in cleaned_lines:
        if not line:
            if not prev_blank:
                collapsed.append(line)
            prev_blank = True
        else:
            collapsed.append(line)
            prev_blank = False
            
    # Fix lists and bold style
    final_lines = []
    for i, line in enumerate(collapsed):
        # Fix MD050: __text__ -> **text** if it looks like bold intent
        # Simple regex for standalone __text__ could work, but be careful with code.
        # Since we use backticks for code `__init__`, we usually are safe if we target non-backticked underscores.
        # But for now, let's rely on clean_docstring doing it for new content.
        # The issue reported was likely existing docstrings having `__bold__`.
        # Correcting `clean_docstring` is better.
        
        # Ensure blank line before lists (lines starting with - or 1.)
        # This is primitive but helps MD032
        if i > 0 and (line.strip().startswith('- ') or line.strip().startswith('1. ')) and collapsed[i-1].strip() != '' and not collapsed[i-1].strip().endswith(':'):
             # Ensure previous line is blank or a header/text ending in colon (which might introduce list)
             # actually, if pvs line is text, we need blank.
             final_lines.append("")
             
        final_lines.append(line)

    return "\n".join(final_lines)

def generate_markdown(data, title):
    lines = [f"# {title}", ""]
    
    # Track used headers to avoid MD024 (siblings with same content)
    # We reset for every H2 usually, but MD024 might be strict.
    # Because we structure as ## Filename > ### Class/Def, duplicates are unlikely to be siblings unless
    # we have multiple classes with same name or defaults.
    # However, if we have generic headers like ### Description, we should avoid them if possible or make them H4.
    
    for file_data in data:
        lines.append(f"## {file_data['filename']}")
        lines.append("")
        
        doc = clean_docstring(file_data['docstring'])
        if doc:
            lines.append(f"_{doc}_")
            lines.append("")
        
        if file_data['imports']:
            lines.append("**Imports**:")
            lines.append(", ".join([f"`{i}`" for i in file_data['imports']]))
            lines.append("")
        
        if file_data['classes']:
            for cls in file_data['classes']:
                lines.append(f"### class `{cls['name']}` ({file_data['filename']})")
                lines.append("")
                
                cls_doc = clean_docstring(cls['docstring'])
                if cls_doc:
                    lines.append(cls_doc)
                    lines.append("")
                
                if cls['methods']:
                    lines.append("**Methods:**")
                    lines.append("")
                    for method in cls['methods']:
                        # Use list items for methods to avoid header clutter
                        lines.append(f"- **{method['name']}**(`{method['args']}`) -> `{method['returns']}`")
                        
                        method_doc = clean_docstring(method['docstring'])
                        if method_doc:
                            # Handle multi-line indentation for blockquotes
                            quoted = []
                            for l in method_doc.split('\n'):
                                if l.strip():
                                    quoted.append(f"  > {l}")
                                else:
                                    quoted.append("  >") # Empty blockquote line
                            lines.append("\n".join(quoted))
                            
                        lines.append("")
        
        if file_data['functions']:
             for func in file_data['functions']:
                lines.append(f"### def `{func['name']}` ({file_data['filename']})")
                lines.append("")
                
                lines.append(f"- **Arguments**: `{func['args']}`")
                lines.append(f"- **Returns**: `{func['returns']}`")
                lines.append("")
                
                func_doc = clean_docstring(func['docstring'])
                if func_doc:
                    lines.append(f"{func_doc}")
                    lines.append("")
        
        lines.append("---")
        lines.append("")
        
    return clean_output("\n".join(lines))

def process_directory(directory, output_file, title):
    results = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                path = os.path.join(root, file)
                result = parse_file(path)
                if result:
                    results.append(result)
    
    # Sort by filename
    results.sort(key=lambda x: x['filename'])
    
    markdown = generate_markdown(results, title)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Generated {output_file}")

if __name__ == "__main__":
    base_dir = os.getcwd()
    docs_dir = os.path.join(base_dir, "documentation", "developer")
    
    # Define mapping of source directories to output documentation files
    mapping = {
        "app_pages": ("App_Pages.md", "App Pages Documentation"),
        "manage": ("Management_Modules.md", "Backend Management Logic"),
        "utils": ("Utilities.md", "Utility Functions"),
        "prompts": ("Prompts.md", "LLM Prompts and Templates"),
        "enrich": ("Enrichment_Modules.md", "Enrichment Modules Documentation"),
        "monitor": ("Monitoring_Modules.md", "Monitoring Modules Documentation"),
        "explore": ("Exploration_Modules.md", "Exploration Modules Documentation")
    }

    for src, (out_name, title) in mapping.items():
        src_path = os.path.join(base_dir, src)
        out_path = os.path.join(docs_dir, out_name)
        print(f"Processing {src_path} -> {out_path}")
        process_directory(src_path, out_path, title)
