import os
import re
import yaml

def load_env_versions(yml_path):
    with open(yml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    versions = {}
    
    # helper to parse "package=version" or "package==version"
    def add_dep(dep):
        if not isinstance(dep, str):
            return
        # Split by = or == or >= etc
        # Conda often uses =, pip uses ==
        parts = re.split(r'[=<>]', dep)
        if len(parts) >= 2:
            pkg = parts[0].strip()
            # Find the first version-like part
            # e.g. "numpy=1.26.4=py311..." -> 1.26.4
            # e.g. "pandas==2.2.2" -> 2.2.2
            # We want the clean version.
            # In conda "pkg=ver=build", parts[1] is version.
            # In pip "pkg==ver", parts[1] is version.
            ver = parts[1].strip()
            # Clean version (remove build strings if any, though usually split handles it)
            if pkg and ver:
                versions[pkg.lower()] = ver
    
    if 'dependencies' in data:
        for dep in data['dependencies']:
            if isinstance(dep, str):
                add_dep(dep)
            elif isinstance(dep, dict) and 'pip' in dep:
                for pip_dep in dep['pip']:
                    add_dep(pip_dep)
    return versions

def clean_requirements(input_path, env_versions):
    clean_lines = []
    
    # Exclusions for Windows-specific or Conda-specific packages
    exclusions = [
        'pywin32', 'pywinpty', 'pyreadline3', 'winshell', 'wincertstore',
        'conda', 'anaconda', 'mamba', 'menuinst', 'navigator-updater',
        'aext', 'jupyterlab_server', 'jupyter_server', # minimal streamlit might not need full jupyter server
        'mkl', 'intel-openmp', 'vc', 'vs2015_runtime', 'ucrt', 'msys2'
    ]
    
    # Streamlit depends on: altair, blinker, cachetools, click, numpy, packaging, pandas, pillow, protobuf, pyarrow, pydeck, rich, tenacity, toml, tornado, typing-extensions, watchdog, gitpython
    # So we must keep those.
    
    with open(input_path, 'r', encoding='utf-16-le') as f: # pip freeze in PowerShell often outputs UTF-16
        try:
            lines = f.readlines()
        except UnicodeError:
            # Fallback to defaults
             with open(input_path, 'r', encoding='utf-8') as f2:
                 lines = f2.readlines()

    processed_pkgs = set()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Parse package name
        if '@' in line:
            pkg = line.split('@')[0].strip()
        elif '==' in line:
            pkg = line.split('==')[0].strip()
        else:
            continue # Skip weird lines?
            
        pkg_lower = pkg.lower()
        
        # Check exclusions
        is_exclude = False
        for exc in exclusions:
            if pkg_lower.startswith(exc):
                is_exclude = True
                break
        if is_exclude:
            continue
            
        version = None
        
        # Priority 1: Keep version from input if explicit (==)
        if '==' in line:
            # Check if it has a local path comment? No pip freeze just says pkg==ver
            version = line.split('==')[1].strip()
        
        # Priority 2: Look up in environment.yml map
        if not version or '@' in line:
            if pkg_lower in env_versions:
                version = env_versions[pkg_lower]
        
        # Fallback: No version (install latest) or keep as is? 
        # Better to have no version than a file path.
        
        if version:
            # Deduplicate
            if pkg_lower not in processed_pkgs:
                clean_lines.append(f"{pkg}=={version}")
                processed_pkgs.add(pkg_lower)
        else:
            # If we can't find a version, and it was a local file, we risk it not being found.
            # But let's verify if it's a standard one.
            # We output just package name
             if pkg_lower not in processed_pkgs:
                clean_lines.append(f"{pkg}")
                processed_pkgs.add(pkg_lower)

    return clean_lines

if __name__ == "__main__":
    env_versions = load_env_versions('environments/environment_cardiateam-env.yml')
    reqs = clean_requirements('current_requirements.txt', env_versions)
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(sorted(reqs)))
    
    print(f"Generated requirements.txt with {len(reqs)} packages.")
