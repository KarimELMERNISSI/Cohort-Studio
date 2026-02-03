import importlib.metadata
import sys
import re

def get_requirements():
    """Parse requirements.txt to get package names."""
    packages = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Extract package name (remove version specifiers)
                # Matches valid package names at start of string
                match = re.match(r'^([A-Za-z0-9_\-\.]+)', line)
                if not match:
                    # simpler split if regex fails or for simple cases like 'pandas==2.2.2'
                    name = re.split(r'[=><~;]', line)[0].strip()
                else:
                    name = match.group(1).strip()
                
                if name:
                    packages.append(name)
    except FileNotFoundError:
        print("requirements.txt not found in parent directory.")
    return sorted(list(set(packages)), key=str.lower)

def get_installed_distributions(required_packages):
    dists = []
    # Create a map of lower-case names to actual names for case-insensitive matching
    installed_map = {d.metadata['Name'].lower(): d for d in importlib.metadata.distributions()}
    
    for req_name in required_packages:
        req_name_lower = req_name.lower()
        if req_name_lower in installed_map:
            dist = installed_map[req_name_lower]
            name = dist.metadata['Name']
            version = dist.version
            license_text = dist.metadata.get('License', 'Unknown')
            
            # refinement for unknown licenses
            if license_text == 'Unknown':
                 classifiers = dist.metadata.get_all('Classifier', [])
                 if classifiers:
                     for c in classifiers:
                         if c.startswith('License ::'):
                             license_text = c.split('::')[-1].strip()
                             break
            
            dists.append({'Name': name, 'Version': version, 'License': license_text})
        else:
             dists.append({'Name': req_name, 'Version': 'Not Installed', 'License': 'Unknown'})
             
    return dists

if __name__ == '__main__':
    required_packages = get_requirements()
    dists = get_installed_distributions(required_packages)
    
    with open("license_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Package':<30} | {'Version':<15} | {'License'}\n")
        f.write("-" * 100 + "\n")
        for d in dists:
            # truncate license if too long
            lic = d['License'].replace('\n', ' ')[:50]
            f.write(f"{d['Name']:<30} | {d['Version']:<15} | {lic}\n")
    print(f"License report written to license_report.txt for {len(dists)} packages.")
