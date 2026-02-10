import importlib.metadata
import re
import sys

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

def get_license(package_name):
    try:
        # standard retrieval
        meta = importlib.metadata.metadata(package_name)
        license_text = meta.get('License', 'Unknown')
        if license_text == 'Unknown':
            # Check classifiers
            classifiers = meta.get_all('Classifier', [])
            for c in classifiers:
                 if c.startswith('License ::'):
                     license_text = c.split('::')[-1].strip()
                     break
        return license_text
    except importlib.metadata.PackageNotFoundError:
        return "Not Installed / Unknown"

def main():
    pkgs = get_requirements()
    
    print("| Package | License |")
    print("| :--- | :--- |")
    
    for pkg in pkgs:
        lic = get_license(pkg)
        # Clean up license text (sometimes it's long or has newlines)
        lic = lic.replace('\n', ' ').replace('|', '/')
        if len(lic) > 50:
             lic = lic[:47] + "..."
        print(f"| **{pkg}** | {lic} |")

if __name__ == "__main__":
    main()
