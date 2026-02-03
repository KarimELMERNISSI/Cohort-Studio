import importlib.metadata
import sys

def get_installed_distributions():
    dists = []
    for dist in sorted(importlib.metadata.distributions(), key=lambda x: x.metadata['Name'].lower()):
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
    return dists

if __name__ == '__main__':
    dists = get_installed_distributions()
    with open("license_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Package':<30} | {'Version':<15} | {'License'}\n")
        f.write("-" * 100 + "\n")
        for d in dists:
            # truncate license if too long
            lic = d['License'].replace('\n', ' ')[:50]
            f.write(f"{d['Name']:<30} | {d['Version']:<15} | {lic}\n")
    print("License report written to license_report.txt")
