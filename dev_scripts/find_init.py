path = r"C:\Users\208452\AppData\Local\anaconda3\envs\cardiateam_v4\Lib\site-packages\yfiles_graphs_for_streamlit\widget.py"

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    found = False
    with open("init_lines.txt", "w", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            if "def __init__" in line:
                out.write(f"Found init at line {i+1}:\n")
                # Print next 50 lines to see args
                for j in range(i, min(i+50, len(lines))):
                    out.write(f"{j+1}: {lines[j]}")
                found = True
                break
    
    if not found:
        print("Init not found")
    else:
        print("Init found, written to init_lines.txt")
            
except Exception as e:
    print(f"Error: {e}")
