import os
import subprocess
from pathlib import Path

def dump():
    backend = Path("backend")
    out_path = Path("project_structure_and_code.txt")
    
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("PROJECT STRUCTURE:\n")
        # dump tree
        result = subprocess.run("tree backend /A /F", capture_output=True, text=True, shell=True)
        out.write(result.stdout)
        out.write("\n\n")
        
        # files to dump
        extensions = {".py", ".ini", ".txt", ".jsonl"}
        
        for root, dirs, files in os.walk(backend):
            if "__pycache__" in root: continue
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in extensions:
                    file_path = Path(root) / f
                    out.write("="*80 + "\n\n")
                    out.write(f"FILE: {file_path.relative_to('backend').as_posix()}\n")
                    out.write("-" * 40 + "\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            out.write(infile.read())
                    except Exception as e:
                        out.write(f"<Error reading file: {e}>\n")
                    out.write("\n\n")

if __name__ == "__main__":
    dump()
