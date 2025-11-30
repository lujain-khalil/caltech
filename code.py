import os
from pathlib import Path

def collect_code_files():
    # Get the directory of this script
    base_dir = Path(__file__).parent
    
    output_lines = []
    
    # Collect .py files from src/ directory
    src_dir = base_dir / "src"
    if src_dir.exists():
        for py_file in sorted(src_dir.rglob("*.py")):
            output_lines.append(f"\n{'='*60}")
            output_lines.append(f"FILE: {py_file.relative_to(base_dir)}")
            output_lines.append('='*60)
            output_lines.append(py_file.read_text())
            
    # Collect .py files from current directory, excluding summary.py
    for py_file in sorted(base_dir.glob("*.py")):
        if py_file.name != "summary.py" and py_file.name != "code.py":
            output_lines.append(f"\n{'='*60}")
            output_lines.append(f"FILE: {py_file.relative_to(base_dir)}")
            output_lines.append('='*60)
            output_lines.append(py_file.read_text())

    # Write to output file
    output_file = base_dir / "collected_code.txt"
    output_file.write_text("\n".join(output_lines))
    print(f"Code collected to: {output_file}")

if __name__ == "__main__":
    collect_code_files()