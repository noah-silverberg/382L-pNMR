#!/usr/bin/env python3
import os
import json
import sys


def process_py_file(filepath, rel_path):
    """Read a .py file and return its content with a header."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        content = f"# Error reading file: {e}\n"
    header = f"\n\n# ===== File: {rel_path} =====\n\n"
    return header + content


def process_ipynb_file(filepath, rel_path):
    """Extract code and markdown cells from a Jupyter Notebook, ignoring outputs."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return f"\n\n# ===== File: {rel_path} =====\n# Error reading notebook: {e}\n"

    header = f"\n\n# ===== File: {rel_path} =====\n\n"
    cells_content = ""

    # Iterate through the notebook cells
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        # Process only code and markdown cells
        if cell_type in ("code", "markdown"):
            cell_header = f"# --- {cell_type.upper()} CELL ---\n"
            cell_text = "".join(cell.get("source", []))
            cells_content += cell_header + cell_text + "\n\n"

    return header + cells_content


def main():
    # The current directory from where the script is run
    base_dir = os.getcwd()
    output_lines = []

    # Get the absolute path of this script to exclude it from processing
    this_script = os.path.abspath(__file__)

    # Walk through all files recursively
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # Skip the script itself
            if os.path.abspath(full_path) == this_script:
                continue
            lower = file.lower()
            rel_path = os.path.relpath(full_path, base_dir)
            if lower.endswith(".py"):
                output_lines.append(process_py_file(full_path, rel_path))
            elif lower.endswith(".ipynb"):
                output_lines.append(process_ipynb_file(full_path, rel_path))

    # Concatenate everything into one file
    output = "\n".join(output_lines)
    output_file = "concatenated_code.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Concatenation complete. Output saved to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
