import os
from pathlib import Path


def dump():
    repo_root = Path(__file__).resolve().parent
    targets = [
        repo_root / "project_code_and_structure.txt",
        repo_root / "project_structure_and_code.txt",
    ]

    exclude_dirs = {"venv", ".venv", "__pycache__", ".pytest_cache", ".git", ".gemini", "data"}

    # Collect all .py files relative to repo_root
    py_files = []
    for p in repo_root.rglob("*.py"):
        rel_p = p.relative_to(repo_root)
        if any(part in exclude_dirs for part in rel_p.parts):
            continue
        py_files.append(rel_p)

    py_files.sort()

    def build_tree_dict(paths):
        tree = {}
        for p in paths:
            curr = tree
            for part in p.parts:
                curr = curr.setdefault(part, {})
        return tree

    def format_tree(node, prefix=""):
        items = sorted(node.keys())
        lines = []
        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + item)
            if node[item]:
                extension = "    " if is_last else "│   "
                lines.extend(format_tree(node[item], prefix + extension))
        return lines

    tree = build_tree_dict(py_files)
    tree_lines = ["dynamic_bots/"] + format_tree(tree)

    header = (
        "Project Structure (.py files only):\n"
        "===================================\n"
        + "\n".join(tree_lines)
        + "\n\n\nProject Code:\n"
        "=============\n\n"
    )

    for target in targets:
        with open(target, "w", encoding="utf-8") as out:
            out.write(header)
            for rel_p in py_files:
                file_path = repo_root / rel_p
                out.write("=" * 80 + "\n")
                out.write(f"FILE: {rel_p.as_posix()}\n")
                out.write("-" * 40 + "\n")
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        out.write(infile.read())
                except Exception as e:
                    out.write(f"<Error reading file: {e}>\n")
                out.write("\n\n\n\n")


if __name__ == "__main__":
    dump()
