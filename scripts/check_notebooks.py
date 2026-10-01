"""Validate source notebooks and reject saved outputs or execution metadata."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No source notebooks found.")
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if set(notebook.metadata) != {"kernelspec", "language_info"}:
            raise SystemExit(f"Unexpected notebook metadata: {path.name}")
        if notebook.metadata.kernelspec.name != "python3":
            raise SystemExit(f"Use the standard project Python kernel: {path.name}")
        if dict(notebook.metadata.language_info) != {"name": "python"}:
            raise SystemExit(f"Clear execution metadata: {path.name}")
        for cell in notebook.cells:
            if cell.metadata or cell.get("attachments"):
                raise SystemExit(f"Clear cell metadata or attachments: {path.name}")
            if cell.cell_type == "code" and (cell.execution_count is not None or cell.outputs):
                raise SystemExit(f"Clear saved notebook output: {path.name}")
    print(f"{len(notebooks)} source notebooks are valid and contain no saved output.")


if __name__ == "__main__":
    main()
