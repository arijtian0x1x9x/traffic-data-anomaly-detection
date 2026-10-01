"""Clear notebook execution state before committing source notebooks."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        notebook.metadata = {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
        for cell in notebook.cells:
            cell.metadata = {}
            cell.pop("attachments", None)
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
        nbformat.write(notebook, path)
        print(f"Cleared {path.name}")


if __name__ == "__main__":
    main()
