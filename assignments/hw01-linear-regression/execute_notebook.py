"""Execute and validate the notebook using the current uv Python environment."""

from pathlib import Path
import base64

import nbformat
from nbclient import NotebookClient


def main():
    target = Path(__file__).resolve().parent / "assignment1.ipynb"
    notebook = nbformat.read(target, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(target.parent)}}).execute()
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    plots = sum("image/png" in output.get("data", {}) for cell in cells for output in cell.outputs)
    assert plots == 6, f"Expected six executed plots, got {plots}"
    nbformat.validate(notebook)
    nbformat.write(notebook, target)
    output_directory = target.parent / "results"
    output_directory.mkdir(exist_ok=True)
    for marker, filename in [("illustration_x =", "rbf_intuition.png"), ("view_positions =", "rbf_features.png")]:
        cell = next(cell for cell in cells if marker in cell.source)
        output = next(output for output in cell.outputs if "image/png" in output.get("data", {}))
        (output_directory / filename).write_bytes(base64.b64decode(output["data"]["image/png"]))
    print(f"Executed {len(cells)} code cells successfully; saved {plots} plots.")


if __name__ == "__main__":
    main()
