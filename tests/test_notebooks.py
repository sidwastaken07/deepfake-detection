import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            ast.parse(src, filename=f"{path.name}[cell {i}]")
        assert not cell.get("outputs"), f"{path.name} cell {i} has saved outputs"
