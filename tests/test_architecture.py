import ast
from pathlib import Path

SOURCE_ROOT = Path("src/llm_refinery")
COMMANDS_AND_WORKFLOWS = ("llm_refinery.commands", "llm_refinery.workflows")


def test_dependency_boundaries_do_not_point_toward_cli_or_workflows():
    violations: list[str] = []
    forbidden_by_layer = {
        "core": (
            "llm_refinery.application",
            "llm_refinery.benchmarks",
            "llm_refinery.providers",
            "llm_refinery.storage",
            *COMMANDS_AND_WORKFLOWS,
        ),
        "storage": (
            "llm_refinery.application",
            "llm_refinery.benchmarks",
            "llm_refinery.providers",
            *COMMANDS_AND_WORKFLOWS,
        ),
        "benchmarks": COMMANDS_AND_WORKFLOWS,
        "providers": COMMANDS_AND_WORKFLOWS,
        "application": COMMANDS_AND_WORKFLOWS,
    }

    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT)
        layer = relative.parts[0]
        forbidden = forbidden_by_layer.get(layer, ())
        if not forbidden:
            continue
        for imported in _internal_imports(path):
            if imported.startswith(forbidden):
                violations.append(f"{relative}: {imported}")

    assert violations == []


def _internal_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("llm_refinery"):
                imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(
                alias.name for alias in node.names if alias.name.startswith("llm_refinery")
            )
    return imports
