from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "src/modules/phoenix_trainer/phoenix_trainer_dpg.py"
QUEUE = ROOT / "src/modules/phoenix_trainer/phoenix_training_queue_standalone.py"
MAIN_WINDOW = ROOT / "src/modules/main_window/main_window.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


def _assert_executable_main(path: Path):
    tree = _tree(path)

    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in tree.body
    )

    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    ]
    assert guards, f"{path.name} must execute main() when run as a script"
    assert any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "main"
        for node in guards[0].body
    )


def test_trainer_script_has_executable_main_guard():
    _assert_executable_main(TRAINER)


def test_queue_script_has_executable_main_guard():
    _assert_executable_main(QUEUE)


def test_queue_launches_real_trainer_worker_process():
    source = QUEUE.read_text(encoding="utf-8")

    assert "QueueManager()" not in source
    assert "subprocess.Popen" in source
    assert '"--auto-start"' in source
    assert '"--auto-exit"' in source


def test_main_window_launches_queue_control_process_not_idle_trainer():
    tree = _tree(MAIN_WINDOW)
    method = _method(tree, "TruScoreMainWindow", "launch_phoenix_trainer")
    method_source = ast.unparse(method)

    assert "phoenix_training_queue_standalone.py" in method_source
    assert "phoenix_trainer_dpg.py" not in method_source
