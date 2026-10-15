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



def test_trainer_uses_explicit_precision_control_not_removed_amp_widget():
    source = TRAINER.read_text(encoding="utf-8")

    assert 'dpg.get_value("mixed_precision")' not in source
    assert 'tag="precision"' in source
    assert 'dpg.get_value("precision")' in source


def test_dataset_detection_does_not_default_unknown_data_to_detectron2():
    tree = _tree(TRAINER)
    method = _method(tree, "PhoenixTrainer", "detect_dataset_type")
    method_source = ast.unparse(method)

    assert "Could not infer a supported trainer" in method_source



def test_trainer_streams_base_trainer_logs_into_phoenix_output():
    source = TRAINER.read_text(encoding="utf-8")

    assert 'register_callback("on_log_message"' in source
    assert "Run ID:" in source
    assert "Resolved training config:" in source


def test_queue_exposes_copyable_requested_and_resolved_configs():
    source = QUEUE.read_text(encoding="utf-8")

    assert "Copy Requested Config" in source
    assert "Resolved Training Config:" in source
    assert "Copy Resolved Config" in source
    assert "dpg.set_clipboard_text" in source



def test_queue_has_persistent_operations_monitor():
    source = QUEUE.read_text(encoding="utf-8")

    assert "Operations Monitor" in source
    assert "QueueMonitorStore" in source
    assert "_poll_active_training()" in source
    assert "Training started on dataset" in source
    assert "Training complete on dataset" in source
    assert "Copy Timeline" in source


def test_trainer_publishes_active_run_reference_before_completion():
    source = TRAINER.read_text(encoding="utf-8")

    assert "recorder.write_dataset_reference(dataset_path)" in source
