import pytest

from modules.phoenix_trainer.training_runtime import (
    _configure_deterministic_attention,
    accumulation_divisor,
    infer_metric_mode,
    metric_improved,
    normalize_precision_mode,
    select_monitor_metric,
    should_optimizer_step,
)


def test_partial_accumulation_group_uses_remainder_divisor():
    # Five batches with accumulation=2 => groups [2, 2, 1].
    assert [accumulation_divisor(i, 5, 2) for i in range(5)] == [2, 2, 2, 2, 1]
    assert [should_optimizer_step(i, 5, 2) for i in range(5)] == [
        False,
        True,
        False,
        True,
        True,
    ]


def test_loss_metrics_are_minimized():
    assert infer_metric_mode("val_loss") == "min"
    assert metric_improved(0.25, 0.40, "min")
    assert not metric_improved(0.50, 0.40, "min")


def test_accuracy_metrics_are_maximized():
    assert infer_metric_mode("accuracy") == "max"
    assert metric_improved(91.0, 90.0, "max")
    assert not metric_improved(89.0, 90.0, "max")


def test_monitor_selection_prefers_task_metric_over_loss():
    name, value, mode = select_monitor_metric(
        {"val_loss": 0.2, "accuracy": 93.5},
        {},
    )
    assert (name, value, mode) == ("accuracy", 93.5, "max")


def test_monitor_selection_respects_explicit_loss_override():
    name, value, mode = select_monitor_metric(
        {"val_loss": 0.2, "accuracy": 93.5},
        {"monitor_metric": "val_loss"},
    )
    assert (name, value, mode) == ("val_loss", 0.2, "min")


def test_missing_configured_monitor_metric_fails_explicitly():
    with pytest.raises(KeyError):
        select_monitor_metric(
            {"val_loss": 0.2},
            {"monitor_metric": "mAP"},
        )


def test_precision_normalization_preserves_legacy_amp_toggle():
    assert normalize_precision_mode({"mixed_precision": False}) == "fp32"
    assert normalize_precision_mode({"mixed_precision": True}) == "fp16"
    assert normalize_precision_mode({"precision": "bfloat16"}) == "bf16"


def test_invalid_precision_fails_explicitly():
    with pytest.raises(ValueError):
        normalize_precision_mode({"precision": "tf32"})



def test_deterministic_attention_forces_math_sdpa():
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeCudaBackends:
        @staticmethod
        def enable_flash_sdp(value):
            calls.append(("flash", value))

        @staticmethod
        def enable_mem_efficient_sdp(value):
            calls.append(("mem_efficient", value))

        @staticmethod
        def enable_cudnn_sdp(value):
            calls.append(("cudnn", value))

        @staticmethod
        def enable_math_sdp(value):
            calls.append(("math", value))

    class FakeBackends:
        cuda = FakeCudaBackends()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()

    _configure_deterministic_attention(FakeTorch())

    assert calls == [
        ("flash", False),
        ("mem_efficient", False),
        ("cudnn", False),
        ("math", True),
    ]
