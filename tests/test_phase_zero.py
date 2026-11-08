from pathlib import Path

import pytest
from pydantic import ValidationError

from waste_assistant.shared.config import load_settings
from waste_assistant.shared.schemas import Prediction

DEV_CONFIG = Path(__file__).resolve().parents[1] / "config" / "dev.yaml"


def test_development_config_loads():
    settings = load_settings(DEV_CONFIG)
    assert settings.locality_id == "demo-locality"
    assert [worker.id for worker in settings.workers] == [
        "worker-1",
        "worker-2",
        "worker-3",
    ]


@pytest.mark.parametrize(
    ("original", "replacement", "error"),
    [
        ("id: worker-2", "id: worker-1", "Worker IDs must be unique"),
        ("status_stale_seconds: 3.0", "status_stale_seconds: 1.0", "stale threshold"),
        (
            "max_queued_image_bytes: 67108864",
            "max_queued_image_bytes: 1",
            "queued-image budget",
        ),
    ],
)
def test_invalid_development_config_is_rejected(tmp_path, original, replacement, error):
    config = tmp_path / "invalid.yaml"
    config.write_text(
        DEV_CONFIG.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match=error):
        load_settings(config)


def test_fake_prediction_is_repeatable_and_cannot_pass_as_real(fake_predictor):
    result = fake_predictor.predict(b"synthetic fixture input")
    assert result == fake_predictor.predict(b"synthetic fixture input")
    assert result.candidate_label == "plastic"
    assert result.score_type == "synthetic_test_score"
    assert result.model_version == "fake-test-only"
    with pytest.raises(ValidationError, match="score_type"):
        Prediction.model_validate(result.model_dump())
