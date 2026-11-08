import io

import pytest
from PIL import Image


def test_artifact_round_trip_uses_exported_label_order(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from waste_assistant.ml.artifacts import PRETRAINED, create_model, export_model
    from waste_assistant.ml.artifacts import image_tensor, preprocessing_from_weights
    from waste_assistant.ml.dataset import CLASSES
    from waste_assistant.ml.predict import Predictor

    torch.set_num_threads(1)
    labels = list(reversed(CLASSES))
    model = create_model(len(labels))
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.zero_()
        model.classifier[-1].bias[0] = 2
    preprocessing = preprocessing_from_weights(100_000)
    directory = tmp_path / "artifact"
    artifact = export_model(
        directory,
        model,
        labels=labels,
        preprocessing=preprocessing,
        threshold=0.5,
        training={},
    )
    image = Image.new("RGB", (300, 256), "red")
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    tensor = image_tensor(
        raw.getvalue(), preprocessing.max_pixels, preprocessing.transform()
    )
    assert torch.equal(tensor, PRETRAINED.transforms()(image))

    def fail_download(*args, **kwargs):
        pytest.fail("Artifact loading tried to download weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fail_download)
    predictor = Predictor(directory)
    prediction = predictor.predict(raw.getvalue())
    assert prediction.candidate_label == labels[0]
    assert prediction.model_version == artifact.model_version
    weights = directory / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(directory)


def test_threshold_maximizes_coverage_at_validation_target():
    torch = pytest.importorskip("torch")
    from waste_assistant.ml.metrics import select_threshold

    scores = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.9])
    correct = torch.tensor([False, True, True, True, True])
    result = select_threshold(scores, correct)
    assert result["threshold"] == pytest.approx(0.4)
    assert result["coverage"] == 0.8
    assert result["accepted_accuracy"] == 1.0
