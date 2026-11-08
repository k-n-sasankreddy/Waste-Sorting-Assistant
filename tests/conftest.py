"""Synthetic predictor fixture. This module stays outside the installed package."""

from typing import Literal, Protocol

import pytest

from waste_assistant.shared.schemas import Contract, Material, Score


class FakePrediction(Contract):
    candidate_label: Material
    score: Score
    score_type: Literal["synthetic_test_score"] = "synthetic_test_score"
    model_version: Literal["fake-test-only"] = "fake-test-only"


class TestPredictor(Protocol):
    def predict(self, image_bytes: bytes) -> FakePrediction: ...


class FixedFakePredictor:
    def predict(self, image_bytes: bytes) -> FakePrediction:
        return FakePrediction(candidate_label="plastic", score=0.75)


@pytest.fixture
def fake_predictor() -> TestPredictor:
    return FixedFakePredictor()
