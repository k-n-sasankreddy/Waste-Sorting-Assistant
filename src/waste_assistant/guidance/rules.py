"""Apply a demonstration ruleset after the model decision."""

from pathlib import Path
from typing import Literal

import yaml

from waste_assistant.shared.schemas import (
    ClassificationFields,
    ClassificationResponse,
    Contract,
    Guidance,
    InferenceResponse,
    Material,
    NonEmpty,
)


class Rule(Contract):
    required_context: tuple[Literal["item_type", "contamination"], ...] = ()
    clarification: NonEmpty | None = None
    message: NonEmpty


class Ruleset(Contract):
    ruleset_id: NonEmpty
    locality_id: NonEmpty
    status: Literal["demonstration_only"]
    rules: dict[Material, Rule]


def load_rules(path: Path) -> Ruleset:
    with path.open(encoding="utf-8") as source:
        return Ruleset.model_validate(yaml.safe_load(source))


def classify(
    fields: ClassificationFields, result: InferenceResponse, ruleset: Ruleset
) -> ClassificationResponse:
    prediction = result.prediction
    needs_review = prediction.score < result.review_threshold
    rule = ruleset.rules.get(prediction.candidate_label)
    if needs_review:
        status = "needs_clarification"
        message = "Try a clearer photo with one visible item, or confirm the material yourself."
    elif rule is None:
        status = "unavailable"
        message = "This demo ruleset has no guidance for the predicted material."
    elif any(getattr(fields, field) is None for field in rule.required_context):
        status = "needs_clarification"
        message = (
            rule.clarification
            or "Provide the missing item details to see demo guidance."
        )
    else:
        status = "available"
        message = rule.message
    return ClassificationResponse(
        request_id=result.request_id,
        status="needs_review" if needs_review else "classified",
        prediction=prediction,
        guidance=Guidance(
            status=status,
            locality_id=ruleset.locality_id,
            ruleset_id=ruleset.ruleset_id,
            ruleset_status=ruleset.status,
            message=message,
        ),
    )
