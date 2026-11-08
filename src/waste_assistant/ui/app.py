"""Upload one waste image and display the gateway's decision."""

import math
import os

import httpx
import streamlit as st
from pydantic import ValidationError

from waste_assistant.shared.config import load_settings
from waste_assistant.shared.schemas import ClassificationResponse, ErrorResponse


def main() -> None:
    st.set_page_config(page_title="Waste Sorting Assistant", page_icon="♻️")
    st.title("Waste Sorting Assistant")
    st.write("Upload a clear photo of one item to identify its likely material.")
    st.info(
        "Demonstration locality only. Check your local collection rules before disposal."
    )

    try:
        settings = load_settings(os.environ.get("WASTE_CONFIG", "config/dev.yaml"))
    except (OSError, ValueError):
        st.error("The app could not load its configuration.")
        return

    with st.form("classification"):
        image = st.file_uploader(
            "Waste item photo",
            type=["jpg", "jpeg", "png", "webp"],
            max_upload_size=math.ceil(settings.limits.max_upload_bytes / (1024 * 1024)),
        )
        st.caption("Locality · Demonstration")
        with st.expander("Optional item details"):
            item_type = st.text_input(
                "Item type", placeholder="For example, a drink bottle", max_chars=128
            )
            contamination = st.selectbox(
                "Food or liquid residue", ["Not specified", "No", "Yes"]
            )
        submitted = st.form_submit_button("Classify photo", type="primary")
    if not submitted:
        return
    if image is None:
        st.error("Choose a photo to classify.")
        return
    if image.size > settings.limits.max_upload_bytes:
        st.error("The photo exceeds the upload limit.")
        return

    fields = {"locality_id": settings.locality_id}
    if item_type.strip():
        fields["item_type"] = item_type.strip()
    if contamination != "Not specified":
        fields["contamination"] = "true" if contamination == "Yes" else "false"
    try:
        with st.spinner("Classifying photo…"):
            reply = httpx.post(
                f"http://127.0.0.1:{settings.gateway_port}/v1/classify",
                data=fields,
                files={"image": (image.name, image.getvalue(), image.type)},
                timeout=settings.limits.request_budget_seconds + 2,
                trust_env=False,
            )
        if reply.status_code != 200:
            error = ErrorResponse.model_validate_json(reply.content)
            st.error(error.message)
            return
        result = ClassificationResponse.model_validate_json(reply.content)
    except httpx.TimeoutException:
        st.error("Classification took too long. Try again.")
        return
    except (httpx.RequestError, ValidationError):
        st.error("The classification gateway is unavailable.")
        return

    st.subheader("Result")
    if result.status == "needs_review":
        st.warning("The model is uncertain about this photo.")
    st.metric("Candidate material", result.prediction.candidate_label.capitalize())
    st.caption(f"Model score · {result.prediction.score:.1%} (uncalibrated)")
    st.subheader("Demo disposal guidance")
    st.write(result.guidance.message)
    if result.guidance.ruleset_id:
        st.caption(f"Ruleset · {result.guidance.ruleset_id}")


if __name__ == "__main__":
    main()
