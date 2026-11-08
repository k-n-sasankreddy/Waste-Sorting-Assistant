"""Technical dashboard for the local worker pool."""

import os

import httpx
import streamlit as st

from waste_assistant.shared.config import load_settings

st.set_page_config(page_title="Worker status")
st.title("Worker status")
st.caption(
    "Local multi-process emulation · each worker holds a full copy of the classifier."
)


@st.fragment(run_every="2s")
def show_status() -> None:
    try:
        settings = load_settings(os.environ.get("WASTE_CONFIG", "config/dev.yaml"))
        response = httpx.get(
            f"http://127.0.0.1:{settings.gateway_port}/v1/system/status",
            timeout=2,
            trust_env=False,
        )
        response.raise_for_status()
        status = response.json()
    except (OSError, ValueError, httpx.HTTPError):
        st.error("Gateway status is unavailable.")
        return

    st.write("Policy · " + status["policy"].replace("_", " "))
    requests, images = st.columns(2)
    requests.metric(
        "Outstanding requests", f"{status['outstanding']} / {status['max_outstanding']}"
    )
    images.metric(
        "Queued image bytes",
        f"{status['queued_image_bytes']:,} / {status['max_queued_image_bytes']:,}",
    )
    st.dataframe(
        [
            {
                "Worker": worker["id"],
                "Eligible": worker["available"],
                "Waiting": worker["waiting"],
                "Active": worker["active"],
                "Assigned": worker["assigned"],
                "Completed": worker["completed"],
                "Model version": worker["model_version"],
            }
            for worker in status["workers"]
        ],
        hide_index=True,
    )


show_status()
