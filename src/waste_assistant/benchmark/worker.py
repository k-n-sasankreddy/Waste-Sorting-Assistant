"""Run a real worker with an optional benchmark-only processing delay."""

import os
import time
from pathlib import Path

from waste_assistant.ml.predict import Predictor
from waste_assistant.worker.app import main


if __name__ == "__main__":
    delay_path = Path(os.environ["WASTE_BENCHMARK_DELAY"])
    predict = Predictor.predict

    def delayed(self, image):
        try:
            delay_ms = int(delay_path.read_text(encoding="ascii"))
        except FileNotFoundError:
            delay_ms = 0
        if delay_ms:
            time.sleep(delay_ms / 1000)
        return predict(self, image)

    Predictor.predict = delayed
    main()
