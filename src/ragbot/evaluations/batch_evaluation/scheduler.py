from __future__ import annotations

import logging
import threading

from ragbot.evaluations.batch_evaluation.evaluate_batch import (
    BatchEvaluationConfig,
    run_span_batch,
)

logger = logging.getLogger(__name__)


class BatchEvaluationScheduler:
    """Runs batch evaluations on a daemon thread at a fixed interval."""

    def __init__(
        self,
        *,
        config: BatchEvaluationConfig,
        interval_seconds: int = 3600,
    ) -> None:
        self.config = config
        self.interval_seconds = max(interval_seconds, 1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="batch-evaluation-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                result = run_span_batch(self.config)
                logger.info(
                    "Batch evaluation cron completed: spans=%s evaluated=%s annotations=%s",
                    result.span_count,
                    len(result.evaluation_df),
                    result.annotation_count,
                )
            except Exception:
                logger.exception("Batch evaluation cron failed")
