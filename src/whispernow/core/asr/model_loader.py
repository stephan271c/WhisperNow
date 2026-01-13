"""
Background model loading thread.

Provides async model loading to keep the UI responsive during model initialization.
"""

from PySide6.QtCore import QThread, Signal

from ...utils.logger import get_logger
from .transcriber import EngineState, TranscriptionEngine

logger = get_logger(__name__)


class ModelLoaderThread(QThread):
    """
    Background thread for loading ASR models without blocking the UI.

    Signals:
        finished: Emitted when loading completes (success: bool, message: str)
        progress: Emitted during download progress (value: float 0.0-1.0)
        state_changed: Emitted when engine state changes (state: EngineState, message: str)
    """

    finished = Signal(bool, str)
    progress = Signal(float)
    state_changed = Signal(EngineState, str)

    def __init__(self, model_name: str, use_gpu: bool = True, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._use_gpu = use_gpu
        self._engine: TranscriptionEngine = None

    def run(self):
        logger.info(
            f"Background loading model: {self._model_name} (GPU={self._use_gpu})"
        )

        try:
            self._engine = TranscriptionEngine(
                model_name=self._model_name,
                use_gpu=self._use_gpu,
                on_state_change=self._on_state_change,
                on_download_progress=self._on_progress,
            )

            self._engine.load_model()

            message = f"Model loaded on {self._engine.device.upper()}"
            logger.info(f"Background model loading complete: {message}")
            self.finished.emit(True, message)

        except Exception as e:
            message = f"Error loading model: {e}"
            logger.exception(f"Background model loading error: {e}")
            self.finished.emit(False, message)

    def _on_state_change(self, state: EngineState, message: str):
        self.state_changed.emit(state, message)

    def _on_progress(self, value: float):
        self.progress.emit(value)

    @property
    def engine(self) -> TranscriptionEngine:
        """
        Get the loaded TranscriptionEngine.

        Only valid after the finished signal is emitted with success=True.
        """
        return self._engine
