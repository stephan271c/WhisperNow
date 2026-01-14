import os
import tarfile
import tempfile
from typing import Callable, Optional

import requests

from ...utils.logger import get_logger
from .backends import get_models_dir
from .model_registry import get_model_by_id

ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]


class ModelDownloader:
    def __init__(self):
        self._cancelled = False
        self._logger = get_logger(__name__)

    def download(
        self,
        model_id: str,
        on_progress: Optional[ProgressCallback] = None,
        on_status: Optional[StatusCallback] = None,
    ) -> bool:
        self._cancelled = False

        model_info = get_model_by_id(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")

        models_dir = get_models_dir()
        os.makedirs(models_dir, exist_ok=True)

        url = model_info.url
        self._logger.info(f"Downloading model from {url}")

        if on_status:
            on_status("Downloading...")

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".tar.bz2", delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name

                # Download with progress
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 8192

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if self._cancelled:
                        self._logger.info("Download cancelled")
                        return False

                    tmp_file.write(chunk)
                    downloaded += len(chunk)

                    if on_progress and total_size > 0:
                        on_progress(downloaded, total_size)

            # Extract archive
            self._logger.info(f"Extracting to {models_dir}")
            if on_status:
                on_status("Extracting files...")
            with tarfile.open(tmp_path, "r:bz2") as tar:
                tar.extractall(path=models_dir)

            self._logger.info(f"Model {model_id} downloaded successfully")
            return True

        except requests.RequestException as e:
            self._logger.error(f"Download failed: {e}")
            raise
        finally:
            # Cleanup temp file
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def cancel(self) -> None:
        self._cancelled = True
