"""
VisionProcessor: estrae frame da video con scene detection.

Analizza video scaricati da media_fetcher.
Estrae frame sui cambi di scena (istogrammi/absdiff) o ogni 10s come fallback.
Ottimizza per vision: frame ridimensionati a max 1080px (lato lungo).
"""

import logging
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_SIDE_PX = 1080
FALLBACK_INTERVAL_SEC = 10
SCENE_CHANGE_THRESHOLD = 0.25  # Soglia differenza istogramma (0-1)
MIN_SCENE_FRAMES = 15  # Minimo frame tra due scene change (~0.5s a 30fps)


class VisionProcessorError(Exception):
    """Eccezione base per errori del VisionProcessor."""

    pass


class VisionProcessor:
    """
    Estrae frame rappresentativi da un video con scene detection.

    Usa differenza tra istogrammi per rilevare cambi di scena.
    Fallback: un frame ogni 10 secondi se nessun cambio significativo.
    """

    def __init__(
        self,
        frames_dir: Union[str, Path] = "temp/frames",
        max_side_px: int = MAX_SIDE_PX,
        fallback_interval_sec: float = FALLBACK_INTERVAL_SEC,
        scene_threshold: float = SCENE_CHANGE_THRESHOLD,
    ):
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.max_side_px = max_side_px
        self.fallback_interval_sec = fallback_interval_sec
        self.scene_threshold = scene_threshold

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Ridimensiona il frame mantenendo aspect ratio, max lato lungo = max_side_px."""
        h, w = frame.shape[:2]
        if max(h, w) <= self.max_side_px:
            return frame
        scale = self.max_side_px / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _compute_histogram_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calcola la differenza tra istogrammi (grayscale) dei due frame. Ritorna 0-1."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def _compute_absdiff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calcola la differenza assoluta media tra frame (normalizzata 0-1)."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff) / 255.0

    def _get_scene_change_indices(
        self, cap: cv2.VideoCapture, fps: float
    ) -> List[int]:
        """Rileva indici di frame dove avviene un cambio di scena."""
        frame_indices: List[int] = []
        prev_frame = None
        frame_idx = 0
        last_scene_idx = -MIN_SCENE_FRAMES * 10  # Forza primo frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                hist_diff = self._compute_histogram_diff(prev_frame, frame)
                abs_diff = self._compute_absdiff(prev_frame, frame)
                combined_diff = (hist_diff + abs_diff) / 2

                if (
                    combined_diff >= self.scene_threshold
                    and frame_idx - last_scene_idx >= MIN_SCENE_FRAMES
                ):
                    frame_indices.append(frame_idx)
                    last_scene_idx = frame_idx
                    logger.debug("Scene change a frame %d (diff=%.3f)", frame_idx, combined_diff)

            prev_frame = frame.copy()
            frame_idx += 1

        return frame_indices

    def _get_fallback_indices(self, total_frames: int, fps: float) -> List[int]:
        """Indici di frame ogni fallback_interval_sec secondi."""
        if fps <= 0 or total_frames <= 0:
            return []
        interval_frames = int(fps * self.fallback_interval_sec)
        if interval_frames <= 0:
            interval_frames = 1
        return list(range(0, total_frames, interval_frames))

    def extract_frames(self, video_path: Union[str, Path]) -> List[str]:
        """
        Estrae frame dal video con scene detection; fallback ogni 10s.

        Args:
            video_path: Percorso del video (es. da media_fetcher).

        Returns:
            Lista di percorsi assoluti dei file JPEG generati.

        Raises:
            VisionProcessorError: Video non valido o non apribile.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VisionProcessorError(f"Video non trovato: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VisionProcessorError(f"Impossibile aprire il video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stem = video_path.stem

        logger.info("Analisi scene: %s (%.1f fps, %d frame)", video_path.name, fps, total_frames)

        scene_indices = self._get_scene_change_indices(cap, fps)
        cap.release()

        # Includi sempre il primo frame
        if 0 not in scene_indices:
            scene_indices.insert(0, 0)

        if len(scene_indices) < 2:
            # Fallback: frame ogni 10 secondi
            frame_indices = self._get_fallback_indices(total_frames, fps)
            logger.info(
                "Pochi cambi di scena (%d), uso fallback ogni %.0fs: %d frame",
                len(scene_indices),
                self.fallback_interval_sec,
                len(frame_indices),
            )
        else:
            frame_indices = scene_indices
            logger.info("Rilevati %d cambi di scena", len(frame_indices))

        # Estrai e salva i frame
        out_paths: List[str] = []
        cap = cv2.VideoCapture(str(video_path))

        for i, idx in enumerate(sorted(set(frame_indices))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = self._resize_frame(frame)
            out_name = f"{stem}_frame_{i:04d}.jpg"
            out_path = self.frames_dir / out_name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            out_paths.append(str(out_path.resolve()))

        cap.release()
        logger.info("Salvati %d frame in %s", len(out_paths), self.frames_dir)

        return out_paths
