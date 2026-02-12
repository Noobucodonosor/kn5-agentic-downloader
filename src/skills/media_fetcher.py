"""
MediaFetcher: scarica video da Instagram con yt-dlp.

Rispetta il vincolo architetturale: verifica l'esistenza prima di scaricare.
Estrae audio in MP3 per Whisper e salva il video per l'estrazione frame.
"""

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

import yt_dlp

logger = logging.getLogger(__name__)


class MediaFetcherError(Exception):
    """Eccezione base per errori del MediaFetcher."""

    pass


class InvalidURLError(MediaFetcherError):
    """URL non valido o non supportato."""

    pass


class VideoRemovedError(MediaFetcherError):
    """Video rimosso o non disponibile."""

    pass


class DownloadError(MediaFetcherError):
    """Errore durante il download."""

    pass


class MediaFetcher:
    """
    Scarica video da Instagram, estrae audio in MP3 e gestisce il cache.

    Richiede ffmpeg installato per l'estrazione audio.
    """

    def __init__(
        self,
        temp_dir: Union[str, Path] = "temp",
        cache_file: Union[str, Path] = "temp/download_cache.json",
    ):
        self.temp_dir = Path(temp_dir)
        self.cache_file = Path(cache_file)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Carica l'indice dei download dalla cache."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug("Cache caricata: %d voci", len(self._cache))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Impossibile caricare cache: %s", e)
                self._cache = {}

    def _save_cache(self) -> None:
        """Salva l'indice dei download nella cache."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as e:
            logger.warning("Impossibile salvare cache: %s", e)

    def _url_hash(self, url: str) -> str:
        """Genera un hash stabile per l'URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _find_downloaded_video(self, key: str) -> Optional[Path]:
        """Trova il file video scaricato (può avere estensioni diverse)."""
        video_extensions = {".mp4", ".webm", ".mkv", ".m4a", ".mov"}
        for p in self.temp_dir.glob(f"{key}.*"):
            if p.suffix.lower() in video_extensions:
                return p
        return None

    def _validate_instagram_url(self, url: str) -> None:
        """Valida che l'URL sia un URL Instagram supportato."""
        if not url or not isinstance(url, str):
            raise InvalidURLError("URL non valido o vuoto")
        url_lower = url.strip().lower()
        if "instagram.com" not in url_lower and "instagr.am" not in url_lower:
            raise InvalidURLError(
                f"URL non supportato (atteso Instagram): {url[:80]}..."
            )

    def check_existing(self, url: str) -> Optional[dict]:
        """
        Verifica se il contenuto per questo URL è già stato scaricato.

        Returns:
            dict con `video_path` e `audio_path` se esistono, altrimenti None.
        """
        self._validate_instagram_url(url)
        key = self._url_hash(url)

        if key not in self._cache:
            logger.debug("URL non in cache: %s", url[:50])
            return None

        entry = self._cache[key]
        video_path = Path(entry.get("video_path", ""))
        audio_path = Path(entry.get("audio_path", ""))

        if not video_path.exists() or not audio_path.exists():
            logger.info("File in cache non più presenti, rimuovo voce: %s", key)
            del self._cache[key]
            self._save_cache()
            return None

        logger.info("Contenuto già presente per URL: %s", url[:50])
        return {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
        }

    def _extract_audio_mp3(self, video_path: Path, audio_path: Path) -> None:
        """Estrae l'audio dal video in formato MP3 usando ffmpeg."""
        logger.info("Estrazione audio in MP3: %s -> %s", video_path, audio_path)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-acodec",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as err:
            raise MediaFetcherError(
                "ffmpeg non trovato. Installalo per estrarre l'audio: "
                "https://ffmpeg.org/download.html"
            ) from err
        except subprocess.CalledProcessError as err:
            raise DownloadError(
                f"Errore ffmpeg durante estrazione audio: {err.stderr}"
            ) from err

    def _handle_ytdlp_error(self, err: Exception, url: str) -> None:
        """Mappa errori yt-dlp in eccezioni chiare."""
        msg = str(err).lower()
        if "unable to extract" in msg or "video unavailable" in msg:
            raise VideoRemovedError(
                f"Video non disponibile o rimosso: {url[:80]}..."
            ) from err
        if "invalid" in msg or "unsupported" in msg or "no video" in msg:
            raise InvalidURLError(f"URL non valido o non supportato: {url[:80]}...") from err
        raise DownloadError(f"Errore download: {err}") from err

    def fetch(self, url: str) -> dict:
        """
        Scarica video da Instagram, estrae audio MP3 e salva in temp.

        Prima esegue check_existing; se il contenuto esiste, restituisce i path.

        Returns:
            dict con `video_path` e `audio_path` (path assoluti).

        Raises:
            InvalidURLError: URL non valido
            VideoRemovedError: video rimosso o non disponibile
            DownloadError: errore durante download/estrazione
        """
        self._validate_instagram_url(url)

        existing = self.check_existing(url)
        if existing:
            return existing

        key = self._url_hash(url)
        video_path = self.temp_dir / f"{key}.mp4"
        audio_path = self.temp_dir / f"{key}.mp3"

        logger.info("Download da Instagram: %s", url[:60])

        ydl_opts = {
            "outtmpl": str(self.temp_dir / f"{key}.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "format": "best[ext=mp4]/best",
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as err:
            self._handle_ytdlp_error(err, url)
        except Exception as err:
            raise DownloadError(f"Errore inatteso durante download: {err}") from err

        # yt-dlp può usare estensioni diverse (.mp4, .webm, .m4a)
        video_path = self._find_downloaded_video(key)
        if not video_path:
            raise DownloadError("Download completato ma file video non trovato")

        logger.info("Video scaricato: %s", video_path)

        self._extract_audio_mp3(video_path, audio_path)

        if not audio_path.exists():
            raise DownloadError("Estrazione audio completata ma file MP3 non trovato")

        self._cache[key] = {
            "url": url,
            "video_path": str(video_path.resolve()),
            "audio_path": str(audio_path.resolve()),
        }
        self._save_cache()

        return {
            "video_path": str(video_path.resolve()),
            "audio_path": str(audio_path.resolve()),
        }
