"""
MediaFetcher: scarica contenuto Instagram (Reel con yt-dlp, carousel/immagini con instaloader).

Reel → yt-dlp (più veloce). Se yt-dlp fallisce o è post/carosello → instaloader.
Output: {media_type, assets, audio_path}; assets solo .jpg, .png, .mp4.
"""

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yt_dlp

try:
    import instaloader
except ImportError:
    instaloader = None

logger = logging.getLogger(__name__)

# Solo questi estensioni in assets (escludi .json, .txt, .xz di instaloader)
ASSETS_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4"}


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

    def _shortcode_from_url(self, url: str) -> Optional[str]:
        """Estrae lo shortcode da URL Instagram (es. /p/ABC123/ o /reel/ABC123/)."""
        if not url:
            return None
        m = re.search(r"/p/([A-Za-z0-9_-]+)", url) or re.search(r"/reel/([A-Za-z0-9_-]+)", url)
        return m.group(1) if m else None

    def _find_downloaded_video(self, key: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        """Trova il file video scaricato (può avere estensioni diverse)."""
        video_extensions = {".mp4", ".webm", ".mkv", ".m4a", ".mov"}
        search_dir = base_dir or self.temp_dir
        for p in search_dir.glob(f"{key}.*"):
            if p.suffix.lower() in video_extensions:
                return p
        return None

    def _is_image_file(self, path: Union[str, Path]) -> bool:
        """True se l'estensione è immagine (.jpg, .webp, .png, ecc.): non usare ffmpeg per audio."""
        return Path(path).suffix.lower() in {".jpg", ".jpeg", ".webp", ".png", ".gif"}

    def _extract_media_type_and_entries(self, url: str) -> Tuple[str, List[dict]]:
        """Estrae metadati con yt-dlp. Nessun format video obbligatorio (supporta immagini)."""
        ydl_opts = {
            "quiet": True,
            "extract_flat": "in_playlist",
            "ignoreerrors": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            raise DownloadError("Impossibile estrarre metadati dal URL")

        # Parsing caroselli: _type == 'playlist' → itera su entries
        if info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            if entries:
                logger.info("Carosello rilevato: %d elementi", len(entries))
                return "carousel", entries
            raise DownloadError("Playlist senza entries")

        # Singolo elemento (o entry unica)
        entries = info.get("entries")
        if entries is not None:
            entries = [e for e in entries if e]
        if not entries:
            entries = [info]
        if len(entries) > 1:
            logger.info("Carosello rilevato: %d elementi", len(entries))
            return "carousel", entries

        entry = entries[0]
        has_duration = entry.get("duration") is not None
        ext = (entry.get("ext") or "").lower()
        image_exts = ("jpg", "jpeg", "png", "webp", "gif")
        if ext in image_exts or (not has_duration and not entry.get("formats")):
            return "image", entries
        return "video", entries

    def _best_thumbnail_url(self, info: dict) -> Optional[str]:
        """Da info_dict restituisce l'URL dell'immagine a risoluzione più alta (thumbnails)."""
        thumbs = info.get("thumbnails") or []
        if not thumbs:
            return info.get("thumbnail") or info.get("preview")
        # Ordina per width o height (priorità width) e prendi la più grande
        def size(t: dict) -> int:
            w = t.get("width") or 0
            h = t.get("height") or 0
            return w * h or (w or h)
        sorted_thumbs = sorted(thumbs, key=size, reverse=True)
        for t in sorted_thumbs:
            u = t.get("url")
            if u:
                return u
        return info.get("thumbnail") or info.get("preview")

    def _try_extract_single_image_fallback(self, url: str, key: str) -> Optional[dict]:
        """
        Fallback quando l'estrazione fallisce o non trova video (es. "There is no video").
        Usa info_dict.url (post statici) o thumbnails a risoluzione più alta.
        Output: {media_type, assets, audio_path} con assets a un solo elemento.
        """
        ydl_opts = {
            "quiet": True,
            "extract_flat": "in_playlist",
            "ignoreerrors": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception:
            return None
        if not info:
            return None

        # 1) url diretto (spesso presente per post statici)
        direct_url = info.get("url")
        # 2) thumbnails: immagine a risoluzione più alta
        if not direct_url:
            direct_url = self._best_thumbnail_url(info)
        if not direct_url:
            direct_url = info.get("thumbnail") or info.get("preview")
        if not direct_url and info.get("entries"):
            first = info["entries"][0]
            if isinstance(first, dict):
                direct_url = first.get("url") or self._best_thumbnail_url(first) or first.get("thumbnail") or first.get("preview")
        if not direct_url:
            return None

        out_path = self.temp_dir / f"{key}.%(ext)s"
        ydl_opts = {
            "outtmpl": str(out_path),
            "quiet": False,
            "no_warnings": False,
            "ignoreerrors": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([direct_url])
        except Exception as e:
            logger.warning("Fallback download immagine fallito: %s", e)
            return None
        image_path = next(self.temp_dir.glob(f"{key}.*"), None)
        if not image_path:
            return None
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        if image_path.suffix.lower() not in image_exts:
            return None
        logger.info("Immagine singola estratta tramite fallback (url/thumbnails)")
        return {"media_type": "image", "assets": [str(image_path.resolve())], "audio_path": None}

    def _assets_only_media(self, paths: List[Path]) -> List[str]:
        """Filtra solo .jpg, .jpeg, .png, .mp4 (esclude .json, .txt, .xz di instaloader)."""
        return [
            str(p.resolve())
            for p in paths
            if p.suffix.lower() in ASSETS_EXTENSIONS
        ]

    def _fetch_with_instaloader(self, shortcode: str, key: str) -> Optional[dict]:
        """
        Scarica post/carosello/immagine con instaloader.
        Restituisce {media_type, assets, audio_path}; assets solo .jpg, .png, .mp4.
        """
        if instaloader is None:
            logger.warning("instaloader non installato, impossibile usare fallback")
            return None
        out_dir = self.temp_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)
        # dirname_pattern con base temp_dir → file in temp_dir/target/
        L = instaloader.Instaloader(
            dirname_pattern=str(self.temp_dir / "{target}"),
            download_pictures=True,
            download_videos=True,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            post_metadata_txt_pattern="",
        )
        try:
            post = instaloader.Post.from_shortcode(L.context, shortcode)
        except Exception as e:
            logger.warning("Instaloader from_shortcode fallito: %s", e)
            return None
        # GraphSidecar = carousel (tutti i nodi), GraphImage = immagine, GraphVideo = video
        if post.typename == "GraphSidecar":
            logger.info("Instaloader: carosello con %d nodi", post.mediacount)
        elif post.typename == "GraphImage":
            logger.info("Instaloader: post immagine singola")
        else:
            logger.info("Instaloader: post video")
        try:
            L.download_post(post, target=key)
        except Exception as e:
            logger.warning("Instaloader download_post fallito: %s", e)
            return None
        # Raccolta solo file media (no .json, .txt)
        media_files = sorted(
            (p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in ASSETS_EXTENSIONS),
            key=lambda p: p.name,
        )
        asset_paths = self._assets_only_media(media_files)
        if not asset_paths:
            logger.warning("Instaloader: nessun file .jpg/.png/.mp4 trovato in %s", out_dir)
            return None
        if post.typename == "GraphSidecar":
            media_type = "carousel"
        elif post.typename == "GraphImage":
            media_type = "image"
        else:
            media_type = "video"
        return {"media_type": media_type, "assets": asset_paths, "audio_path": None}

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
            dict con media_type, assets, audio_path se tutti gli asset esistono, altrimenti None.
        """
        self._validate_instagram_url(url)
        key = self._url_hash(url)

        if key not in self._cache:
            logger.debug("URL non in cache: %s", url[:50])
            return None

        entry = self._cache[key]
        # Compatibilità con cache vecchia (video_path/audio_path)
        assets_list = entry.get("assets")
        if not assets_list and entry.get("video_path"):
            assets_list = [entry["video_path"]]
        for p in (assets_list or []):
            if not Path(p).exists():
                logger.info("Asset in cache non più presente: %s, rimuovo voce: %s", p, key)
                del self._cache[key]
                self._save_cache()
                return None
        ap = entry.get("audio_path")
        if ap and not Path(ap).exists():
            logger.info("Audio in cache non più presente, rimuovo voce: %s", key)
            del self._cache[key]
            self._save_cache()
            return None

        audio_resolved = str(Path(ap).resolve()) if ap else None
        resolved_assets = [str(Path(p).resolve()) for p in (assets_list or [])]

        logger.info("Contenuto già presente per URL: %s", url[:50])
        return {
            "media_type": entry.get("media_type", "video"),
            "assets": resolved_assets,
            "audio_path": audio_resolved,
        }

    def _extract_audio_mp3(self, video_path: Path, audio_path: Path) -> bool:
        """Estrae l'audio dal video in MP3. Restituisce True se ok, False se fallisce (es. video muto)."""
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
            return True
        except FileNotFoundError as err:
            logger.warning("ffmpeg non trovato: %s", err)
            return False
        except subprocess.CalledProcessError as err:
            logger.warning("Estrazione audio fallita (video muto?): %s", getattr(err, "stderr", err))
            return False

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
        Scarica contenuto da Instagram (carousel, immagine o video), estrae audio se presente.

        Prima esegue check_existing; se il contenuto esiste, restituisce il dict in cache.

        Returns:
            dict con media_type ("carousel"|"image"|"video"), assets (lista path), audio_path (str o None).
        """
        self._validate_instagram_url(url)

        existing = self.check_existing(url)
        if existing:
            return existing

        key = self._url_hash(url)
        shortcode = self._shortcode_from_url(url)

        def try_instaloader_fallback() -> Optional[dict]:
            if not shortcode:
                return None
            result = self._fetch_with_instaloader(shortcode, key)
            if result:
                self._cache[key] = {"url": url, **result}
                self._save_cache()
                return result
            return None

        # Ibrido: Reel → yt-dlp per primo (più veloce); se fallisce o è post/carosello → instaloader
        try:
            media_type, entries = self._extract_media_type_and_entries(url)
        except yt_dlp.utils.DownloadError as err:
            msg = str(err).lower()
            fallback = try_instaloader_fallback()
            if fallback:
                logger.info("yt-dlp fallito, usato instaloader per shortcode %s", shortcode)
                return fallback
            if "no video" in msg or "there is no video" in msg:
                img_fallback = self._try_extract_single_image_fallback(url, key)
                if img_fallback:
                    self._cache[key] = {"url": url, **img_fallback}
                    self._save_cache()
                    return img_fallback
            self._handle_ytdlp_error(err, url)
        except InvalidURLError as err:
            fallback = try_instaloader_fallback()
            if fallback:
                logger.info("yt-dlp fallito, usato instaloader per shortcode %s", shortcode)
                return fallback
            msg = str(err).lower()
            if "no video" in msg or "there is no video" in msg:
                img_fallback = self._try_extract_single_image_fallback(url, key)
                if img_fallback:
                    self._cache[key] = {"url": url, **img_fallback}
                    self._save_cache()
                    return img_fallback
            raise
        except Exception as err:
            fallback = try_instaloader_fallback()
            if fallback:
                return fallback
            raise DownloadError(f"Errore estrazione metadati: {err}") from err

        logger.info("Tipo rilevato: %s, download da Instagram: %s", media_type, url[:60])

        if media_type == "carousel":
            out_dir = self.temp_dir / key
            out_dir.mkdir(parents=True, exist_ok=True)
            # Nessun format video obbligatorio; ignoreerrors per non fermarsi su elementi immagine
            ydl_opts = {
                "outtmpl": str(out_dir / "%(playlist_index)04d.%(ext)s"),
                "quiet": False,
                "no_warnings": False,
                "noplaylist": False,
                "ignoreerrors": True,
                "postprocessors": [],
            }
            # Scarica playlist; se abbiamo entries con URL, scarica ogni entry (più resiliente)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except yt_dlp.utils.DownloadError as err:
                fallback = try_instaloader_fallback()
                if fallback:
                    return fallback
                self._handle_ytdlp_error(err, url)
            asset_paths = self._assets_only_media(
                sorted((p for p in out_dir.iterdir() if p.is_file()), key=lambda p: p.name)
            )
            if not asset_paths:
                fallback = try_instaloader_fallback()
                if fallback:
                    return fallback
                raise DownloadError("Carosello: nessun asset scaricato")
            logger.info("Carosello scaricato: %d asset (solo .jpg/.png/.mp4)", len(asset_paths))
            self._cache[key] = {"url": url, "media_type": "carousel", "assets": asset_paths, "audio_path": None}
            self._save_cache()
            return {"media_type": "carousel", "assets": asset_paths, "audio_path": None}

        if media_type == "image":
            out_path = self.temp_dir / f"{key}.%(ext)s"
            ydl_opts = {
                "outtmpl": str(out_path),
                "quiet": False,
                "no_warnings": False,
                "ignoreerrors": True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except yt_dlp.utils.DownloadError as err:
                fallback = try_instaloader_fallback()
                if fallback:
                    return fallback
                img_fb = self._try_extract_single_image_fallback(url, key)
                if img_fb:
                    self._cache[key] = {"url": url, **img_fb}
                    self._save_cache()
                    return img_fb
                self._handle_ytdlp_error(err, url)
            candidates = [p for p in self.temp_dir.iterdir() if p.name.startswith(key) and p.suffix.lower() in ASSETS_EXTENSIONS]
            if not candidates:
                fallback = try_instaloader_fallback()
                if fallback:
                    return fallback
                img_fb = self._try_extract_single_image_fallback(url, key)
                if img_fb:
                    self._cache[key] = {"url": url, **img_fb}
                    self._save_cache()
                    return img_fb
                raise DownloadError("Download completato ma file immagine non trovato")
            assets = [str(Path(p).resolve()) for p in sorted(candidates, key=lambda x: x.name)]
            self._cache[key] = {"url": url, "media_type": "image", "assets": assets, "audio_path": None}
            self._save_cache()
            return {"media_type": "image", "assets": assets, "audio_path": None}

        # video / reel (nessun format obbligatorio tipo bestvideo per non fallire su edge case)
        ydl_opts = {
            "outtmpl": str(self.temp_dir / f"{key}.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "format": "best[ext=mp4]/best/best",
            "ignoreerrors": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as err:
            self._handle_ytdlp_error(err, url)
        except Exception as err:
            raise DownloadError(f"Errore inatteso durante download: {err}") from err

        video_path = self._find_downloaded_video(key)
        if not video_path:
            raise DownloadError("Download completato ma file video non trovato")
        logger.info("Video scaricato: %s", video_path)

        # Audio condizionale: solo per file video; .jpg/.webp/.png → audio_path = None, niente ffmpeg
        audio_resolved = None
        if not self._is_image_file(video_path):
            audio_path = self.temp_dir / f"{key}.mp3"
            if self._extract_audio_mp3(video_path, audio_path) and audio_path.exists():
                audio_resolved = str(audio_path.resolve())
            else:
                logger.info("Audio non disponibile o estrazione fallita, proseguo senza audio")

        assets = [str(video_path.resolve())]
        self._cache[key] = {
            "url": url,
            "media_type": "video",
            "assets": assets,
            "audio_path": audio_resolved,
        }
        self._save_cache()
        return {"media_type": "video", "assets": assets, "audio_path": audio_resolved}
