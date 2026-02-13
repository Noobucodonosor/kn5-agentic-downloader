#!/usr/bin/env python3
"""
Orchestratore Insta-Knowledge-Extractor.

Workflow: scarica Reel → estrae frame → trascrive audio → genera note Markdown → salva in output/.
URL da riga di comando o variabile d'ambiente REEL_URL.
"""

import argparse
import hashlib
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from src.skills.media_fetcher import (
    MediaFetcher,
    InvalidURLError,
    VideoRemovedError,
    DownloadError,
)
from src.skills.vision_processor import VisionProcessor, VisionProcessorError
from src.skills.knowledge_synthesizer import (
    KnowledgeSynthesizer,
    TranscriptionError,
    NotesGenerationError,
    APIKeyError,
)

TEMP_VIDEO = "temp/video"
TEMP_FRAMES = "temp/frames"
OUTPUT_DIR = "output"

# Prezzi API OpenAI (stima per logging costi; aggiornare se cambiano)
WHISPER_USD_PER_MINUTE = 0.006
GPT4O_USD_PER_1M_INPUT = 2.50
GPT4O_USD_PER_1M_OUTPUT = 10.00


def ensure_dirs():
    """Crea temp/video, temp/frames e output se non esistono."""
    Path(TEMP_VIDEO).mkdir(parents=True, exist_ok=True)
    Path(TEMP_FRAMES).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_reel_url(args_url: Optional[str], env_key: str = "REEL_URL") -> Optional[str]:
    """Restituisce l'URL da argomento o da variabile d'ambiente."""
    if args_url:
        return args_url.strip()
    return os.environ.get(env_key, "").strip() or None


def reel_slug_from_url(url: str) -> Optional[str]:
    """Estrae uno slug dal URL (es. /reel/ABC123/ -> ABC123)."""
    if not url:
        return None
    m = re.search(r"/reel/([A-Za-z0-9_-]+)", url) or re.search(
        r"/p/([A-Za-z0-9_-]+)", url
    )
    if m:
        return re.sub(r"[^\w\-]", "", m.group(1))[:64]
    return None


def output_filename(url: str) -> str:
    """Nome file per le note: slug del reel o timestamp."""
    slug = reel_slug_from_url(url)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if slug:
        return f"notes_{slug}_{ts}.md"
    return f"notes_{ts}.md"


def estimate_audio_minutes(audio_path: str) -> float:
    """Stima durata audio in minuti da dimensione file (MP3 ~128 kbps)."""
    p = Path(audio_path)
    if not p.exists():
        return 0.0
    size_mb = p.stat().st_size / (1024 * 1024)
    # 128 kbps ≈ 0.96 MB/min
    return size_mb / 0.96 if size_mb > 0 else 0.0


def _session_key(url: str) -> str:
    """Stesso identificatore usato da MediaFetcher per video/frame (hash URL)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def request_cleanup(
    url: str,
    all_frames: List[str],
    output_generated: bool,
    temp_video: str = TEMP_VIDEO,
    temp_frames: str = TEMP_FRAMES,
) -> None:
    """
    Chiede all'utente se eliminare i file temporanei del processo corrente.
    Eseguita solo se output_generated è True. Elimina solo video/frame di questo run.
    """
    if not output_generated:
        return
    key = _session_key(url)
    video_dir = Path(temp_video)
    frames_dir = Path(temp_frames)

    try:
        answer = input(
            "Vuoi eliminare i file temporanei (video e frame) scaricati? (s/n): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        logger.info("Input annullato, file temporanei conservati")
        return

    if answer == "s":
        removed = 0
        # Cartella temp/video/{key}/ (carousel) o file temp/video/{key}.*
        key_dir = video_dir / key
        if key_dir.is_dir():
            for f in key_dir.iterdir():
                try:
                    f.unlink()
                    removed += 1
                except OSError as e:
                    logger.warning("Impossibile eliminare %s: %s", f, e)
            try:
                key_dir.rmdir()
                removed += 1
            except OSError as e:
                logger.warning("Impossibile rimuovere directory %s: %s", key_dir, e)
        for f in video_dir.glob(f"{key}.*"):
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                logger.warning("Impossibile eliminare %s: %s", f, e)
        # Frame: solo quelli generati in questo run (all_frames)
        for p in all_frames:
            path = Path(p)
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                except OSError as e:
                    logger.warning("Impossibile eliminare %s: %s", path, e)
        logger.info("Pulizia completata: eliminati %d file/directory", removed)
        print("File temporanei eliminati.")
    else:
        path_video = video_dir.resolve()
        path_frames = frames_dir.resolve()
        logger.info("File conservati in %s e %s", path_video, path_frames)
        print(f"File conservati in {path_video} e {path_frames}")


def print_cost_log(
    whisper_minutes: float,
    usage: dict,
    model: str = "gpt-4o",
):
    """Stampa stima token e costi API. Whisper solo se audio processato (whisper_minutes > 0)."""
    cost_whisper = whisper_minutes * WHISPER_USD_PER_MINUTE
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", pt + ct)

    print("\n--- Logging di costo (stima) ---")
    if whisper_minutes > 0:
        print(f"Whisper: ~{whisper_minutes:.2f} min → ~${cost_whisper:.4f}")
    else:
        print("Whisper: non usato (nessun audio)")
    print(f"Chat ({model}): {pt} input, {ct} output, {total} total token")
    cost_input = (pt / 1_000_000) * GPT4O_USD_PER_1M_INPUT
    cost_output = (ct / 1_000_000) * GPT4O_USD_PER_1M_OUTPUT
    print(f"Chat (stima): input ~${cost_input:.4f}, output ~${cost_output:.4f}")
    print(f"Costo totale stimato: ~${cost_whisper + cost_input + cost_output:.4f}")
    print("--------------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scarica un Reel Instagram, estrae frame, trascrive e genera note Markdown."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="URL del Reel Instagram (oppure imposta REEL_URL)",
    )
    args = parser.parse_args()

    url = get_reel_url(args.url)
    if not url:
        print("Errore: fornisci un URL (argomento o variabile REEL_URL).", file=sys.stderr)
        sys.exit(1)

    ensure_dirs()

    fetcher = MediaFetcher(temp_dir=TEMP_VIDEO, cache_file="temp/download_cache.json")
    processor = VisionProcessor(frames_dir=TEMP_FRAMES)

    try:
        result = fetcher.fetch(url)
    except InvalidURLError as e:
        print(f"URL non valido: {e}", file=sys.stderr)
        sys.exit(1)
    except VideoRemovedError as e:
        print(f"Video non disponibile: {e}", file=sys.stderr)
        sys.exit(1)
    except DownloadError as e:
        print(f"Errore download: {e}", file=sys.stderr)
        sys.exit(1)

    media_type = result.get("media_type", "video")
    assets = result["assets"]
    audio_path = result.get("audio_path")

    print(f"Contenuto scaricato con successo (tipo: {media_type}, asset: {len(assets)})")

    # Gestione frame: video → extract_frames; immagini → aggiungi a all_frames
    all_frames = []
    video_exts = {".mp4", ".mov"}
    image_exts = {".jpg", ".jpeg", ".png"}
    for asset in assets:
        suf = Path(asset).suffix.lower()
        if suf in video_exts:
            try:
                frames = processor.extract_frames(asset)
                all_frames.extend(frames)
            except VisionProcessorError as e:
                print(f"Errore estrazione frame da {asset}: {e}", file=sys.stderr)
        elif suf in image_exts:
            all_frames.append(asset)

    print(f"Numero di frame/immagini: {len(all_frames)}")
    print("Percorso dei file:")
    for p in all_frames:
        print(f"  {p}")

    try:
        synthesizer = KnowledgeSynthesizer()
    except APIKeyError as e:
        print(f"Errore configurazione: {e}", file=sys.stderr)
        sys.exit(1)

    # Trascrizione condizionale
    if audio_path:
        try:
            transcription = synthesizer.transcribe_audio(audio_path)
        except TranscriptionError as e:
            print(f"Errore trascrizione: {e}", file=sys.stderr)
            sys.exit(1)
        print("Trascrizione completata")
    else:
        transcription = "Nessun audio disponibile per questo post."

    try:
        notes, usage = synthesizer.generate_notes(transcription, all_frames)
    except NotesGenerationError as e:
        print(f"Errore generazione note: {e}", file=sys.stderr)
        sys.exit(1)

    out_name = output_filename(url)
    out_path = Path(OUTPUT_DIR) / out_name
    out_path.write_text(notes, encoding="utf-8")
    print(f"Note salvate in: {out_path.resolve()}")

    # Costi: Whisper solo se l'audio è stato effettivamente processato
    whisper_min = estimate_audio_minutes(audio_path) if audio_path else 0.0
    print_cost_log(whisper_min, usage)

    # Pulizia interattiva solo se l'output è stato generato con successo
    request_cleanup(url, all_frames, output_generated=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
