"""
KnowledgeSynthesizer: trascrizione Whisper e sintesi note Markdown.

Usa l'API Whisper di OpenAI per l'audio e Chat Completions con vision per le note.
Carica le API key da .env; gestisce errori di connessione e quota.
"""

import base64
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Agisci come un esperto di apprendimento rapido. Usa la trascrizione e i frame forniti per creare note in formato Markdown. Struttura: Titolo, Concetti Chiave, Spiegazione Dettagliata (riferendo i testi visti nei frame), e Takeaway pratici."""


class KnowledgeSynthesizerError(Exception):
    """Eccezione base per errori del KnowledgeSynthesizer."""

    pass


class TranscriptionError(KnowledgeSynthesizerError):
    """Errore durante la trascrizione audio."""

    pass


class NotesGenerationError(KnowledgeSynthesizerError):
    """Errore durante la generazione delle note."""

    pass


class APIKeyError(KnowledgeSynthesizerError):
    """API key mancante o non caricata."""

    pass


class KnowledgeSynthesizer:
    """
    Trascrive audio con Whisper (OpenAI) e genera note Markdown da trascrizione + frame.
    """

    def __init__(self, env_path: Union[str, Path] = ".env"):
        load_dotenv(env_path)
        self._api_key = os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise APIKeyError(
                "OPENAI_API_KEY non trovata. Impostala in .env o nelle variabili d'ambiente."
            )
        self._client = OpenAI(api_key=self._api_key)

    def transcribe_audio(self, audio_path: Union[str, Path]) -> str:
        """
        Invia l'MP3 all'API Whisper di OpenAI e restituisce il testo.

        Args:
            audio_path: Percorso del file MP3.

        Returns:
            Testo della trascrizione.

        Raises:
            TranscriptionError: File non trovato, connessione o quota API.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"File audio non trovato: {audio_path}")

        logger.info("Trascrizione audio: %s", audio_path.name)

        try:
            with open(audio_path, "rb") as f:
                response = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            text = response.text if hasattr(response, "text") else str(response)
            logger.info("Trascrizione completata (%d caratteri)", len(text))
            return text
        except APIConnectionError as e:
            raise TranscriptionError(
                f"Errore di connessione all'API OpenAI: {e}"
            ) from e
        except RateLimitError as e:
            raise TranscriptionError(
                f"Quota API superata o rate limit. Riprova più tardi: {e}"
            ) from e
        except APIStatusError as e:
            raise TranscriptionError(f"Errore API OpenAI: {e}") from e

    def _load_image_base64(self, image_path: Union[str, Path]) -> str:
        """Carica un'immagine e la restituisce in Base64 (data URL)."""
        path = Path(image_path)
        if not path.exists():
            raise NotesGenerationError(f"Immagine non trovata: {path}")
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        # OpenAI accetta data:image/jpeg;base64,...
        suffix = path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        return f"data:{mime};base64,{data}"

    def generate_notes(
        self,
        transcription: str,
        image_paths: List[Union[str, Path]],
        model: str = "gpt-4o",
    ) -> Tuple[str, dict]:
        """
        Genera note Markdown da trascrizione e frame (vision).

        Carica le immagini in Base64 e le invia con la trascrizione a OpenAI.

        Args:
            transcription: Testo della trascrizione audio.
            image_paths: Lista di percorsi delle immagini (frame).
            model: Modello OpenAI con vision (default gpt-4o).

        Returns:
            Tupla (note_markdown, usage) con usage contenente prompt_tokens,
            completion_tokens, total_tokens per il logging dei costi.

        Raises:
            NotesGenerationError: Immagini mancanti, connessione o quota API.
        """
        content = [
            {
                "type": "text",
                "text": (
                    "Trascrizione dell'audio del video:\n\n"
                    f"{transcription}\n\n"
                    "Analizza anche i frame allegati (in ordine) e crea le note "
                    "riferendo testi e contenuti visibili nei frame."
                ),
            },
        ]

        for i, p in enumerate(image_paths):
            try:
                data_url = self._load_image_base64(p)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )
            except NotesGenerationError:
                logger.warning("Salto immagine non trovata: %s", p)
                continue

        if len(content) == 1:
            logger.warning("Nessuna immagine caricata, generazione solo da trascrizione")

        logger.info("Invio a OpenAI (%s): trascrizione + %d immagini", model, len(content) - 1)

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            notes = response.choices[0].message.content
            usage = {}
            if getattr(response, "usage", None):
                u = response.usage
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                }
            logger.info("Note generate (%d caratteri)", len(notes))
            return notes, usage
        except APIConnectionError as e:
            raise NotesGenerationError(
                f"Errore di connessione all'API OpenAI: {e}"
            ) from e
        except RateLimitError as e:
            raise NotesGenerationError(
                f"Quota API superata o rate limit. Riprova più tardi: {e}"
            ) from e
        except APIStatusError as e:
            raise NotesGenerationError(f"Errore API OpenAI: {e}") from e
