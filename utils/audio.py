"""
audio.py — Non-blocking text-to-speech for coaching cues
==========================================================

Wraps ``pyttsx3`` in a dedicated daemon thread so the TTS engine
never blocks the main Streamlit / WebRTC thread.  If pyttsx3 ran
on the main thread the entire video feed would freeze while the AI
speaks.

Architecture:
    A persistent background thread owns the pyttsx3 engine and
    monitors a ``queue.Queue`` for text to speak.  The public
    ``speak()`` function simply enqueues the text and returns
    immediately.  ``stop()`` sends a sentinel to shut down the
    thread cleanly.

Graceful degradation:
    If pyttsx3 is not installed or the OS has no TTS backend, all
    calls silently no-op so the rest of the app continues running.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()


class TTSEngine:
    """Non-blocking text-to-speech wrapper around pyttsx3.

    Usage:
        tts = TTSEngine()          # starts the background thread
        tts.speak("Brace your core!")
        tts.speak("Great rep!")     # queued, plays after the first
        tts.stop()                  # shutdown (optional — daemon thread)
    """

    def __init__(self, rate: int = 175, volume: float = 0.9) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._available = False
        self._thread: Optional[threading.Thread] = None

        try:
            import pyttsx3  # noqa: F401
            self._available = True
        except (ImportError, RuntimeError) as exc:
            logger.warning("pyttsx3 unavailable (%s) — TTS disabled.", exc)
            return

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="neurofit-tts",
        )
        self._thread.start()

    @property
    def available(self) -> bool:
        return self._available

    def speak(self, text: str) -> None:
        """Queue *text* for speech.  Returns immediately (non-blocking)."""
        if not self._available or not text:
            return
        self._queue.put(text)

    def flush(self) -> None:
        """Drain all pending speech items so TTS stops after the current
        utterance finishes.  Does not kill the worker thread."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        """Signal the worker to shut down and wait for it."""
        if self._thread is None or not self._thread.is_alive():
            return
        self.flush()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        """Background loop: own the pyttsx3 engine, pull text from the
        queue, speak it, repeat.  Runs until ``_SENTINEL`` is received."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
        except Exception as exc:
            logger.error("pyttsx3 engine init failed: %s", exc)
            self._available = False
            return

        while True:
            try:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                engine.say(str(item))
                engine.runAndWait()
            except Exception as exc:
                logger.warning("TTS playback error: %s", exc)

        try:
            engine.stop()
        except Exception:
            pass
