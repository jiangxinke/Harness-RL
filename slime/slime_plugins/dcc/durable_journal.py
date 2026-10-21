"""Fresh ordered JSONL journals with durability outside policy deadlines.

Only close() certifies that every queued record has been written and synced.
A failed writer never retries, appends to an existing journal, or publishes a
completion marker. Callers serialize before enqueueing to freeze exact bytes.
"""
from __future__ import annotations

import os
from pathlib import Path
import queue
import threading


class OrderedDurableJSONLWriter:
    """One ordered writer thread; count includes only successful per-line fsyncs.

    The in-memory queue is deliberately unbounded: policy calls must not wait
    for storage. A process crash may lose queued records, and therefore cannot
    produce a successful close or a valid generation completion marker.
    """

    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")
        self._queue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        self._error = None
        self._count = 0
        self._submitted = 0
        self._stop = object()
        self._worker = threading.Thread(target=self._run, name="dcc-durable-journal", daemon=True)
        try:
            self._worker.start()
        except BaseException:
            self.stream.close()
            raise

    @property
    def count(self):
        with self._lock:
            return self._count

    def write_line(self, line):
        """Enqueue already serialized text, returning its ordered sequence number."""
        if not isinstance(line, str) or not line.endswith("\n"):
            raise ValueError("A serialized journal line must be a newline-terminated string")
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise ValueError("Journal is closed")
            sequence = self._submitted
            self._submitted += 1
            self._queue.put(line)
            return sequence

    def _record_error(self, error):
        with self._lock:
            if self._error is None:
                self._error = error
            else:
                self._error.add_note(f"Journal cleanup also failed: {type(error).__name__}: {error}")

    def _run(self):
        try:
            while True:
                line = self._queue.get()
                if line is self._stop:
                    break
                if self.stream.write(line) != len(line):
                    raise OSError("Incomplete journal line write")
                self.stream.flush()
                os.fsync(self.stream.fileno())
                with self._lock:
                    self._count += 1
            # Retain the original close-time sync, including for empty journals.
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException as error:
            self._record_error(error)
        finally:
            try:
                self.stream.close()
            except BaseException as error:
                self._record_error(error)

    def close(self):
        """Wait for the writer and re-raise any write/flush/fsync/close failure."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(self._stop)
        self._worker.join()
        with self._lock:
            error = self._error
        if error is not None:
            raise error


def close_durable_journals(journals, *, failure=None):
    """Drain every sink, preserving any active primary exception and all errors."""
    errors = []
    for journal in journals:
        close = getattr(journal, "close", None)
        if close is None:
            continue
        try:
            close()
        except BaseException as error:
            errors.append(error)
    if errors:
        primary = failure if failure is not None else errors.pop(0)
        for error in errors:
            if error is primary:
                continue
            primary.add_note(f"Journal cleanup also failed: {type(error).__name__}: {error}")
            for note in tuple(getattr(error, "__notes__", ())):
                primary.add_note(note)
        if failure is None:
            raise primary
