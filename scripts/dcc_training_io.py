"""Fresh-run journals and complete checkpoint publication, without model imports.

Sync is opt-in and only used at checkpoint or terminal boundaries. Receipts
confirm successful API calls and a closed-log readback, not storage guarantees
beyond the filesystem's own flush/fsync/rename semantics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

CHECKPOINT_COMPLETE = "checkpoint_complete.json"
TRAINING_COMPLETE = "training_complete.json"


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value, *, sync=False):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        stream.flush()
        if sync:
            os.fsync(stream.fileno())


def _gather(value, world_size, distributed):
    if world_size == 1:
        return [value]
    values = [None] * world_size
    distributed.all_gather_object(values, value)
    return values


def prepare_training_output(output, *, rank, world_size, distributed=None):
    """Rank zero exclusively claims an empty directory; no resume is implied."""
    status = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise FileExistsError("Training output directory must be empty; use a new directory")
            owner = output / ".dcc-training-owner.json"
            _write_json(owner, {"world_size": world_size, "fresh_run": True})
            # Also detect a different writer racing the initial emptiness check.
            if any(path != owner for path in output.iterdir()):
                raise FileExistsError("Training output directory changed while claiming it")
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


class TrainingJournal:
    """One exclusive line-buffered writer, closed before terminal validation."""

    def __init__(self, output, rank, *, sync=False):
        self.output, self.rank, self.sync = output, rank, sync
        self.path = output / f"train_rank{rank}.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.records = 0

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.stream.close()

    def record(self, **value):
        line = json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.stream.write(line + "\n")
        self.records += 1
        return line

    def checkpoint_boundary(self):
        if self.sync:
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def close_and_validate(self, expected_steps):
        try:
            self.checkpoint_boundary()
        finally:
            self.stream.close()
        count, starts, finished, step = 0, 0, 0, 0
        first, last = None, None
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise ValueError("Training journal has an incomplete final record")
                value = json.loads(line)
                count += 1
                event = value.get("event")
                if count == 1:
                    first = event
                last = value
                if event == "start":
                    starts += 1
                elif event == "step":
                    step += 1
                    if type(value.get("step")) is not int or value["step"] != step:
                        raise ValueError("Training journal optimizer steps are not continuous")
                elif event == "finished":
                    finished += 1
                elif event == "failed":
                    raise ValueError("Training journal contains a failed event")
        if (count != self.records or starts != 1 or finished != 1 or first != "start"
                or step != expected_steps or last.get("event") != "finished"
                or type(last.get("optimizer_steps")) is not int
                or last["optimizer_steps"] != expected_steps):
            raise ValueError("Training journal does not contain the complete successful run")
        return {"rank": self.rank, "file": self.path.name, "records": count,
                "optimizer_steps": step, "closed_log_validated": True}


def publish_checkpoint(output, name, step, writer, *, sync=False):
    """Keep the original checkpoint layout; hide partial writes in a sibling."""
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError("Checkpoint name must be a visible directory basename")
    destination = output / name
    if destination.exists():
        raise FileExistsError("Checkpoint destination already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.incomplete-", dir=output))
    published = False
    try:
        writer(staging)
        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_symlink():
                raise ValueError("Checkpoint staging must not contain symlinks")
            if path.is_file():
                files.append({"file": str(path.relative_to(staging)), "bytes": path.stat().st_size})
                if sync:
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
        if not files or (staging / CHECKPOINT_COMPLETE).exists():
            raise ValueError("Checkpoint writer produced no files or used the reserved completion marker")
        _write_json(staging / CHECKPOINT_COMPLETE,
                    {"optimizer_steps": step, "files": files, "complete": True}, sync=sync)
        if sync:
            directories = [path for path in staging.rglob("*") if path.is_dir()]
            for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
                _sync_directory(path)
            _sync_directory(staging)
        if destination.exists():
            raise FileExistsError("Checkpoint destination appeared during save")
        staging.rename(destination)
        published = True
        if sync:
            _sync_directory(output)
    except BaseException:
        # Preserve incomplete artifacts for diagnosis, without a ready marker.
        (destination if published else staging).joinpath(CHECKPOINT_COMPLETE).unlink(missing_ok=True)
        raise


def finalize_training(journal, expected_steps, *, world_size, distributed=None):
    """Only rank zero publishes success, after every rank closed and read back."""
    try:
        local = journal.close_and_validate(expected_steps)
    except Exception as error:
        local = {"rank": journal.rank, "error": f"{type(error).__name__}: {error}"}
    receipts = _gather(local, world_size, distributed)
    if (len(receipts) != world_size or any(not isinstance(item, dict) for item in receipts)
            or {item.get("rank") for item in receipts} != set(range(world_size))
            or any(item.get("closed_log_validated") is not True
                   or item.get("optimizer_steps") != expected_steps for item in receipts)):
        raise RuntimeError(f"Training logs failed terminal validation: {receipts}")
    if journal.rank == 0:
        output = journal.output
        final_checkpoint = output / "final" / CHECKPOINT_COMPLETE
        checkpoint = json.loads(final_checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("complete") is not True or checkpoint.get("optimizer_steps") != expected_steps:
            raise ValueError("Final checkpoint is not complete for the requested optimizer steps")
        destination = output / TRAINING_COMPLETE
        temporary = output / ".training_complete.pending.json"
        if destination.exists():
            raise FileExistsError("Training completion receipt already exists")
        published = False
        try:
            _write_json(temporary, {"complete": True, "optimizer_steps": expected_steps,
                        "world_size": world_size, "logs": receipts, "final_checkpoint": "final",
                        "sync_requested": journal.sync}, sync=journal.sync)
            temporary.rename(destination)
            published = True
            if journal.sync:
                _sync_directory(output)
        except BaseException:
            (destination if published else temporary).unlink(missing_ok=True)
            raise
