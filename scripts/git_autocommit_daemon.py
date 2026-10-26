#!/usr/bin/env python3
"""scripts/git_autocommit_daemon.py — harness-level auto commit via filesystem watch.

OpenCode V2's plugin API no longer exposes tool/edit hooks, so the "commit on every
modification" guarantee lives here as a watchdog daemon (independent of the agent).

Behavior:
  * watches /root/autodl-tmp/vla-omni-bench recursively (excl. data/ models/ runs/
    viz/ .git/ __pycache__/ node_modules/); excluded subtrees are never watched
  * debounces 3s; stages ONLY touched files; commits with a deterministic message
  * handles create, modify, delete and rename (both source and destination)
  * skips ignored files, merge/rebase, empty diffs; failures are logged, never fatal

Usage:
  nohup python scripts/git_autocommit_daemon.py >> runs/autocommit.log 2>&1 &
  kill "$(pgrep -f scripts/git_autocommit_daemon.py)"
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path("/root/autodl-tmp/vla-omni-bench").resolve()
# These only match at the repository root (e.g. models/ weights, runs/ logs).
EXCLUDE_ROOT_DIRS = {".git", "data", "models", "runs", "viz"}
# These match at any depth.
EXCLUDE_ANY_DIRS = {"__pycache__", "node_modules"}
EXCLUDE_SUFFIXES = {".npz", ".safetensors", ".part", ".incomplete", ".pyc"}


def _excluded(rel: Path) -> bool:
    parts = rel.parts
    if not parts:
        return False
    return parts[0] in EXCLUDE_ROOT_DIRS or any(p in EXCLUDE_ANY_DIRS for p in parts)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True)


def _rel(path: str) -> str | None:
    """Return the path relative to ROOT, or None if it is outside the repo."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return None


def _ignored(rel: str) -> bool:
    return git("check-ignore", "-q", "--", rel).returncode == 0


def _exists_or_tracked(rel: str) -> bool:
    if (ROOT / rel).exists():
        return True
    # Deleted files still need staging when they were tracked.
    return git("ls-files", "--error-unmatch", "--", rel).returncode == 0


def _merge_or_rebase() -> bool:
    return ((ROOT / ".git" / "MERGE_HEAD").exists()
            or (ROOT / ".git" / "rebase-merge").is_dir()
            or (ROOT / ".git" / "rebase-apply").is_dir())


def safe_commit(files: set[str]):
    rel = sorted({r for f in files
                  if (r := _rel(f)) and not _ignored(r) and _exists_or_tracked(r)})
    if not rel:
        return
    if not git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true":
        return
    if _merge_or_rebase():
        print("[autocommit] skip: merge/rebase in progress", flush=True)
        return

    added = git("add", "--", *rel)
    if added.returncode != 0:
        print(f"[autocommit] add failed: {added.stderr.strip()}", flush=True)
        return

    stat = git("diff", "--cached", "--stat")
    if stat.returncode != 0 or not stat.stdout.strip():
        return

    short = " | ".join(stat.stdout.strip().splitlines()[:6])
    committed = git("commit", "-q", "-m", f"auto: {', '.join(rel[:4])}", "-m", short)
    if committed.returncode != 0:
        print(f"[autocommit] commit failed: {committed.stderr.strip()}", flush=True)
        return
    print(f"[autocommit] committed: {', '.join(rel[:4])}", flush=True)


class Handler(FileSystemEventHandler):
    def __init__(self, observer: Observer):
        super().__init__()
        self.observer = observer
        self.watched: set[str] = set()
        self.pending: set[str] = set()
        self.timer: threading.Timer | None = None
        self.lock = threading.Lock()

    def watch(self, directory: str):
        """Watch one directory (non-recursive) unless it is excluded."""
        p = Path(directory).resolve()
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            return
        if _excluded(rel):
            return
        if not p.is_dir() or str(p) in self.watched:
            return
        self.observer.schedule(self, str(p), recursive=False)
        self.watched.add(str(p))

    def watch_tree(self, directory: str = str(ROOT)):
        """Watch every non-excluded directory under `directory`."""
        base = Path(directory).resolve()
        try:
            base.relative_to(ROOT)
        except ValueError:
            return
        for dirpath, dirnames, _ in os.walk(base):
            current = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if not _excluded((current / d).relative_to(ROOT))
            ]
            self.watch(dirpath)

    def _on_event(self, path: str):
        p = Path(path)
        try:
            rel = p.resolve().relative_to(ROOT)
        except (ValueError, OSError):
            return
        if p.suffix in EXCLUDE_SUFFIXES:
            return
        if _excluded(rel):
            return
        with self.lock:
            self.pending.add(str(p))
            if self.timer is not None:
                self.timer.cancel()
            self.timer = threading.Timer(3.0, self._flush)
            self.timer.start()

    def _flush(self):
        with self.lock:
            files = self.pending
            self.pending = set()
            self.timer = None
        safe_commit(files)

    def on_modified(self, event):
        if not event.is_directory:
            self._on_event(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            self.watch_tree(event.src_path)
        else:
            self._on_event(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._on_event(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            self.watch_tree(event.dest_path)
        else:
            self._on_event(event.src_path)
            self._on_event(event.dest_path)


def main():
    print(f"[autocommit] watching {ROOT}", flush=True)
    print(f"[autocommit] pid {os.getpid()}", flush=True)
    observer = Observer()
    handler = Handler(observer)
    handler.watch_tree()
    observer.start()

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    try:
        while not stop.wait(60):
            pass
    finally:
        observer.stop()
        observer.join()
        print("[autocommit] stopped", flush=True)


if __name__ == "__main__":
    main()
