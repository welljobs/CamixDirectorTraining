"""Small cross-platform primitives. No optional GUI/native package is needed."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid


def atomic_replace(source, destination):
    # A Windows download/antivirus reader may briefly hold the target open.
    for attempt in range(100):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path, timeout=60, poll=.1):
    """One-byte lock on Windows, flock on POSIX; release on exception/process exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    handle.seek(0, os.SEEK_END)
    if not handle.tell():
        handle.write(b'0')
        handle.flush()
    deadline = time.monotonic() + timeout
    locked = False
    windows = os.name == 'nt'
    try:
        while True:
            handle.seek(0)
            try:
                if windows:
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another process is using this workspace/download; wait for it to finish')
                time.sleep(poll)
        yield
    finally:
        if locked:
            handle.seek(0)
            if windows:
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
