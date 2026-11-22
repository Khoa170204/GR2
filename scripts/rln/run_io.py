"""File identities and serialized updates shared across Kaggle runs."""
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame):
    records = frame[["sentence", "label"]].to_dict("records")
    content = json.dumps(records, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield
