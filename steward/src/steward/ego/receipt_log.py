"""Opt-in, best-effort final public job receipts (one process per JSONL file)."""
import json
import os
from pathlib import Path
import threading
import time

_lock = threading.Lock()


def append(job, admission, wall_started):
    directory = os.environ.get('STEWARD_RECEIPT_LOG_DIR')
    if not directory:
        return
    try:
        # Use the exact public serializer used by skill_status, after finish hooks.
        from steward.ego.geometry_relocalization import event_view
        receipt = event_view(job.to_dict())
        row = dict(jobId=job.id, skill=job.skill, params=admission.get('params', {}),
                   simTime=dict(started=job.started, finished=job.finished),
                   wallTime=dict(started=wall_started, finished=time.time()), receipt=receipt)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
        with _lock:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'receipts-{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(line)
    except Exception:
        # Telemetry must not alter success, failure, cancellation or cleanup.
        pass


def configuration(config):
    """First event for a runtime, before any motion or receipt is published."""
    directory = os.environ.get('STEWARD_RECEIPT_LOG_DIR')
    if not directory:
        return
    try:
        with _lock:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'receipts-{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event='capability_configuration',
                    wallTime=time.time(), capabilityConfiguration=config), ensure_ascii=False)+'\n')
    except (OSError, ValueError, TypeError):
        pass
