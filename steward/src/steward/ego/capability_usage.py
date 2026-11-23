"""Passive, lossless counters for evidence that otherwise keeps only its last value."""
import json
import os
from pathlib import Path
import threading
import time
import uuid

_lock = threading.Lock()


def record(capability, *, entity=None, stamp=None, measured=False, action=False, reason=None, stage=None, event_id=None,
           evidence=None):
    directory = os.environ.get('STEWARD_RECEIPT_LOG_DIR')
    if not directory:
        return
    try:
        row = dict(schema='capability_usage.v1', event='capability_usage', id=event_id or uuid.uuid4().hex,
            capability=capability, entity=entity, observedAt=stamp, stage=stage,
            triggered=True, measured=bool(measured), action=bool(action), reason=reason, wallTime=time.time())
        if evidence is not None:
            row['endpointEvidence'] = evidence
        with _lock:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'capability-usage-{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        return row['id']
    except (OSError, ValueError, TypeError):
        pass
