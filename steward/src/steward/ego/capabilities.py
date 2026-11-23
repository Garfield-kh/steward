"""Versioned external-test baseline. Environment overrides apply to retained capabilities.

Retired switches are tombstones: attempts to enable removed code are recorded,
never silently reactivated. Child ASSEMBLY_SUPPLY_RELIABILITY flags inherit the retained umbrella.
"""
from steward.assembly_profile import enabled as assembly_enabled

import json
import os
from pathlib import Path
import subprocess

PROFILE = "assembly_capabilities-external-v1"
DEFAULTS = json.loads(Path(__file__).with_name("assembly_capabilities.json").read_text())
RETIRED = frozenset(['STEWARD_ASSEMBLY_DESCENT_TIMEOUT', 'STEWARD_ASSEMBLY_WRIST_REPLAN', 'STEWARD_ASSEMBLY_COMPLETION_GATE', 'STEWARD_ASSEMBLY_RELEASE_GEOMETRY', 'STEWARD_ASSEMBLY_BEARING_SURFACE', 'STEWARD_ASSEMBLY_INTERFACE_DIRECTION', 'STEWARD_ASSEMBLY_RELEASE_GATE', 'STEWARD_ASSEMBLY_ENDPOINT_CLEARANCE', 'STEWARD_ASSEMBLY_MOUTH_CAPTURE', 'STEWARD_ASSEMBLY_EMPTY_TRANSIT', 'STEWARD_ASSEMBLY_WORKSITE_ADVICE', 'STEWARD_ASSEMBLY_LOOK_COMPLETION', 'STEWARD_ASSEMBLY_CONSECUTIVE_SEATING', 'STEWARD_ASSEMBLY_SUPPLY_RELIABILITY_CROSS_SUPPORT'])


def get(name):
    if name not in DEFAULTS:
        raise KeyError("unregistered capability: " + name)
    if name in RETIRED:
        return '0'
    if name in os.environ:
        return os.environ[name]
    if name in ('STEWARD_ASSEMBLY_SUPPLY_RELIABILITY_HOLDING', 'STEWARD_ASSEMBLY_SUPPLY_RELIABILITY_SIZE', 'STEWARD_ASSEMBLY_SUPPLY_RELIABILITY_ROLE'):
        return get('STEWARD_ASSEMBLY_SUPPLY_RELIABILITY')
    if not assembly_enabled():
        if name == "STEWARD_PERCEPTION_HZ":
            return "10"
        if DEFAULTS[name] == "1":
            return "0"
    return DEFAULTS[name]


def snapshot():
    root = Path(__file__).resolve().parents[4]
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'],
            text=True, stderr=subprocess.DEVNULL, timeout=2).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = 'unknown', None
    return dict(profile=PROFILE, profileRevision='entry_layout', schema='capabilities.v1', codeVersion=revision, dirty=dirty,
        values={k:get(k) for k in sorted(DEFAULTS)}, retired=sorted(RETIRED),
        overrides={k:v for k,v in os.environ.items() if k in DEFAULTS and k not in RETIRED},
        ignoredRetiredOverrides={k:os.environ[k] for k in sorted(RETIRED) if k in os.environ})
