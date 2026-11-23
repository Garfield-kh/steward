"""Compare real dispatch and HTTP registration against independently collected references."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / 'tests/expected/runtime-profile-surface.json').read_text())


@pytest.mark.parametrize('profile', ['default', 'toy'])
def test_runtime_prediction_execution_and_api_match_reference(profile):
    env = {k: v for k, v in os.environ.items() if not k.startswith('STEWARD_') and k != 'SUBGOAL_SUCCESS'}
    env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if profile == 'toy':
        env.update(STEWARD_ASSEMBLY_PROFILE='toy', STEWARD_FASTEN_MODE='seat', STEWARD_NOSLIP='5')
    result = subprocess.run([sys.executable, str(ROOT / 'tests/profile_runtime_surface.py'), '--check-catalog'],
                            env=env, capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    expected = json.loads(json.dumps(EXPECTED[profile]))
    if profile == 'toy':
        # The only wire change is metadata on the existing read-only response.
        expected['routeParameters']['GET /ego/control'] = "lambda q, b: {**runtime.control(), 'assemblyProfile': 'toy'}"
    assert actual == expected
    markers = [line.removeprefix('STEWARD_EGO_INTERFACE ') for line in result.stderr.splitlines()
               if line.startswith('STEWARD_EGO_INTERFACE ')]
    if profile == 'toy':
        diagnostic, = map(json.loads, markers)
        assert diagnostic['profile'] == 'toy'
        assert diagnostic['routes'] == actual['routes']
        assert set(diagnostic['skills']) == {skill for skill, accepted in actual['prediction'].items()
                                            if accepted and not skill.startswith('_')}
        assert Path(diagnostic['module']) == ROOT / 'src/steward/ego/server.py'
        assert Path(diagnostic['runtimeModule']) == ROOT / 'src/steward/ego/runtime.py'
    else:
        assert not markers
