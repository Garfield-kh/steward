"""Toy process settings and model instructions survive the Harness launch boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def clean_env(profile):
    env = {k: v for k, v in os.environ.items() if not k.startswith('STEWARD_') and k != 'SUBGOAL_SUCCESS'}
    env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    if profile == 'toy':
        env.update(STEWARD_ASSEMBLY_PROFILE='toy', STEWARD_FASTEN_MODE='seat', STEWARD_NOSLIP='5')
    return env


@pytest.mark.parametrize('profile', ['default', 'toy'])
def test_chat_environment_preserves_toy_settings_and_default_precedence(profile):
    script = '''
import json
from steward.assembly_profile import harness_environment
values = harness_environment({'STEWARD_ASSEMBLY_PROFILE':'off', 'STEWARD_FASTEN_MODE':'off',
 'STEWARD_NOSLIP':'1', 'SUBGOAL_SUCCESS':'1', 'TEST_HARNESS_SETTING':'file'})
print(json.dumps({k:values[k] for k in ('STEWARD_ASSEMBLY_PROFILE','STEWARD_FASTEN_MODE',
 'STEWARD_NOSLIP','SUBGOAL_SUCCESS','TEST_HARNESS_SETTING')}))
'''
    actual = json.loads(subprocess.check_output([sys.executable, '-c', script], env=clean_env(profile), text=True))
    assert actual == {'STEWARD_ASSEMBLY_PROFILE': 'toy' if profile == 'toy' else 'off',
                      'STEWARD_FASTEN_MODE': 'seat' if profile == 'toy' else 'off',
                      'STEWARD_NOSLIP': '5' if profile == 'toy' else '1',
                      'SUBGOAL_SUCCESS': '0' if profile == 'toy' else '1',
                      'TEST_HARNESS_SETTING': 'file'}


@pytest.mark.parametrize('profile', ['default', 'toy'])
def test_run_ego_environment_matches_process_profile(tmp_path, profile):
    if shutil.which('bash') is None:
        pytest.skip('The shell launcher requires Bash')
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    launcher = scripts / 'run-ego.sh'
    shutil.copyfile(ROOT / 'scripts/run-ego.sh', launcher)
    binary = tmp_path / 'dsh/node_modules/.bin/dsh'
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/usr/bin/env bash\nprintf "%s %s %s\\n" "$STEWARD_ASSEMBLY_PROFILE" "$STEWARD_FASTEN_MODE" "$STEWARD_NOSLIP"\n')
    binary.chmod(0o755)
    (tmp_path / '.env').write_text('STEWARD_ASSEMBLY_PROFILE=off\nSTEWARD_FASTEN_MODE=off\nSTEWARD_NOSLIP=1\n')
    result = subprocess.check_output(['bash', str(launcher), 'test'], env=clean_env(profile), text=True)
    assert result.strip() == ('toy seat 5' if profile == 'toy' else 'off off 1')


@pytest.mark.parametrize('profile', ['default', 'toy'])
def test_robot_launch_instructions_match_reference(profile):
    script = '''
import json
from steward.world.chat import robot_brief
from steward.diagnosis import _failed_target_findings
print(json.dumps({'brief':robot_brief('steward-bot'),
 'findings':_failed_target_findings([{'message':'drive_screw e:1 failed', 'step':'e:1'}], {'objects':[]})}))
'''
    # The exact model-facing brief is collected from each reference tree.
    fixture = json.loads((ROOT / 'tests/expected/launch-profile-surface.json').read_text())
    actual = subprocess.check_output([sys.executable, '-c', script], env=clean_env(profile), text=True)
    assert json.loads(actual) == fixture[profile]


def test_toy_cached_model_must_match_the_active_generator():
    script = """
import json
from steward.assembly_profile import validate_toy_model
from steward.assets.generators import steward_bot
from steward.assets.library import template_hash
robot = steward_bot().to_dict()
robot.update(id='w:robot-1', asset={'id':'steward-bot','version':template_hash(robot)})
scenario = {'robots':['w:robot-1'],'sceneState':{'entities':[robot]}}
validate_toy_model(scenario)
robot['asset']['version'] = 'old-default-model'
try:
    validate_toy_model(scenario)
except ValueError as exc:
    assert 'build-toy-assembly' in str(exc)
else:
    raise AssertionError('stale model accepted')
print(json.dumps(True))
"""
    assert json.loads(subprocess.check_output([sys.executable, '-c', script], env=clean_env('toy'), text=True))


@pytest.mark.parametrize('profile', ['default', 'toy'])
def test_cli_child_receives_selected_profile_and_requested_phase(profile):
    script = '''
import io, json, os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from steward import cli
seen = {}
def capture(args):
    seen.update(vars(args))
    return 0
cli.cmd_up = capture
cli.main(['--root','profile-root','up','--scenario','toy-assembly','--phase','phase1-sam'])
assert seen['phase'] == 'phase1-sam' and seen['scenario'] == 'toy-assembly'
assert seen['speed'] == 1 and seen['body'] is None and seen['dispatch_task']
def popen(cmd, **kwargs):
    assert cmd[1:3] == ['-m','steward.cli']
    assert '--segmentation-config' in cmd and cmd[-2:] == ['--body','steward-bot']
    assert kwargs['env'].get('STEWARD_ASSEMBLY_PROFILE') == os.environ.get('STEWARD_ASSEMBLY_PROFILE')
    assert kwargs['env']['PYTHONPATH'] == os.environ['PYTHONPATH']
    return SimpleNamespace(stdout=io.StringIO('{"egoApi":"http://localhost:7420"}\\n'))
with patch.object(cli.subprocess, 'Popen', popen):
    process = cli._EgoProcess(SimpleNamespace(**seen), 'http://localhost:7410', Path('out'),
                             embodiment='steward-bot', api_port=7420)
assert process.url == 'http://localhost:7420'
print(json.dumps(True))
'''
    assert json.loads(subprocess.check_output([sys.executable, '-c', script], env=clean_env(profile), text=True))


def test_installed_harness_check_rejects_mismatched_environment_and_stale_plugin(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('The installed Harness check requires Node.js and built plugins')
    checker = ROOT / 'dsh/scripts/check-ego-profile.mjs'
    module = ROOT / 'dsh/packages/dsh-tool-ego/lib/index.js'
    if not module.exists():
        pytest.skip('Build the Harness packages before checking installed plugins')
    good = subprocess.run([node, str(checker), '--expect', 'toy', '--module', str(module)],
                          env=clean_env('toy'), capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert 'complete_assembly' in json.loads(good.stdout)['skills']
    wrong_env = subprocess.run([node, str(checker), '--expect', 'toy', '--module', str(module)],
                               env=clean_env('default'), capture_output=True, text=True)
    assert wrong_env.returncode != 0 and 'does not match expected' in wrong_env.stderr
    stale = tmp_path / 'stale.mjs'
    stale.write_text(f"import * as plugin from {json.dumps(module.as_uri())};\n"
                     "export function apply(ctx) { for (const tool of plugin.egoTools(ctx.ego, 'off')) "
                     "ctx.tools.register(tool); }\n")
    wrong_module = subprocess.run([node, str(checker), '--expect', 'toy', '--module', str(stale)],
                                  env=clean_env('toy'), capture_output=True, text=True)
    assert wrong_module.returncode != 0 and 'exposes the wrong toy interface' in wrong_module.stderr
