"""Compare default behavior with bundled upstream expectations, in clean processes."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'steward' / 'src'


def upstream_reference():
    fixture = Path(__file__).with_name('fixtures') / 'upstream-defaults.json'
    return json.loads(fixture.read_text())


def clean_env(**overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith('STEWARD_') and k != 'SUBGOAL_SUCCESS'}
    env.update(PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.update(overrides)
    return env


def run(script, **overrides):
    return json.loads(subprocess.check_output([sys.executable, '-'], input=script, env=clean_env(**overrides), text=True))


def test_default_robot_and_runtime_match_upstream():
    script = '''
import json, sys, types
from dataclasses import asdict
from steward.assets.generators import steward_bot, sideboard
from steward.assets.cobot import cobot_magic
from steward.ego import skills
from steward.ego.runtime import RuntimeConfig
from steward.scene.fasteners import FastenConfig
actual = dict(robot=steward_bot().to_dict(), cobot=cobot_magic().to_dict(), runtime=asdict(RuntimeConfig()), fasten=asdict(FastenConfig.from_env({})),
              sideboard=sideboard(drawer_width=0.5).to_dict())
from types import SimpleNamespace
def drive_trace(module):
    trace = []
    ex = SimpleNamespace(config=module.MotionConfig())
    for method in ('_dock', '_align_lateral', '_restore_planned_reach', '_ramp_joint', '_set_joint',
                   '_reach_out', '_wait', '_exit_long_reach', '_backoff'):
        setattr(ex, method, lambda *args, method=method: trace.append((method, args[1:] if method=='_dock' else args)))
    job = module.Job('job', 'drive_screw', 'receipt')
    plan = dict(spot=[0.1,0.2], gripSpot=[0.11,0.22], lift=0.9, pressLift=0.85,
                wristYaw=0.5, wristRoll=-1.5707963267948966, driveS=1.5, target='screw', object='driver')
    module.SkillExecutor._drive_screw(ex, job, plan)
    return trace, job.log, job.message
assert hasattr(skills.SkillExecutor, '_insert_screw_into_hole')
print(json.dumps(dict(values=actual, drive_trace=drive_trace(skills))))
'''
    actual = run(script)
    reference = upstream_reference()
    assert actual['values'] == reference['values']
    assert actual['drive_trace'] == reference['drive_trace']


def test_profile_settings_and_explicit_overrides():
    script = '''
import json, os
from steward.assembly_profile import TOY_DEFAULTS
from steward.assets.generators import steward_bot
from steward.ego.runtime import RuntimeConfig
from steward.ego import capabilities
from steward.scene.fasteners import FastenConfig
from steward.world.support_fixture import enabled_from_env
print(json.dumps(dict(env={k:os.environ[k] for k in TOY_DEFAULTS},
 rate=RuntimeConfig().perception_hz, wrist=capabilities.get('STEWARD_WRIST_CAMERA'),
 fixture=enabled_from_env(), fasten=FastenConfig.from_env().mode,
 links=[l.id for l in steward_bot().rigid.links])))
'''
    result = run(script, STEWARD_ASSEMBLY_PROFILE='toy')
    assert result['env']['STEWARD_ASSEMBLY_SESSION_IDENTITY'] == '0'
    assert result['rate'] == 5
    assert result['wrist'] == '1'
    assert result['fixture'] is True
    assert result['fasten'] == 'off'
    assert 'hand_view' in result['links']
    result = run(script, STEWARD_ASSEMBLY_PROFILE='toy', STEWARD_ASSEMBLY_SESSION_IDENTITY='1',
                 STEWARD_WRIST_CAMERA='0', STEWARD_PERCEPTION_HZ='7', STEWARD_FASTEN_MODE='seat')
    assert result['env']['STEWARD_ASSEMBLY_SESSION_IDENTITY'] == '1'
    assert result['rate'] == 7
    assert result['wrist'] == '0'
    assert result['fasten'] == 'seat'


def test_capabilities_and_fixture_are_off_by_default():
    result = run('''
import json
from steward.ego import capabilities
from steward.world.support_fixture import enabled_from_env
print(json.dumps(dict(enabled=[k for k,v in capabilities.DEFAULTS.items()
 if v=='1' and capabilities.get(k)!='0'], fixture=enabled_from_env())))
''')
    assert result == {'enabled': [], 'fixture': False}


class _DefaultBranch(ast.NodeTransformer):
    def generic_visit(self, node):
        # Python 3.12 adds empty type-parameter lists to otherwise identical ASTs.
        if hasattr(node, 'type_params') and not node.type_params:
            del node.type_params
        return super().generic_visit(node)

    def visit_If(self, node):
        test, positive = node.test, True
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test, positive = test.operand, False
        if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == '_assembly_enabled':
            selected = node.orelse if positive else node.body
            result = []
            for child in selected:
                value = self.visit(child)
                if value is not None:
                    result.extend(value if isinstance(value, list) else [value])
            return result
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        return None if node.module == 'steward.assembly_profile' else node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def test_default_shared_code_matches_upstream():
    # Compare every shared module that contains a profile branch. This covers
    # camera extrinsics, planning, actuator sequences, serialization and world rules.
    paths = sorted(p for p in (SOURCE / 'steward').rglob('*.py')
                   if 'enabled as _assembly_enabled' in p.read_text())
    assert paths
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        actual_tree = _DefaultBranch().visit(ast.parse(path.read_text()))
        digest = hashlib.sha256(ast.dump(actual_tree).encode()).hexdigest()
        assert digest == upstream_reference()['default_ast_sha256'][relative], relative
    assert {p.relative_to(ROOT).as_posix() for p in paths} == set(upstream_reference()['default_ast_sha256'])


def test_harness_default_and_profile_match_reference_sources():
    import shutil
    node = shutil.which('node')
    if node is None:
        pytest.skip('Harness comparison requires Node.js')
    relative_tools = 'steward/dsh/packages/dsh-tool-ego/src/index.ts'
    relative_parser = 'steward/dsh/packages/dsh-ego-runtime-client/src/sensor-packet.ts'
    helper = ROOT / 'steward/dsh/test/assembly-profile-surface.mjs'

    def surface(tools, parser, **settings):
        return json.loads(subprocess.check_output([node, str(helper), str(tools), str(parser)],
                          env=clean_env(UV_THREADPOOL_SIZE='1', **settings), text=True))

    expected = upstream_reference()['harness']
    actual = surface(ROOT / relative_tools, ROOT / relative_parser)
    assert actual == expected
    recorded = json.loads(Path(__file__).with_name('expected').joinpath('assembly-tool-surface.json').read_text())
    assert surface(ROOT / relative_tools, ROOT / relative_parser, STEWARD_ASSEMBLY_PROFILE='toy') == recorded
