"""Collect the actual runtime dispatch and HTTP surface without a world or renderer.

Run in a fresh interpreter with PYTHONPATH selecting the implementation under test.
Only the view renderer and motion leaves are doubles; constructors, prediction,
execution dispatch, route construction and API registration are production code.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

SKILLS = ('complete_assembly', 'regrasp_upright', 'assemble_onto', 'fasten_onto',
          'seat_screw', 'drive_screw', 'flip_assembly', 'insert_screw_into_hole',
          'navigate', 'pick', 'place', 'pour', 'open', 'close',
          '_assembly_observe', '_inhand_view', '_pivot_align', '_simple_driver',
          '_fastener_contact', '_fastener_reset', '_stage_screw', '_upright_tool', '__invalid__')


def surface(check_catalog=False):
    from steward.assets.generators import steward_bot
    from steward.ego.runtime import EgoRuntime
    from steward.ego.skills import Job
    from steward.ego import server

    renderer = types.ModuleType('steward.ego.view')
    renderer.EgoViewRenderer = lambda: SimpleNamespace(close=lambda: None)
    with patch.dict(sys.modules, {'steward.ego.view': renderer}):
        runtime = EgoRuntime(SimpleNamespace(), steward_bot().to_dict())
    runtime.ego_state = lambda: runtime.estimator.to_scene()
    runtime._observe_decision = lambda *args, **kwargs: None
    runtime._poll_once = lambda: None
    predictions = {}
    for skill in SKILLS:
        try:
            receipt = runtime.predict(skill, {'x': 0, 'y': 0})
        except KeyError as exc:
            # Internal primitives read mandatory parameters before producing a receipt.
            predictions[skill] = {'requiredParameter': str(exc)}
        else:
            predictions[skill] = not any('unknown skill' in issue for issue in receipt['issues'])

    if check_catalog:
        from steward.ego.skill_catalog import PUBLIC_SKILLS
        accepted = {skill for skill, value in predictions.items() if value and not skill.startswith('_')}
        assert accepted == set(PUBLIC_SKILLS)
        advertised = runtime.predict('__invalid__', {})['issues'][0].split('known: ', 1)[1]
        assert set(advertised.split(', ')) == set(PUBLIC_SKILLS)

    executor = runtime.executor
    executor._clock = executor._on_finish = executor._on_error = executor._home_empty = None
    executor._begin = lambda *args: None
    executor._state = lambda: {'stamp': 0}
    calls = []
    for method in ('_fasten', '_assemble_onto', '_drive', '_pick', '_place', '_place_floor',
                   '_open', '_close', '_pour', '_drive_screw', '_insert_screw_into_hole'):
        if hasattr(executor, method):
            setattr(executor, method, lambda *args, method=method: calls.append(method))
    executions = {}
    for skill in SKILLS:
        calls.clear()
        job = Job('job', skill, 'receipt')
        executor._run(job, skill, {'skill': skill, 'plan': {'path': [], 'yaw': 0}})
        executions[skill] = {'state': job.state, 'dispatch': list(calls)}

    routes = server._routes(runtime)
    # This startup read must never apply an input or advance the simulation.
    assert ('GET', '/ego/control') in server.UNRECORDED
    from steward.assembly_profile import enabled
    expected_control = dict(runtime.control())
    if enabled():
        expected_control['assemblyProfile'] = 'toy'
    assert routes['GET', '/ego/control']({}, {}) == expected_control
    tree = ast.parse(inspect.getsource(server))
    lambdas = {n.lineno: ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Lambda)}
    route_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_routes')
    helpers = {n.name: ast.unparse(n) for n in route_function.body if isinstance(n, ast.FunctionDef)}
    parameters = {f'{method} {path}': lambdas[fn.__code__.co_firstlineno]
                  for (method, path), fn in routes.items()}
    with patch.object(server, 'JsonServer', side_effect=lambda handlers, **kwargs: handlers):
        registered = server.ego_api(runtime)
    with patch.object(runtime, 'apply', side_effect=AssertionError('profile lookup must be unrecorded')):
        assert registered['GET', '/ego/control']({}, {}) == expected_control
    # Include the separately installed cancellable face-match handler.
    assert set(registered) == set(routes)
    return {'prediction': predictions, 'execution': executions,
            'routes': sorted(f'{method} {path}' for method, path in registered),
            'routeParameters': parameters, 'routeHelpers': helpers,
            'unrecorded': sorted(f'{method} {path}' for method, path in server.UNRECORDED)}


if __name__ == '__main__':
    print(json.dumps(surface(check_catalog='--check-catalog' in sys.argv), sort_keys=True))
