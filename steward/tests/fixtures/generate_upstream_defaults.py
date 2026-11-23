"""Freeze upstream expectations from an exported source tree; see README.md."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream_tree', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('upstream-defaults.json'))
    args = parser.parse_args()
    test = Path(__file__).parents[1] / 'test_assembly_profile_defaults.py'
    spec = importlib.util.spec_from_file_location('default_comparison', test)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)

    def reference(path):
        source = (args.upstream_tree / path).read_text()
        if path.endswith('assets/generators.py'):
            # Preserve the documented drawerless-sideboard indentation correction.
            source = source.replace('\n    return _entity("sideboard", "object", "fixed", [link])',
                                    '\n        return _entity("sideboard", "object", "fixed", [link])')
        return source

    names = ('assets/generators', 'assets/cobot', 'ego/runtime', 'ego/skills', 'scene/fasteners')
    sources = {name: reference(f'steward/src/steward/{name}.py') for name in names}
    # Reuse the test's observation procedure; all observed implementations below
    # come from the exported upstream tree, never from the implementation under test.
    function = next(n for n in ast.parse(test.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'test_default_robot_and_runtime_match_upstream')
    script = next(n.value for n in ast.walk(function) if isinstance(n, ast.Constant)
                  and isinstance(n.value, str) and 'def drive_trace(module)' in n.value)
    prefix = '''import json, sys, types
from dataclasses import asdict
modules = {}
for key, source in SOURCES.items():
    name = 'baseline_' + key.replace('/', '_')
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, 'exec'), module.__dict__)
    modules[key] = module
steward_bot = modules['assets/generators'].steward_bot
sideboard = modules['assets/generators'].sideboard
cobot_magic = modules['assets/cobot'].cobot_magic
skills = modules['ego/skills']
RuntimeConfig = modules['ego/runtime'].RuntimeConfig
FastenConfig = modules['scene/fasteners'].FastenConfig
'''.replace('SOURCES', repr(sources))
    values = comparison.run(prefix + script[script.index('actual = dict('):])
    fingerprints = {}
    paths = sorted(p for p in (comparison.SOURCE / 'steward').rglob('*.py')
                   if 'enabled as _assembly_enabled' in p.read_text())
    for path in paths:
        relative = path.relative_to(comparison.ROOT).as_posix()
        tree = comparison._DefaultBranch().visit(ast.parse(reference(relative)))
        fingerprints[relative] = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as temp:
        tools, packet = Path(temp) / 'tools.ts', Path(temp) / 'parser.ts'
        tools.write_text(reference('steward/dsh/packages/dsh-tool-ego/src/index.ts'))
        packet.write_text(reference('steward/dsh/packages/dsh-ego-runtime-client/src/sensor-packet.ts'))
        helper = comparison.ROOT / 'steward/dsh/test/assembly-profile-surface.mjs'
        harness = json.loads(subprocess.check_output(['node', str(helper), str(tools), str(packet)],
                                                    env=comparison.clean_env(), text=True))
    result = dict(source='upstream default reference snapshot', **values,
                  default_ast_sha256=fingerprints, harness=harness)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
