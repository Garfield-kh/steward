# Assembly reference fixtures

English | [中文](README.zh.md)

`upstream-defaults.json` freezes the expected robot and Cobot models, runtime and fastening defaults, a drawer-bearing sideboard, screw-driving call trace, 31 default-branch AST fingerprints, and the Harness tool/camera reference. It was generated from public upstream `b0af02e596da18316ca815f15b89a23ea2b55608`, not from the assembly implementation. The only source correction is the documented indentation of the drawerless sideboard return. Metadata under `source` is descriptive and is not a Git dependency.

To regenerate deliberately from the repository root (Python 3.11, NumPy, pytest and Node.js available):

```sh
mkdir -p steward/var/upstream-reference
git archive b0af02e596da18316ca815f15b89a23ea2b55608 steward/src \
  steward/dsh/packages/dsh-tool-ego/src steward/dsh/packages/dsh-ego-runtime-client/src \
  | tar -x -C steward/var/upstream-reference
PYTHONPATH=steward/src python steward/tests/fixtures/generate_upstream_defaults.py \
  steward/var/upstream-reference
```

Only this manual generation step needs upstream history. Tests never invoke it. Review regenerated expectations independently; do not update the reference to make an implementation regression pass. The generator reads reference implementations from the exported tree, executes their constructors and driving routine, and uses the same observation/AST projection as the test. AST fingerprints use Python's AST representation, omitting empty type-parameter lists introduced in Python 3.12. This reference was generated and verified with Python 3.11.
