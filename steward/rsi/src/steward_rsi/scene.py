"""Derive the assembly kits of the toy-assembly scenario from its definition and a World State.

The scenario goal (``toy_assembly_goal`` in ``steward/authoring/service.py``) lists structural parts
and their relative poses. Its constraints form connected groups; each group is a *kit* whose
*body* is the entity that appears as ``a``. The scene definition (``toy_assembly_spec``) places
every entity ``on`` a support. From these two and a World State snapshot:

* screw holes: ``screw-hole`` affordances of the body and of the kit parts (World State);
* which part asset belongs on which body hole: the goal constraint whose ``offset`` lies nearest
  to the hole's position in the body frame (parts of one asset are interchangeable);
* the kit table: the support the scene definition places the body ``on``;
* authorized screws: entities with a ``screw`` affordance that the scene definition places on the
  kit table.

Table and screws come from the scene definition, so a start state exported mid-assembly (fastened
screws, a body resting on its parts) gives the same kit as a scratch start.

No part names are written in this module.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path

SERVICE_RELATIVE = Path("steward/src/steward/authoring/service.py")
BUNDLED_GOAL = Path(__file__).resolve().parent / "data" / "toy_assembly_goal.json"
BUNDLED_SPEC = Path(__file__).resolve().parent / "data" / "toy_assembly_spec.json"


# ------------------------------------------------------------------ goal
def goal_from_service(service_py: Path, function: str = "toy_assembly_goal") -> dict:
    """Literal return value of ``function`` in ``service_py``, read with ``ast`` (nothing is imported)."""
    tree = ast.parse(service_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    return ast.literal_eval(stmt.value)
    raise LookupError(f"{function} not found in {service_py}")


def load_goal(repo_root: Path | None) -> tuple[dict, str]:
    """The scenario goal and its source: the repository definition, else the bundled copy."""
    if repo_root is not None:
        service = repo_root / SERVICE_RELATIVE
        if service.is_file():
            return goal_from_service(service), str(SERVICE_RELATIVE)
    return json.loads(BUNDLED_GOAL.read_text()), "bundled toy_assembly_goal.json"


class _SpecEvaluator:
    """Evaluate the statements of a scene-spec function without importing or executing its module.

    Accepted: literals, names assigned earlier in the function, arithmetic, f-strings, list
    comprehensions over ``range(...)``, starred list items and ``math`` constants. Anything else
    raises :class:`ValueError`.
    """

    BINARY = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
              ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
              ast.Pow: lambda a, b: a ** b}
    UNARY = {ast.USub: lambda a: -a, ast.UAdd: lambda a: +a}
    MATH = {"pi": math.pi, "tau": math.tau, "e": math.e}

    def __init__(self):
        self.names: dict = {}

    def run(self, function: ast.FunctionDef):
        for stmt in function.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # docstring
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                self.names[stmt.targets[0].id] = self.eval(stmt.value)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                return self.eval(stmt.value)
            else:
                raise ValueError(f"unsupported statement at line {stmt.lineno}: {type(stmt).__name__}")
        raise ValueError(f"{function.name} has no return statement")

    def eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            out = []
            for item in node.elts:
                if isinstance(item, ast.Starred):
                    out.extend(self.eval(item.value))
                else:
                    out.append(self.eval(item))
            return out if isinstance(node, ast.List) else tuple(out)
        if isinstance(node, ast.Dict):
            out = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    out.update(self.eval(value))
                else:
                    out[self.eval(key)] = self.eval(value)
            return out
        if isinstance(node, ast.Name):
            if node.id in self.names:
                return self.names[node.id]
            raise ValueError(f"unknown name {node.id!r} at line {node.lineno}")
        if isinstance(node, ast.BinOp) and type(node.op) in self.BINARY:
            return self.BINARY[type(node.op)](self.eval(node.left), self.eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY:
            return self.UNARY[type(node.op)](self.eval(node.operand))
        if isinstance(node, ast.JoinedStr):
            return "".join(self.eval(v) for v in node.values)
        if isinstance(node, ast.FormattedValue):
            spec = self.eval(node.format_spec) if node.format_spec is not None else ""
            return format(self.eval(node.value), spec)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math"
                and node.attr in self.MATH):
            return self.MATH[node.attr]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
                and not node.keywords):
            return range(*(self.eval(a) for a in node.args))
        if isinstance(node, ast.ListComp):
            return list(self._comprehension(node.elt, node.generators))
        raise ValueError(f"unsupported expression at line {node.lineno}: {type(node).__name__}")

    def _comprehension(self, elt, generators):
        if not generators:
            yield self.eval(elt)
            return
        gen, rest = generators[0], generators[1:]
        if not isinstance(gen.target, ast.Name) or gen.is_async:
            raise ValueError(f"unsupported comprehension target at line {gen.target.lineno}")
        saved = dict(self.names)
        try:
            for value in self.eval(gen.iter):
                self.names[gen.target.id] = value
                if all(self.eval(cond) for cond in gen.ifs):
                    yield from self._comprehension(elt, rest)
        finally:
            self.names = saved


def spec_from_service(service_py: Path, function: str = "toy_assembly_spec") -> dict:
    """Return value of the scene-spec ``function`` in ``service_py``, read with ``ast`` (nothing is imported)."""
    tree = ast.parse(service_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return _SpecEvaluator().run(node)
    raise LookupError(f"{function} not found in {service_py}")


def load_spec(repo_root: Path | None) -> tuple[dict, str]:
    """The toy-assembly scene definition and its source: the repository definition, else the bundled copy."""
    if repo_root is not None:
        service = repo_root / SERVICE_RELATIVE
        if service.is_file():
            return spec_from_service(service), str(SERVICE_RELATIVE)
    return json.loads(BUNDLED_SPEC.read_text()), "bundled toy_assembly_spec.json"


# ------------------------------------------------------------------ geometry
def quat_rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    # v + 2w(u×v) + 2u×(u×v), u = (x, y, z)
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    ccx, ccy, ccz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return [vx + 2 * (w * cx + ccx), vy + 2 * (w * cy + ccy), vz + 2 * (w * cz + ccz)]


def quat_angle_deg(q1, q2) -> float:
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def norm(a):
    return math.sqrt(dot(a, a))


# ------------------------------------------------------------------ world state access
class World:
    """Indexed view of one World State dictionary."""

    def __init__(self, state: dict):
        self.state = state
        self.entities = {e["id"]: e for e in state.get("entities", [])}
        self.relations = state.get("relations", [])

    def pose(self, eid: str) -> dict:
        return self.entities[eid]["pose"]["value"]

    def asset(self, eid: str) -> str:
        return (self.entities[eid].get("asset") or {}).get("id", "")

    def affordances(self, eid: str):
        for link in self.entities[eid].get("body", {}).get("links", []):
            for item in link.get("affordances", []):
                value = item.get("value")
                if isinstance(value, dict):
                    yield value

    def holes(self, eid: str) -> dict[str, dict]:
        return {a["screw-hole"]["id"]: a["screw-hole"] for a in self.affordances(eid) if "screw-hole" in a}

    def is_screw(self, eid: str) -> bool:
        return any("screw" in a for a in self.affordances(eid))

    def is_free_object(self, eid: str) -> bool:
        e = self.entities[eid]
        return e.get("role") == "object" and (e.get("body", {}).get("root") or {}).get("value") == "free"

    def hole_world(self, eid: str, hole_id: str) -> tuple[list, list]:
        """World-frame entry point and axis of a hole."""
        hole = self.holes(eid)[hole_id]
        pose = self.pose(eid)
        entry = hole.get("entry") or hole["p"]
        p = [pose["p"][i] + v for i, v in enumerate(quat_rotate(pose["q"], entry))]
        return p, quat_rotate(pose["q"], hole.get("axis") or [0, 0, 1])

    def supports(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for r in self.relations:
            if r.get("type") == "supported-by":
                out.setdefault(r["a"]["entity"], set()).add(r["b"]["entity"])
        return out

    def support_closure(self, start: set[str]) -> set[str]:
        sup = self.supports()
        seen = set(start)
        frontier = list(start)
        while frontier:
            for nxt in sup.get(frontier.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    def held_by_robot(self) -> set[str]:
        robots = {eid for eid, e in self.entities.items() if e.get("role") == "robot"}
        held = set()
        for r in self.relations:
            if r.get("type") != "attached":
                continue
            a, b = r["a"]["entity"], r["b"]["entity"]
            if a in robots:
                held.add(b)
            if b in robots:
                held.add(a)
        return held

    def fasteners(self) -> dict[str, list[dict]]:
        """Screw entity id -> list of ``{entity, hole}`` it currently fastens."""
        out = {}
        for eid, e in self.entities.items():
            value = ((e.get("stateVars") or {}).get("fastener") or {}).get("value") or {}
            if value.get("holes"):
                out[eid] = value["holes"]
        return out


# ------------------------------------------------------------------ kits
@dataclasses.dataclass
class Kit:
    body: str
    parts: list[str]
    table: str
    screws: list[str]
    # body hole id -> asset id of the part that belongs there
    hole_asset: dict[str, str]
    position_tol_m: float

    @property
    def members(self) -> set[str]:
        return {self.body, *self.parts}


def kit_groups(goal: dict) -> list[tuple[str, list[str], list[dict]]]:
    """``(body, parts, constraints)`` per connected group of goal constraints."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in goal["constraints"]:
        parent[find(c["a"])] = find(c["b"])
    groups: dict[str, list[dict]] = {}
    for c in goal["constraints"]:
        groups.setdefault(find(c["a"]), []).append(c)
    out = []
    for constraints in groups.values():
        bodies = {c["a"] for c in constraints}
        if len(bodies) != 1:
            raise ValueError(f"goal group with more than one anchor entity: {sorted(bodies)}")
        body = bodies.pop()
        out.append((body, [c["b"] for c in constraints], constraints))
    return out


def placements(spec: dict) -> dict[str, str]:
    """Entity id -> id of the entity the scene definition places it ``on``."""
    return {e["id"]: e["on"] for e in spec.get("entities", []) if e.get("on")}


def derive_kits(goal: dict, spec: dict, initial: World) -> list[Kit]:
    """Kits of ``goal``; tables and authorized screws from ``spec``, hole geometry from ``initial``."""
    tol = float(goal.get("tolerance", {}).get("positionM", 0.04))
    on = placements(spec)

    def rests_on(eid: str, table: str) -> bool:
        seen = set()
        while eid in on and eid not in seen:
            seen.add(eid)
            eid = on[eid]
            if eid == table:
                return True
        return False

    kits = []
    for body, parts, constraints in kit_groups(goal):
        table = on.get(body)
        if not table:
            raise ValueError(f"the scene definition does not place {body} on a support")
        hole_asset = {}
        for hole_id, hole in initial.holes(body).items():
            x, y = hole["p"][0], hole["p"][1]
            nearest = min(constraints, key=lambda c: math.hypot(c["offset"][0] - x, c["offset"][1] - y))
            hole_asset[hole_id] = initial.asset(nearest["b"])
        screws = sorted(eid for eid in initial.entities if initial.is_screw(eid) and rests_on(eid, table))
        kits.append(Kit(body=body, parts=parts, table=table, screws=screws, hole_asset=hole_asset,
                        position_tol_m=tol))
    return kits


def select_kit(kits: list[Kit], initial: World, task_name: str, kit_body: str = "") -> Kit:
    """The kit whose body is ``kit_body``, else whose body asset id has ``task_name`` as a token."""
    if kit_body:
        for kit in kits:
            if kit.body == kit_body:
                return kit
        raise ValueError(f"kit_body {kit_body!r} is not a goal anchor; anchors: {[k.body for k in kits]}")
    matches = [k for k in kits
               if task_name in initial.asset(k.body).replace("_", "-").split("-")]
    if len(matches) != 1:
        raise ValueError(f"task {task_name!r} matches {len(matches)} kits by body asset "
                         f"({[initial.asset(k.body) for k in kits]}); set [task].kit_body")
    return matches[0]
