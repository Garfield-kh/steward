"""Assemble the fixed review question sheet and ask the configured advisor.

Inputs are the outputs of ``score`` (``score.json``), short digests of each episode's receipt logs
and, optionally, rendered frames. The advisor's answer is written to ``advice.md`` next to the exact
prompt (``review_prompt.md``) so every recommendation can be traced to what the model saw.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .providers import Image, Provider

SYSTEM = (
    "You advise a team that improves a simulated mobile robot on a toy-assembly task by changing one "
    "thing at a time. A separate controller model drives the robot through Steward skills; you only "
    "read statistics, event digests and images. Base every claim on the evidence given, say when the "
    "evidence is insufficient, and do not invent numbers."
)

QUESTIONS = """\
Answer these four questions, each under its own heading, in {language}:

1. Worst stage - Which stage of the funnel (placed, aligned, seated, complete) loses the most? Cite the counts.
2. Most likely cause - What most likely causes the loss at that stage? Cite events, numbers or images. If several causes fit, rank them and say what evidence would separate them.
3. One change - Name exactly one change to try next and where it is made (the task text given to the controller, an operator note, a skill parameter, or a STEWARD_* setting). Be concrete enough that someone can apply it without asking you.
4. Verify from an intermediate state - Say how to test the change without whole episodes: which episode and elapsed time to export with `python -m steward_rsi save-start` (just before the worst stage), how many episodes to start from that state, which counts must improve, and which whole-episode regression check must still pass before the change is kept.
"""

MAX_EXITS = 8
MAX_FAILURES = 6


def _jsonl(paths):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def receipt_digest(receipts_dir: Path) -> dict:
    """Counts and a few recent rows from ``receipts-*.jsonl`` and ``assembly-sessions-*.jsonl``."""
    if not receipts_dir.is_dir():
        return {"available": False}
    skills = collections.Counter()
    failures = []
    for row in _jsonl(sorted(receipts_dir.glob("receipts-*.jsonl"))):
        if not row.get("skill"):
            continue
        receipt = row.get("receipt") or {}
        state = receipt.get("state", "unknown")
        skills[f"{row['skill']}:{state}"] += 1
        if state not in ("succeeded", "running"):
            failures.append({"skill": row["skill"], "state": state, "message": str(receipt.get("message", ""))[:200]})
    events = collections.Counter()
    recoveries = collections.Counter()
    exits = []
    last_holes = None
    for row in _jsonl(sorted(receipts_dir.glob("assembly-sessions-*.jsonl"))):
        event = row.get("event")
        if event == "heartbeat":
            continue
        events[event] += 1
        if event == "recovery":
            recoveries[str(row.get("reason"))] += 1
        if event == "exit":
            exits.append({"job": row.get("job"), "decision": row.get("decision"), "reason": row.get("reason"),
                          "simTime": round(float(row.get("simTime") or 0.0), 1)})
            if row.get("holeOutcomes"):
                last_holes = [{k: h.get(k) for k in ("hole", "state", "released")} for h in row["holeOutcomes"]]
    return {"available": True, "skillOutcomes": dict(skills), "recentFailures": failures[-MAX_FAILURES:],
            "assemblyEvents": dict(events), "recoveryReasons": dict(recoveries),
            "lastExits": exits[-MAX_EXITS:], "lastHoleOutcomes": last_holes}


def _progress(result: dict) -> tuple:
    if not result.get("valid"):
        return (-1, -1, -1)
    c = result["counts"]
    return (c["seated"], c["aligned"], c["placed"])


def choose_episodes(results: list[dict], limit: int = 4) -> list[dict]:
    """Least-progressed episodes first, plus one completed episode for contrast when there is one."""
    valid = [r for r in results if r.get("valid")]
    chosen = sorted(valid, key=_progress)[:limit]
    done = next((r for r in valid if r["complete"] and r not in chosen), None)
    if done is not None:
        chosen = chosen[: limit - 1] + [done]
    return chosen


def collect_images(run_dirs: dict[str, Path], chosen: list[dict], limit: int) -> list[Image]:
    images = []
    for result in chosen:
        frames = run_dirs.get(result["run"], Path("/nonexistent")) / "frames"
        if not frames.is_dir():
            continue
        named = [frames / f"final-{view}.{ext}" for view in ("kit", "overview") for ext in ("jpg", "png")]
        timed = sorted([*frames.glob("t*-kit.jpg"), *frames.glob("t*-kit.png")])[-1:]
        for path in [*named, *timed]:
            if path.is_file() and len(images) < limit:
                images.append(Image.from_file(path, f"{result['run']} {path.stem}"))
    return images


def build_prompt(summary: dict, results: list[dict], digests: dict[str, dict], task_text: str,
                 funnel_md: str, language: str, image_labels: list[str]) -> str:
    chosen = [{k: r.get(k) for k in ("run", "seed", "endReason", "wallS", "counts", "maxLateralMm",
                                     "seatedMilestonesWallS", "holeFirstSeatedWallS", "supportedByKitTable", "kitHeldByRobot",
                                     "foreignMoved", "foreignGrasped", "foreignScrewsUsed", "complete",
                                     "completeStrict")}
              | {"interfaces": [{k: i.get(k) for k in ("hole", "placed", "aligned", "seated", "lateralMm", "screw")}
                                for i in r.get("interfaces", [])],
                 "events": digests.get(r["run"], {"available": False})}
              for r in results]
    parts = [
        "## Task given to the controller", "", task_text.strip()[:6000], "",
        "## Funnel", "", funnel_md.strip(), "",
        "Stage definitions: placed = a hole of the right part is within the goal position tolerance of the body "
        "hole with collinear axes; aligned = additionally its lateral offset is within the threshold; seated = "
        "additionally a screw fastens the two holes; complete = every hole seated with an authorized screw, "
        "assembly resting on its own table, not held, and nothing from the other kit moved or used.", "",
        f"Worst stage by conversion: {summary.get('worstStage')}", "",
        "## Selected episodes (least progress first)", "", "```json", json.dumps(chosen, indent=1)[:40000], "```", "",
    ]
    if image_labels:
        parts += ["## Images", "", "Attached in order: " + "; ".join(image_labels), ""]
    parts += ["## Questions", "", QUESTIONS.format(language=language)]
    return "\n".join(parts)


def review(score_path: Path, run_dirs: dict[str, Path], provider: Provider, task_text: str, language: str,
           out_dir: Path) -> Path:
    data = json.loads(score_path.read_text())
    summary, results = data["summary"], data["episodes"]
    funnel_md = (score_path.parent / "funnel.md").read_text() if (score_path.parent / "funnel.md").is_file() else ""
    chosen = choose_episodes(results)
    digests = {r["run"]: receipt_digest(run_dirs[r["run"]] / "receipts") for r in chosen if r["run"] in run_dirs}
    cfg = provider.cfg
    images = collect_images(run_dirs, chosen, cfg.max_images) if cfg.send_images and cfg.max_images else []
    prompt = build_prompt(summary, chosen, digests, task_text, funnel_md, language, [i.label for i in images])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review_prompt.md").write_text(f"<!-- system -->\n{SYSTEM}\n\n<!-- user -->\n{prompt}\n")
    reply = provider.complete(SYSTEM, prompt, images)
    advice = out_dir / "advice.md"
    advice.write_text(f"<!-- advisor: {cfg.name} model={reply.model} finish={reply.finish} "
                      f"images={len(images)} usage={json.dumps(reply.usage)} -->\n\n{reply.text.strip()}\n")
    return advice
