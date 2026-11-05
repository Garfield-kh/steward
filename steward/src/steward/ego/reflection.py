"""Evidence for the robot agent's reflection on a failed step: what kind of failure it was, what the
robot knows about the objects involved, and which objects could be the thing the instruction meant.

Perception names an object from its shape and, with robot-side segmentation, from the word the
segmenter matched; neither knows what the operator calls it. A puzzle box printed with a picture
can be segmented as ``picture`` and a flat lid has no prototype at all, so a step aimed at "the
box" fails on an object that exists under another name. Each object therefore carries its naming
evidence (perceived alternatives, segmenter names and the prompt words behind them, size against
the prototype of its category) and the doubts that evidence raises; a query ranks objects by how
well the evidence fits a name and an approximate size. Everything comes from the Ego-State.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from steward.diagnosis import classify_failure
from steward.ego.classify import SHAPES, UNKNOWN, prototype


@dataclass
class ReflectionConfig:
    candidate_limit: int = 6  # candidates returned when the caller gives no limit
    doubt_confidence: float = 0.6  # a category held with less probability than this is doubtful
    # A measured extent off its category's prototype by more than this (log ratio, any axis) is doubtful.
    size_tolerance: float = 0.6
    size_scale: float = 0.35  # log-ratio scale of the size fit exp(-mean |log ratio| / scale)
    # A measured dimension below the sought one costs this share of its log ratio: objects are often
    # seen only in part (a box seen from above measures as its lid), rarely larger than they are.
    size_under_weight: float = 0.25
    # An object whose bottom lies this far above the top of its support has a part below it unseen.
    hidden_gap_m: float = 0.02
    # Match score of a name sharing a word with the sought one ("dinner table" for "table").
    shared_word_score: float = 0.8
    # Failure causes (steward.diagnosis keys) for which other objects may be the one meant; a
    # blocked path or a full surface is about the scene, not about which object was named.
    candidate_causes: tuple[str, ...] = ("perception-missing", "out-of-reach", "low-confidence", "grasp-miss",
                                         "support-unknown", "unknown")
    min_alternative: float = 0.05  # perceived categories below this probability are not listed


@dataclass(frozen=True)
class Evidence:
    """What perception accumulated for one object, apart from any label the agent gave it."""

    perceived: dict[str, float]  # normalised category probabilities from perception alone
    segmenter: dict[str, float]  # mean segmenter score per category over the views that named it
    label: str | None = None


def _dims(size) -> list[float]:
    return sorted((float(v) for v in size), reverse=True)


def _words(name: str) -> set[str]:
    return {w for w in re.split(r"[\s_\-]+", name.lower()) if w}


def size_fit(extent, size, scale: float, under_weight: float = 1.0) -> float:
    """How well a measured extent matches an approximate size, in (0, 1].

    :param extent: measured full extents, metres, any order.
    :param size: one to three approximate dimensions, metres, any order; compared largest first
        with the largest measured dimensions.
    :param scale: log-ratio scale.
    :param under_weight: share of the log ratio a measured dimension below the sought one costs.
    :returns: ``exp(-mean cost / scale)`` with cost ``|log ratio|``, times ``under_weight`` when
        the measured dimension is the smaller."""
    got, want = _dims(extent), _dims(size)[:3]
    costs = [math.log(max(g, 1e-3) / max(w, 1e-3)) for g, w in zip(got, want)]
    costs = [c if c >= 0 else -c * under_weight for c in costs]
    return math.exp(-sum(costs) / len(costs) / scale)


def hidden_gap(obj: dict, support: dict | None) -> float:
    """How far an object's bottom lies above the top of the object it rests on, metres (0 without a support).

    :param obj: the object's Ego API summary entry.
    :param support: the summary entry of the object it rests on, if known.
    :returns: the gap; positive when something below the object is not seen."""
    if support is None:
        return 0.0
    ext, sup = obj.get("extentM") or obj.get("sizeM"), support.get("extentM") or support.get("sizeM")
    if not ext or not sup:
        return 0.0
    return float(obj["position"][2] - ext[2] / 2 - (support["position"][2] + sup[2] / 2))


def doubts(obj: dict, evidence: Evidence | None, prompt_words: dict[str, list[str]], cfg: ReflectionConfig,
           support: dict | None = None) -> list[str]:
    """Reasons to doubt the name an object goes by.

    :param obj: the object's Ego API summary entry.
    :param evidence: its perception evidence, if it is a perceived track.
    :param prompt_words: segmenter prompt words per category.
    :param cfg: thresholds.
    :param support: the summary entry of the object it rests on, if known.
    :returns: one sentence per doubt; empty when the name is well supported."""
    found = []
    category = obj["category"]
    if evidence is not None and evidence.label:
        found.append(f"you labelled it {evidence.label}; perception alone calls it "
                     f"{max(evidence.perceived, key=evidence.perceived.get) if evidence.perceived else UNKNOWN}")
    if category == UNKNOWN:
        found.append("perception could not name it")
    elif category not in SHAPES and (evidence is None or not evidence.label):
        found.append(f"the robot has no shape knowledge of a {category}: the name comes from appearance alone")
    if category != UNKNOWN and evidence is not None and not evidence.label:
        p = evidence.perceived.get(category, 0.0)
        if p < cfg.doubt_confidence:
            found.append(f"only {p:.0%} sure it is a {category}")
    if evidence is not None and evidence.segmenter:
        top = max(evidence.segmenter, key=evidence.segmenter.get)
        if top != category:
            words = "/".join(prompt_words.get(top, [])) or top
            found.append(f"the segmenter saw it as {top} (prompt {words}, score {evidence.segmenter[top]:.2f}), not {category}")
    gap = hidden_gap(obj, support)
    if gap > cfg.hidden_gap_m:
        found.append(f"its bottom lies {gap:.2f} m above the top of {obj['on']}: a part below it is not seen "
                     f"(it may be the top of something taller, or rest on something unseen)")
    proto = prototype(category)
    extent = obj.get("extentM") or obj.get("sizeM")
    if proto is not None and extent:
        off = max(abs(math.log(max(g, 1e-3) / w)) for g, w in zip(_dims(extent), _dims(proto)))
        if off > cfg.size_tolerance:
            found.append(f"its size {'x'.join(f'{v:.2f}' for v in _dims(extent))} m is unlike a typical {category} "
                         f"({'x'.join(f'{v:.2f}' for v in _dims(proto))} m)")
    return found


def describe(obj: dict, evidence: Evidence | None, prompt_words: dict[str, list[str]], cfg: ReflectionConfig,
             support: dict | None = None) -> dict:
    """An object with its naming evidence and doubts, for the agent to compare with the instruction.

    :param obj: the object's Ego API summary entry.
    :param evidence: its perception evidence, if any.
    :param prompt_words: segmenter prompt words per category.
    :param cfg: thresholds.
    :param support: the summary entry of the object it rests on, if known.
    :returns: id, names, evidence, size, colour, support, position and doubts."""
    entry = {
        "id": obj["id"],
        "category": obj["category"],
        "role": obj.get("role"),
        "extentM": obj.get("extentM") or obj.get("sizeM"),
        "color": obj.get("color"),
        "on": obj.get("on"),
        "position": obj.get("position"),
        "lastSeenAgoS": obj.get("lastSeenAgoS"),
    }
    if evidence is not None:
        entry["perceived"] = {k: round(v, 3) for k, v in sorted(evidence.perceived.items(), key=lambda kv: -kv[1])
                              if v >= cfg.min_alternative}
        if evidence.segmenter:
            entry["segmenter"] = {k: {"score": round(v, 3), "prompts": prompt_words.get(k, [])}
                                  for k, v in sorted(evidence.segmenter.items(), key=lambda kv: -kv[1])}
        if evidence.label:
            entry["label"] = evidence.label
    entry["doubts"] = doubts(obj, evidence, prompt_words, cfg, support)
    return entry


def match(obj: dict, evidence: Evidence | None, name: str | None, size, cfg: ReflectionConfig,
          support: dict | None = None) -> tuple[float, list[str]]:
    """How well an object fits a sought name and approximate size.

    The score adds the perceived probability of the name (1 for a matching role, label or category,
    ``shared_word_score`` for one sharing a word with it),
    the segmenter's mean score for it, and the size fit (against ``size``, or else against the
    name's prototype when it has one); a doubted name adds a little, since a misnamed object is
    what a failed lookup by name misses.

    :param obj: the object's Ego API summary entry.
    :param evidence: its perception evidence, if any.
    :param name: the sought category, role or vocabulary word, if any.
    :param size: approximate dimensions in metres, if any.
    :param cfg: thresholds.
    :param support: the summary entry of the object it rests on; a hidden part below the object
        counts towards its height.
    :returns: the score and the reasons behind it."""
    score, why = 0.0, []
    if name:
        sought = _words(name)
        shared = next((n for n in (obj.get("role"), obj["category"]) if n and n != name and sought & _words(n)), None)
        if name in (obj.get("role"), obj["category"]):
            score += 1.0
            why.append(f"it goes by {name}")
        elif shared:
            score += cfg.shared_word_score
            why.append(f"its name {shared} shares a word with {name}")
        else:
            p = (evidence.perceived.get(name, 0.0) if evidence else obj.get("categories", {}).get(name, 0.0))
            if p >= cfg.min_alternative:
                score += p
                why.append(f"perception gives {name} {p:.0%}")
        seg = evidence.segmenter.get(name, 0.0) if evidence else 0.0
        if seg:
            score += seg
            why.append(f"the segmenter named it {name} (score {seg:.2f})")
    extent = obj.get("extentM") or obj.get("sizeM")
    reference = size or (prototype(name) if name else None)
    if reference and extent:
        gap = hidden_gap(obj, support)
        if gap > cfg.hidden_gap_m:
            extent = [*extent[:2], extent[2] + gap]
        fit = size_fit(extent, reference, cfg.size_scale, cfg.size_under_weight)
        score += fit
        why.append(f"size fit {fit:.2f} against {'x'.join(f'{v:.2f}' for v in _dims(reference))} m")
    if doubts(obj, evidence, {}, cfg, support):
        score += 0.2
        why.append("its current name is doubtful")
    return score, why


def candidates(objects: list[dict], evidence: dict[str, Evidence], prompt_words: dict[str, list[str]],
               cfg: ReflectionConfig, name: str | None = None, size=None, exclude=(), limit: int | None = None) -> list[dict]:
    """Objects that could be what a name and size refer to, best first.

    Without a name or size, the objects whose current name is doubtful, most doubts first.

    :param objects: Ego API summary objects.
    :param evidence: perception evidence per object id.
    :param prompt_words: segmenter prompt words per category.
    :param cfg: thresholds and the default limit.
    :param name: sought category, role or vocabulary word.
    :param size: approximate dimensions in metres.
    :param exclude: object ids to leave out.
    :param limit: maximum number of candidates.
    :returns: described objects; with a name or size each carries ``score`` and ``why``, and
        objects without any evidence for the name or size are left out."""
    limit = cfg.candidate_limit if limit is None else limit
    by_id = {o["id"]: o for o in objects}
    pool = [o for o in objects if o["id"] not in set(exclude)]
    described = []
    for obj in pool:
        support = by_id.get(obj.get("on"))
        entry = describe(obj, evidence.get(obj["id"]), prompt_words, cfg, support)
        if name or size:
            entry["score"], entry["why"] = match(obj, evidence.get(obj["id"]), name, size, cfg, support)
            entry["score"] = round(entry["score"], 3)
            if not entry["why"]:
                continue
        elif not entry["doubts"]:
            continue
        described.append(entry)
    key = (lambda e: -e["score"]) if (name or size) else (lambda e: -len(e["doubts"]))
    return sorted(described, key=key)[:limit]


TARGET_KEYS = ("target", "support", "on", "entity", "near", "reference", "container")


def mentioned(arguments: dict | None, message: str) -> list[str]:
    """Ego-State ids a failed call was about: its id arguments, then ids its failure message names.

    :param arguments: the failed call's arguments, possibly nested one level (``params``).
    :param message: the failure text.
    :returns: distinct ids in order of mention."""
    args = dict(arguments or {})
    if isinstance(args.get("params"), dict):
        args = {**args["params"], **args}
    ids = [args[k] for k in TARGET_KEYS if isinstance(args.get(k), str)]
    ids += [e for e in args.get("entities", []) if isinstance(e, str)] if isinstance(args.get("entities"), list) else []
    ids += re.findall(r"\be:\d+\b", message)
    return list(dict.fromkeys(i for i in ids if re.fullmatch(r"e:\d+", i)))


def reflect(objects: list[dict], evidence: dict[str, Evidence], prompt_words: dict[str, list[str]],
            cfg: ReflectionConfig, failure: dict) -> dict:
    """Evidence for reflecting on one failed step.

    :param objects: Ego API summary objects.
    :param evidence: perception evidence per object id.
    :param prompt_words: segmenter prompt words per category.
    :param cfg: thresholds.
    :param failure: ``{tool?, arguments?, message, name?, size?}``; ``name`` and ``size`` describe
        what the instruction asked for when the agent knows them.
    :returns: the failure cause with its usual remedy, the objects the call was about (``missing``
        lists ids the Ego-State no longer holds), and, for a cause in ``candidate_causes``, candidates for the sought object: by
        ``name``/``size``, else by the category the call asked for or of its first target, plus the
        doubtfully named objects.
    :raises ValueError: no failure message."""
    message = failure.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("reflect needs the failure message")
    cause = classify_failure(message)
    by_id = {o["id"]: o for o in objects}
    ids = mentioned(failure.get("arguments"), message)
    targets = [describe(by_id[i], evidence.get(i), prompt_words, cfg, by_id.get(by_id[i].get("on"))) for i in ids if i in by_id]
    args = failure.get("arguments") or {}
    name = failure.get("name") or (args.get("category") if isinstance(args.get("category"), str) else None)
    if name is None and targets:
        name = targets[0]["category"]
    size = failure.get("size")
    exclude = [t["id"] for t in targets]
    found = []
    if cause.key in cfg.candidate_causes:
        found = candidates(objects, evidence, prompt_words, cfg, name, size, exclude) if (name or size) else []
        seen = {c["id"] for c in found} | set(exclude)
        found += candidates(objects, evidence, prompt_words, cfg, exclude=list(seen))[:max(0, cfg.candidate_limit - len(found))]
    return {
        "cause": {"key": cause.key, "label": cause.label, "advice": cause.advice},
        "sought": {"name": name, "size": size},
        "targets": targets,
        "missing": [i for i in ids if i not in by_id],
        "candidates": found,
    }


def _brief(entry: dict) -> str:
    doubt = f"（存疑：{entry['doubts'][0]}）" if entry["doubts"] else ""
    score = f" 吻合 {entry['score']:.2f}" if "score" in entry else ""
    return f"{entry['id']} {entry['category']}{score}{doubt}"


def activity_text(failure: dict, result: dict, limit: int = 3) -> str:
    """The reflection as one entry of the robot's activity record, for the operator console.

    :param failure: the reflected failure (``tool``, ``message``).
    :param result: what :func:`reflect` returned for it.
    :param limit: objects listed per group.
    :returns: the failure, its cause, the objects involved and the best candidates."""
    parts = [f"{failure.get('tool') or '?'}：{failure['message'][:240]}"]
    if result["cause"]["key"] != "unknown":
        parts.append(f"归类：{result['cause']['label']}")
    if result["targets"]:
        parts.append("涉及：" + "；".join(_brief(t) for t in result["targets"][:limit]))
    if result["missing"]:
        parts.append("已不在 Ego State：" + ", ".join(result["missing"]))
    if result["candidates"]:
        sought = result["sought"]["name"]
        parts.append(f"候选{f'（{sought}）' if sought else ''}：" + "；".join(_brief(c) for c in result["candidates"][:limit]))
    return "\n".join(parts)
