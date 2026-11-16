"""Ego API for the Cobot Magic twin used by ``steward cobot``.

The mission thread owns the estimator and (usually) the skills. This server publishes that same
belief to the console's Ego State panel, and also the plan / activity / control surface the
operator chat and plan card already read from a full Ego Runtime -- so ``steward cobot`` fills the
same console panels as ``steward up``.

When ``skills`` is passed, ``/skill/*`` and ``/ego/control`` are live: the Ego Harness can take a
turn. Without them the routes stay absent rather than present and failing.
"""

from __future__ import annotations

import collections
import itertools
import math
import threading
import uuid
from typing import TYPE_CHECKING, Callable

from steward.ego.estimator import EgoEstimator
from steward.ego.runtime import PLAN_STEP_STATES
from steward.ego.view import EgoViewRenderer
from steward.net import NO_CONTENT, HttpError, JsonServer
from steward.viewloop import ViewLoop, ViewSettings, frame_response

if TYPE_CHECKING:
    from steward.ego.cobot_skills import CobotSkills, Job

ACTIVITY_MAX = 200


class CobotEgoView:
    """Publishes one estimator's belief, plan and (optionally) skills over HTTP for the console.

    ``live`` supplies the robot's current sensor frame. The mission looks at the table a few times
    per round, so without it the panel would show the arm frozen where the last look left it --
    while the arm is in fact mid-motion. Proprioception is continuous; only exteroception is
    look-by-look.
    """

    def __init__(self, estimator: EgoEstimator, live: Callable[[], dict] | None = None,
                 skills: CobotSkills | None = None) -> None:
        self.estimator = estimator
        self.live = live
        self.skills = skills
        self._lock = threading.Lock()
        self.plan: dict | None = None
        self.activity: collections.deque[dict] = collections.deque(maxlen=ACTIVITY_MAX)
        self._activity_ids = itertools.count(1)
        self.boot = uuid.uuid4().hex
        self.control_owner: str | None = None
        self._alerts: list[dict] = []
        self._alert_ids = itertools.count(1)
        renderer = EgoViewRenderer()
        self.view = ViewLoop("ego", lambda settings: self._render(renderer, settings), renderer.close).start()
        if skills is not None and skills._on_finish is None:
            skills._on_finish = self._skill_finished

    def _scene(self):
        frame = None if self.live is None else self.live()
        with self._lock:
            return self.estimator.to_scene(frame)

    def _render(self, renderer: EgoViewRenderer, settings: ViewSettings):
        scene = self._scene().to_dict()
        return renderer.render(scene, settings), "image/jpeg", {"stamp": self.estimator.stamp}

    def _skill_finished(self, job: Job) -> None:
        result = {"succeeded": "完成", "failed": "失败", "stopped": "已停止"}.get(job.state, job.state)
        text = f"{job.skill} {result}" + (f"：{job.message}" if job.state != "succeeded" and job.message else "")
        self.log_activity("skill", text)

    # ------------------------------------------------------------------ plan / activity
    def log_activity(self, kind: str, text: str) -> None:
        """Append one console activity row (``step``, ``skill``, ``state``, ``difficulty``, …)."""
        with self._lock:
            self.activity.append({"id": next(self._activity_ids), "t": round(self.estimator.stamp, 1),
                                  "kind": kind, "text": text[:500]})

    def set_plan(self, plan: dict) -> dict:
        """Publish what the twin is trying to do; same contract as :meth:`EgoRuntime.set_plan`."""
        goal = plan.get("goal")
        steps = plan.get("steps", [])
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("a plan needs a goal")
        if not isinstance(steps, list) or not all(isinstance(st, dict) and isinstance(st.get("name"), str)
                                                   and st.get("status") in PLAN_STEP_STATES for st in steps):
            raise ValueError(f"plan steps need a name and a status in {sorted(PLAN_STEP_STATES)}")
        record = {"goal": goal[:300], "source": str(plan.get("source", "mission"))[:60],
                  "steps": [{"name": st["name"][:200], "status": st["status"]} for st in steps[:30]],
                  "round": plan.get("round"), "current": str(plan.get("current") or "")[:400],
                  "difficulty": str(plan.get("difficulty") or "")[:600],
                  "state": str(plan.get("state") or "running")[:40],
                  "t": round(self.estimator.stamp, 2)}
        with self._lock:
            previous, self.plan = self.plan or {}, record
        if record["current"] and record["current"] != previous.get("current"):
            self.log_activity("step", record["current"])
        if record["difficulty"] and record["difficulty"] != previous.get("difficulty"):
            self.log_activity("difficulty", record["difficulty"])
        if record["state"] != previous.get("state", "running"):
            self.log_activity("state", record["state"])
        return record

    def plan_view(self, since: int = 0) -> dict:
        """Latest plan, the skill running now, and activity newer than ``since``."""
        running = None
        if self.skills is not None and self.skills.current is not None and self.skills.current.state == "running":
            running = self.skills.current.to_dict()
        with self._lock:
            plan = dict(self.plan) if self.plan else None
            activity = [dict(e) for e in self.activity if e["id"] > since]
            newest = self.activity[-1]["id"] if self.activity else 0
        return {"plan": plan, "skill": running, "activity": activity, "activityLast": newest,
                "boot": self.boot, "t": round(self.estimator.stamp, 2)}

    # ------------------------------------------------------------------ control / alerts
    def control(self) -> dict:
        return {"owner": self.control_owner}

    def acquire_control(self, owner: str) -> dict:
        if not owner:
            raise ValueError("control needs an owner name")
        with self._lock:
            if self.control_owner not in (None, owner):
                raise HttpError(409, f"control is held by {self.control_owner}")
            self.control_owner = owner
        if self.skills is not None:
            self.skills.stop()
        return self.control()

    def release_control(self, owner: str) -> dict:
        with self._lock:
            if self.control_owner not in (None, owner):
                raise HttpError(409, f"control is held by {self.control_owner}")
            self.control_owner = None
        return self.control()

    def alerts_view(self) -> dict:
        with self._lock:
            active = [dict(a) for a in self._alerts if not a.get("acked")]
            return {"alerts": [dict(a) for a in self._alerts[-20:]], "active": active}

    def post_alert(self, body: dict) -> dict:
        alert = {"id": next(self._alert_ids), "message": str(body.get("message") or "")[:500],
                 "source": str(body.get("source") or "api")[:40], "level": str(body.get("level") or "error")[:20],
                 "acked": False}
        with self._lock:
            self._alerts.append(alert)
        self.log_activity("alert", alert["message"])
        return alert

    def acknowledge(self, alert_id: int | None = None) -> dict:
        with self._lock:
            for alert in self._alerts:
                if alert_id is None or alert["id"] == alert_id:
                    alert["acked"] = True
        return self.alerts_view()

    # ------------------------------------------------------------------ skills
    def _require_owner(self, owner: str | None) -> None:
        with self._lock:
            held = self.control_owner
        if held is not None and owner != held:
            raise HttpError(409, f"control is held by {held}; pass owner={held!r}")

    def skill_status(self, job_id: str) -> dict:
        if self.skills is None:
            raise HttpError(501, "this cobot Ego API has no skill endpoints")
        job = self.skills.jobs.get(job_id)
        if job is None:
            raise HttpError(404, f"unknown job {job_id}")
        return job.to_dict()

    def skill_wait(self, job_id: str, timeout_s: float) -> dict:
        if self.skills is None:
            raise HttpError(501, "this cobot Ego API has no skill endpoints")
        return self.skills.wait(job_id, timeout_s).to_dict()

    def skill_stop(self) -> dict:
        if self.skills is None:
            raise HttpError(501, "this cobot Ego API has no skill endpoints")
        return self.skills.stop()

    # ------------------------------------------------------------------ routes
    def routes(self) -> dict:
        def view_frame(q, b):
            frame = self.view.latest(int(q["after"]) if q.get("after") else None)
            if frame is not None:
                return frame_response(frame, self.view.status())
            status = self.view.status()
            if status["frame"] is None and status["settings"] is None:
                raise HttpError(409, "the ego view is not configured; POST /ego/view first")
            if status["error"] and status["frame"] is None:
                raise HttpError(503, f"the ego view cannot render: {status['error']}")
            return NO_CONTENT

        def control(b):
            if b.get("action") == "acquire":
                return self.acquire_control(b.get("owner", ""))
            if b.get("action") == "release":
                return self.release_control(b.get("owner", ""))
            raise ValueError("action must be acquire or release")

        routes = {
            ("GET", "/ego/view"): lambda q, b: self.view.status(),
            ("POST", "/ego/view"): lambda q, b: self.view.configure(ViewSettings.parse(b)),
            ("GET", "/ego/view/frame"): view_frame,
            ("GET", "/ego/summary"): lambda q, b: self.summary(),
            ("GET", "/ego/state"): lambda q, b: self._scene().to_dict(),
            ("GET", "/ego/alerts"): lambda q, b: self.alerts_view(),
            ("POST", "/ego/alert"): lambda q, b: self.post_alert(b),
            ("POST", "/ego/alerts/ack"): lambda q, b: self.acknowledge(b.get("id")),
            ("GET", "/ego/plan"): lambda q, b: self.plan_view(int(q.get("since", 0))),
            ("POST", "/ego/plan"): lambda q, b: self.set_plan(b),
            ("GET", "/ego/control"): lambda q, b: self.control(),
            ("POST", "/ego/control"): lambda q, b: control(b),
        }
        if self.skills is not None:
            routes |= {
                ("POST", "/skill/run"): lambda q, b: self._run_receipt(b),
                ("GET", "/skill/status"): lambda q, b: self.skill_status(q["job"]),
                ("POST", "/skill/wait"): lambda q, b: self.skill_wait(b["job"], float(b.get("timeoutS", 120))),
                ("POST", "/skill/stop"): lambda q, b: self.skill_stop(),
            }
        return routes

    def _run_receipt(self, body: dict) -> dict:
        """Start a skill from an Ego Harness ``/skill/run`` body."""
        if self.skills is None:
            raise HttpError(501, "this cobot Ego API has no skill endpoints")
        self._require_owner(body.get("owner"))
        receipt = body.get("receipt") or body
        skill = receipt.get("skill") or body.get("skill")
        if not skill:
            raise ValueError("skill receipt needs a skill name")
        payload = {"id": str(receipt.get("id") or f"api-{skill}"), "plan": dict(receipt.get("plan") or {})}
        return self.skills.start(skill, payload).to_dict()
    def summary(self) -> dict:
        """The object list behind the Ego State view, in the shape the console already reads."""
        with self._lock:
            scene = self.estimator.to_scene(None if self.live is None else self.live())
            stamp = self.estimator.stamp
            held = self.estimator.held_id
        objects = []
        for ent in scene.entities:
            if ent.role == "robot":
                continue
            try:
                size = ent.rigid.links[0].geometry["collision"][0]["size"]
            except (AttributeError, KeyError, IndexError):
                size = []
            cov = (ent.pose.unc or {}).get("cov") if ent.pose.unc else None
            objects.append({
                "id": ent.id,
                "category": ent.category.value,
                "position": [round(float(v), 3) for v in ent.pose.value["p"]],
                "sizeM": [round(float(v), 3) for v in size],
                "uncertaintyM": round(math.sqrt((cov[0] + cov[7]) / 2), 4) if cov else None,
                "existence": float(ent.existence),
                "held": ent.id == held,
            })
        return {"stamp": stamp, "mode": "cobot", "objects": objects}

    def stop(self) -> None:
        self.view.stop()


def cobot_ego_api(estimator: EgoEstimator, port: int = 0, bind_wait_s: float = 0.0,
                  live: Callable[[], dict] | None = None,
                  skills: CobotSkills | None = None) -> JsonServer:
    """Serve one estimator's belief (and optional skills) on ``port`` (0 picks a free one).

    The returned server exposes ``.cobot_view`` so the mission thread can publish plans.
    """
    view = CobotEgoView(estimator, live=live, skills=skills)
    server = JsonServer(view.routes(), port=port, name="ego", bind_wait_s=bind_wait_s)
    server.cobot_view = view  # type: ignore[attr-defined]
    server.start()

    original_stop = server.stop

    def stop() -> None:
        view.stop()
        original_stop()

    server.stop = stop  # type: ignore[method-assign]
    return server
