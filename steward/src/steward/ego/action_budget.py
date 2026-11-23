"""Action time from controller stamps; wall time only protects stalled IO.

No sensor polling in remaining(): it is also called from the sensor transport.
The reader feeds the clock even when passive phase timing is disabled.
"""
from steward.ego import capabilities
from dataclasses import dataclass, field
import math
import time
from statistics import median


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_ACTION_BUDGET') != '0'


def ratio_enabled():
    return capabilities.get('STEWARD_ASSEMBLY_CLOCK_RATIO') != '0'


@dataclass
class Clock:
    stamp: float = 0.
    wall: float = field(default_factory=lambda: time.monotonic())
    ratio: float = 1.
    initialized: bool = False
    sample_wall: float = 0.
    sample_sim: float = 0.
    windows: list = field(default_factory=list)

    def observe(self, stamp):
        now = time.monotonic()
        if not math.isfinite(stamp):
            return
        if not self.initialized:
            self.stamp, self.wall, self.initialized = stamp, now, True
        elif stamp > self.stamp:
            # High-water observed ratio tolerates a later increase in load. Do
            # not learn from repeated stamps (a deadlock must not renew itself).
            if not ratio_enabled():
                self.ratio = max(self.ratio, (now-self.wall)/(stamp-self.stamp))
            self.stamp, self.wall = stamp, now

    def measure(self, sim_s, wall_s):
        """Only active control waits, in seconds/seconds; no planning/idle gap.

        Aggregate one sensor second before division; median of five windows
        rejects isolated IO pauses and adapts after three sustained windows.
        Repeated/frozen stamps do not renew the watchdog or teach a ratio.
        """
        if not ratio_enabled() or not all(math.isfinite(v) and v>0 for v in (sim_s, wall_s)):
            return
        self.sample_sim += sim_s
        self.sample_wall += wall_s
        if self.sample_sim >= 1.:
            self.windows = (self.windows+[self.sample_wall/self.sample_sim])[-5:]
            self.ratio = max(1., float(median(self.windows)))
            self.sample_sim = self.sample_wall = 0.

    def ratio_report(self):
        return dict(method='active_control_wait_window_median', unit='wall_s/sim_s',
            windowSimS=1., retainedWindows=len(self.windows), maxWindows=5)


def clock(ex):
    value = getattr(ex, '_action_clock', None)
    if value is None:
        value = ex._action_clock = Clock()
    return value


def reader(ex, fn):
    def read(*args, **kwargs):
        state = fn(*args, **kwargs)
        clock(ex).observe(float(state['stamp']))
        return state
    return read


class Comparable:
    """Order by the effective wall guard without sampling two different nows."""
    def __lt__(self, other):
        return endpoint(self) < endpoint(other)

    def __le__(self, other):
        return endpoint(self) <= endpoint(other)


@dataclass
class Deadline(Comparable):
    clock: Clock
    seconds: float
    basis: object = None
    start: float = field(init=False)
    wall_start: float = field(default_factory=lambda: time.monotonic())

    def __post_init__(self):
        self.start = self.clock.stamp

    @property
    def sim_s(self):
        return max(0., self.clock.stamp-self.start)

    def wall_limit(self):
        # Reuse the executor's existing stalled-sensor allowance as bootstrap;
        # four full measured runtimes provide headroom for scheduling/IO.
        return max(900., self.seconds*self.clock.ratio*4)

    def left(self):
        now = time.monotonic()
        if self.sim_s >= self.seconds:
            return 0.
        return max(0., min(self.wall_start+self.wall_limit()-now,
                           self.clock.wall+max(900., self.wall_limit())-now))

    def report(self):
        return dict(clock='simulation', budgetSimS=round(self.seconds,3), simS=round(self.sim_s,3),
            wallS=round(time.monotonic()-self.wall_start,3), wallSimRatio=round(self.clock.ratio,3),
            wallGuardS=round(self.wall_limit(),3), basis=self.basis,
            **({'ratioEstimator':self.clock.ratio_report()} if ratio_enabled() else {}),
            stopReason=('sim_budget' if self.sim_s >= self.seconds else
                        'wall_deadlock_guard' if self.left() <= 0 else None))


@dataclass
class Limits(Comparable):
    deadlines: tuple

    def left(self):
        return min(left(d) for d in self.deadlines)


def make(ex, seconds, basis=None):
    if not enabled():
        return time.monotonic()+seconds
    c = clock(ex)
    if not c.initialized:
        c.observe(float(ex._state()['stamp']))
    return Deadline(c, seconds, basis)


def left(deadline):
    return deadline.left() if isinstance(deadline, (Deadline, Limits, Watchdog)) else deadline-time.monotonic()


def expired(deadline):
    return left(deadline) <= 0


def limited(*deadlines):
    return Limits(deadlines) if any(isinstance(d, (Deadline, Limits, Watchdog)) for d in deadlines) else min(deadlines)


def reserve(deadline, seconds):
    if not isinstance(deadline, Deadline):
        return deadline-seconds
    d = Deadline(deadline.clock, max(0., deadline.seconds-seconds), deadline.basis)
    d.start, d.wall_start = deadline.start, deadline.wall_start
    return d


def report(deadline, *, compact=False):
    row = deadline.report() if isinstance(deadline, Deadline) else {}
    return {k:v for k,v in row.items() if k in ('clock','budgetSimS','simS','stopReason')} if compact else row


def frame_window(ex, frames=1):
    from steward.ego.runtime import RuntimeConfig
    return frames*getattr(ex, '_frame_period_s', RuntimeConfig.keyframe_max_s)


def observation(ex, frames=1):
    # Capture and process a new keyframe: at most two configured frame periods.
    return frame_window(ex,2*frames)



def frame_timeout(ex, legacy, frames=1):
    return frame_window(ex,frames) if enabled() else legacy


def local_motion_estimate(controller, frames=1):
    """Full local arm traversal plus servo settling and requested fresh views.

    Recovery can touch lift, reach and all three wrist axes. Reuse their actual
    speed/range limits and _set_joint's eight sensor seconds of settling each.
    No travel allowance: recovery stays inside its already admitted local path.
    """
    g=controller.rt.esim.arm.geometry;cfg=controller.ex.config
    speed=min(cfg.gentle_lift_speed,cfg.arm_speed or cfg.gentle_lift_speed)
    slide=((g.lift_max-g.lift_min)+(g.reach_max-g.reach_min))/speed
    wrist=2*sum(getattr(g,k,0.) for k in ('wrist_yaw_max','wrist_roll_max','wrist_pitch_max'))/cfg.wrist_pitch_speed
    return math.ceil((slide+wrist+5*8.+observation(controller.ex,frames))*1000)/1000


def drive_estimate(controller, holes=1):
    """Conservative motion envelope, recomputed from robot and perceived route.

    Supply, each interface hole, and parking; each journey can turn at both ends.
    Pick + stroke + park each have at most two full lift/reach traversals.
    Every wrist can turn through its full joint range in each of those phases.
    Servo corrections get the same eight-second limit as _set_joint.
    """
    ex = controller.ex
    cfg = ex.config
    g = controller.rt.esim.arm.geometry
    scene = controller.rt.ego_state()
    points = [e.pose.value['p'][:2] for e in scene.entities]
    o = ex._state()['odometry']
    points.append([o['x'], o['y']])
    diameter = max((math.dist(a,b) for a in points for b in points), default=0.)
    phases = 2+holes
    slide = (g.lift_max-g.lift_min)+(g.reach_max-g.reach_min)
    speed = min(cfg.gentle_lift_speed, cfg.arm_speed or cfg.gentle_lift_speed)
    wrist = 2*sum(getattr(g, k, 0.) for k in ('wrist_yaw_max','wrist_roll_max','wrist_pitch_max'))
    terms = dict(travelSimS=phases*diameter/cfg.gentle_speed,
        turnSimS=phases*2*math.pi/cfg.gentle_turn_speed,
        slideSimS=phases*2*slide/speed,
        wristSimS=phases*wrist/cfg.wrist_pitch_speed,
        # Five joints, one corrective settle each, in each manipulation phase.
        servoSimS=phases*5*8., observationSimS=(2+holes)*observation(ex),
        dwellSimS=holes*.5)
    # Navigation already allows 3x its nominal ramp duration for convergence.
    terms['travelSimS'] *= 3
    terms['turnSimS'] *= 3
    terms={k:math.ceil(v*1000)/1000 for k,v in terms.items()}
    return sum(terms.values()), dict(method='perceived_route_and_robot_motion_envelope',
        terms=terms, routeDiameterM=round(diameter,3), manipulationPhases=phases)


@dataclass
class Watchdog(Comparable):
    clock: Clock
    seconds: float
    start: float = field(default_factory=lambda: time.monotonic())

    def left(self):
        return self.start+max(self.seconds, self.seconds*self.clock.ratio*4)-time.monotonic()


def watchdog(ex, seconds):
    return Watchdog(clock(ex), seconds) if enabled() else time.monotonic()+seconds


def endpoint(value):
    if isinstance(value, Deadline):
        return value.wall_start+value.wall_limit() if value.sim_s < value.seconds else -math.inf
    if isinstance(value, Watchdog):
        return value.start+max(value.seconds,value.seconds*value.clock.ratio*4)
    if isinstance(value, Limits):
        return min(endpoint(d) for d in value.deadlines)
    return value
