"""Precision increments in the last successfully sent base target's frame.

This is a command ledger, never a measurement of the robot's actual pose.
No odometry position or heading may initialize the ledger.
"""
from copy import deepcopy
import math
import threading

from steward.ego import capabilities


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_BASE_COMMAND_LEDGER') != '0'


class BaseHold:
    def __init__(self):
        self.lock = threading.RLock()
        self.episode = self.stamp = self.base = self.command_stamp = None
        self.generation = self.sequence = 0
        self.valid = False
        self.reason = 'no_successful_base_command'

    def invalidate(self, reason):
        with self.lock:
            self.generation += 1
            self.valid = False
            self.reason = reason

    def read(self, state):
        with self.lock:
            episode, stamp = state.get('episode'), state.get('stamp')
            if episode != self.episode:
                self.invalidate('episode_changed')
            elif self.stamp is not None and stamp is not None and stamp < self.stamp:
                self.invalidate('sensor_clock_reset')
            self.episode, self.stamp = episode, stamp
            if episode is None:
                self.invalidate('episode_unavailable')
        return state

    def snapshot(self):
        with self.lock:
            return dict(base=deepcopy(self.base), episode=self.episode,
                        stamp=self.stamp, commandStamp=self.command_stamp, generation=self.generation,
                        sequence=self.sequence, valid=self.valid, reason=self.reason,
                        source='successfully_sent_command_not_actual_pose')

    def matches(self, saved):
        now = self.snapshot()
        return now['valid'] and saved['valid'] and all(
            now[k] == saved[k] for k in ('episode', 'generation', 'sequence', 'base'))

    def send(self, send, command):
        # Invalidate before transport: a failed/partial send is not a hold proof.
        if 'base' not in command:
            if command.get('turnOver'):
                self.invalidate('turn_over_command')
            return send(command)
        with self.lock:
            if command.get('turnOver'):
                self.invalidate('turn_over_command')
            self.invalidate('base_command_pending_or_failed')
            token = self.generation
            episode = self.episode
            stamp = self.stamp
            base = deepcopy(command['base'])
        result = send(command)
        with self.lock:
            self.sequence += 1
            if token == self.generation and episode is not None:
                if len(base) == 3 and all(math.isfinite(v) for v in base):
                    self.base = list(base)
                    self.command_stamp = stamp
                    self.valid, self.reason = True, None
        return result


def install(ex):
    """Install at executor construction, before its first movement."""
    record = ex._baseCommandLedger = BaseHold()
    read, send = ex._state, ex._command

    def state():
        return record.read(read())

    def command(value):
        if 'base' in value and record.stamp is None:
            state()  # identify the episode only; never seed a target from this pose
        return record.send(send, value)

    ex._state, ex._command = state, command


def invalidate(ex, reason):
    record = getattr(ex, '_baseCommandLedger', None)
    if record is not None:
        record.invalidate(reason)
        if reason == 'executor_cancel_or_reset':
            # The first command after cancellation must identify its episode anew.
            with record.lock:
                record.episode = record.stamp = None


def snapshot(ex):
    record = getattr(ex, '_baseCommandLedger', None)
    return record.snapshot() if record is not None else dict(
        base=None, episode=None, valid=False, reason='command_ledger_unavailable')


def target(base, lateral):
    x, y, yaw = base
    return [x-math.sin(yaw)*lateral, y+math.cos(yaw)*lateral, yaw]


def local_increment(before, after):
    x, y, yaw = before
    dx, dy = after[0]-x, after[1]-y
    return [math.cos(yaw)*dx+math.sin(yaw)*dy,
            -math.sin(yaw)*dx+math.cos(yaw)*dy, after[2]-yaw]


def begin_receipt(ex, lateral):
    before = snapshot(ex)
    return dict(baseHoldBefore=before, baseTargetBefore=deepcopy(before['base']),
                plannedLocalIncrement=[0., lateral, 0.],
                incrementUnits=['m', 'm', 'rad'], baseCommandSent=False,
                confirmed=False, reason=None)


def finish_receipt(ex, receipt):
    after = snapshot(ex)
    before = receipt['baseHoldBefore']
    receipt.update(baseHoldAfter=after, baseTargetAfter=deepcopy(after['base']))
    receipt['baseCommandSent'] = after.get('sequence', 0) > before.get('sequence', 0)
    comparable = before['valid'] and after['valid'] and before['episode'] == after['episode']
    receipt['actualSentLocalIncrement'] = (
        local_increment(before['base'], after['base']) if comparable else None)
    receipt['confirmed'] = bool(comparable)
    receipt['confirmationMeaning'] = 'command_difference_only_not_actual_motion'


def settled_execution(before, observation):
    """Encoder differences at the new measurement's established hold window."""
    window = (observation or {}).get('window', {})
    samples = window.get('holdSamples', [])
    if not window.get('holdEstablished') or not samples:
        return dict(available=False, reason='stable_remeasurement_window_unavailable')
    after = samples[-1]
    if before.get('episode') != after.get('episode'):
        return dict(available=False, reason='episode_changed')
    delta = {k:float(v-before['joints'][k]) for k,v in after['joints'].items()
             if k in before['joints']}
    return dict(available=True, sampling='remeasurement_hold_window_end',
                fromStamp=before['stamp'], toStamp=after['stamp'],
                jointDelta=delta, reachM=delta.get('reach'),
                wristYawRad=delta.get('wrist_yaw'), windowId=window.get('id'))
