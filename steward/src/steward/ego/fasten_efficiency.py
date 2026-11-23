"""Keep a multi-hole job moving after optional alignment/view work."""
from steward.ego import capabilities
import time

ENABLED = capabilities.get('STEWARD_ASSEMBLY_FASTEN_EFFICIENCY') != '0'


def align_next(controller, job, attempt):
    from steward.ego.fasten_align import align_after_first
    from steward.ego.skills import SkillFailed
    if not ENABLED: return align_after_first(controller,job,attempt)
    start=time.monotonic()
    try:
        return align_after_first(controller,job,attempt)
    except SkillFailed as exc:
        controller.check()
        if controller.rt.estimator.held_id:
            raise
        # Every remaining seat still predicts against the current registration.
        # A failed optional view/push must not turn unattempted into completed.
        job.postcondition.setdefault('holeContinuations',[]).append(dict(hole=attempt['hole'],
            reason='optional_alignment_failed_continue_fresh_seat_prediction',detail=str(exc),
            wallS=round(time.monotonic()-start,3)))
        controller.warn('optional_alignment_failed_continue',hole=attempt['hole'],detail=str(exc))


def final_view(controller, job, attempts):
    if not attempts: return
    if ENABLED and all(a.get('holeSeatObservation') for a in attempts):
        job.postcondition['savedSeatView'] = dict(trips=1, reason='direct_hole_views_completed')
        return
    if ENABLED and all(a.get('seatConfirmed') for a in attempts):
        job.postcondition['savedSeatView']=dict(trips=1,reason='all_release_observations_confirmed',
            estimatedWallSavedS=getattr(controller,'_last_seat_view_wall_s',None),
            estimateSource='previous_same_job_view' if hasattr(controller,'_last_seat_view_wall_s') else 'unmeasured')
        return
    start=time.monotonic()
    try:
        controller.close_seat_view(job,attempts)
    finally:
        if ENABLED: controller._last_seat_view_wall_s=time.monotonic()-start
