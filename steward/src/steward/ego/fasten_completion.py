"""Completion is independent seating evidence, never an executed release."""
from steward.ego import capabilities
import time
from steward.ego import action_budget as ab
from copy import deepcopy

REPORT = capabilities.get('STEWARD_ASSEMBLY_COMPLETION_REPORT') != '0'
RETRY = capabilities.get('STEWARD_ASSEMBLY_COMPLETION_RETRY') != '0'
CONFIRM = capabilities.get('STEWARD_ASSEMBLY_COMPLETION_CONFIRM') != '0'
OBSERVATIONS = 2
OBSERVATION_BUDGET_S = 12.
REPLAYS_PER_HOLE = 1


def progress(rows, part, body):
    for row in rows:
        row.update(confirmedSeated=row['status']=='confirmed', unknown=row['status']=='unknown')
    remaining=[r['hole'] for r in rows if not r['confirmedSeated']]
    line=(f'接口 {part} → {body}：已释放 {sum(r["released"] for r in rows)}/{len(rows)}；'
          f'确认 {len(rows)-len(remaining)}/{len(rows)} 入座。')
    pending=[r['hole'] for r in rows if not r['released']]
    line += ('未完成孔 '+', '.join(remaining)+'；其余未能目视确认，禁止盲目补料。' if remaining else '各孔均已放置一颗，均已确认入座。')
    if pending: line += '仅续做未释放孔：'+', '.join(pending)+'。'
    return dict(interfaceSummary=line,guidance=line,remainingHoles=remaining)


def observe_unknown(controller, job, attempt, check):
    """A missing identity permits observation retry, never a blind refill."""
    if attempt.get('holeSeatObservation') or not RETRY or check.get('passed') is not None:
        return check
    from steward.ego.fasten_sequence import budget, DriveExpired, remaining
    from steward.ego.wrist_fasten import released
    from steward.ego.skills import SkillFailed
    seconds=(OBSERVATIONS*(ab.frame_window(controller.ex)+ab.observation(controller.ex))
        if ab.enabled() else OBSERVATION_BUDGET_S)
    used=attempt.get('confirmationRetries',0)
    spent=attempt.get('confirmationRetrySimS' if ab.enabled() else 'confirmationRetryWallS',0.)
    if (used>=OBSERVATIONS or spent>=seconds) and attempt.get('completionRetry'):
        job.postcondition['completionRetry']=deepcopy(attempt['completionRetry'])
        return check
    start=time.monotonic()
    row=dict(maxObservations=OBSERVATIONS,maxReplays=REPLAYS_PER_HOLE,
             budgetS=seconds,observations=used,replays=attempt.get('seatRetryCount',0))
    attempt['completionRetry']=row
    deadline=ab.make(controller.ex,max(0.,seconds-spent),'wrist frame plus head capture/process periods per confirmation observation') if ab.enabled() else start+min(max(0.,OBSERVATION_BUDGET_S-spent),remaining(OBSERVATION_BUDGET_S))
    try:
        with budget(deadline):
            while used<OBSERVATIONS and check.get('passed') is None:
                controller.check();used+=1
                attempt['confirmationRetries']=used;row['observations']=used
                released(controller,job,attempt,None)
                check=controller.seat_posterior(attempt)
                if check.get('passed') is None:
                    controller.close_seat_view(job,[attempt])
                    check=controller.seat_posterior(attempt)
    except DriveExpired:
        row['stopReason']='confirmation_sim_limit_or_deadlock' if ab.enabled() else 'confirmation_wall_limit'
    except SkillFailed as exc:
        controller.check();row['stopReason']='observation_failed: '+str(exc)
    attempt['confirmationRetryWallS']=attempt.get('confirmationRetryWallS',0.)+time.monotonic()-start
    if ab.enabled():
        attempt['confirmationRetrySimS']=spent+deadline.sim_s
        row.update(ab.report(deadline))
    row.update(wallS=round(attempt['confirmationRetryWallS'],3),status=check['status'])
    row.setdefault('stopReason','confirmed' if check.get('passed') is True else
                   'observed_miss' if check.get('passed') is False else 'identity_or_seating_unresolved_no_refill')
    job.postcondition['completionRetry']=deepcopy(row)
    return check
