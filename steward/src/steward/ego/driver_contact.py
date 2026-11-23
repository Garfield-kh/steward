"""One observed absolute contact endpoint, with encoder stall classification."""
# Existing contact allowance, not an achievable incremental motor resolution.
STEP_M = .002


def lower(ctl, plan, motion):
    from steward.ego import tip_entry_servo as servo
    from steward.ego.fasten_driver import motion_sample
    from steward.ego.skills import SkillFailed
    from steward.ego.key_observation import finish
    ex = ctl.ex
    start = ex._state()['stamp']
    row = motion['lowerContact'] = dict(status='observing', source='lift_encoder_following',
        forceObserved=False, policy='observed_absolute_endpoint', steps=0)
    motion['stage'] = 'lowering'
    try:
        ctl.check()
        opening = ctl.rt.esim.arm.geometry.jaw_opening
        q = float(ex._state()['joints']['lift'])
        # Nominal geometry gets only to the clearance pose. The terminal motion
        # always needs a current relative measurement at that pose.
        if q-plan['lift'] > opening/2:
            ex._set_joint('lift', plan['lift']+opening/4)
        servo.driver_hover(ctl, plan, motion)
        current = plan.get('_entryCurrent')
        if current is None:
            raise SkillFailed('near_contact_unresolved')
        servo.current_view(ctl, current, motion, phase='descent')
        evidence = current.get('tipEntryServo', {})
        frame = (evidence.get('frames') or [{}])[-1]
        checkpoint = (evidence.get('keyObservations') or [None])[-1]
        if not frame.get('bothVisible') or frame.get('stamp') != evidence.get('lastStamp'):
            finish(checkpoint, action='hold_for_observation', reason='near_contact_unresolved')
            raise SkillFailed('near_contact_unresolved')
        previous_endpoint=plan['lift']
        if not servo.driver_adopt(ctl, plan, motion):
            finish(checkpoint,action='hold_for_observation',reason=evidence.get('endpointUpdateReason'))
            raise SkillFailed('observed_endpoint_unreachable')
        finish(checkpoint, measured=True, action='update_endpoint' if plan['lift']!=previous_endpoint else 'continue')
        state = ex._state(); q0 = float(state['joints']['lift']); end = float(plan['lift'])
        if not ctl.rt.esim.arm.reachable_path([dict(state['joints']), dict(state['joints'], lift=end)]):
            raise SkillFailed('observed_endpoint_unreachable')
        speed = min(ex.config.gentle_lift_speed, ex.config.arm_speed or ex.config.gentle_lift_speed)
        travel = abs(q0-end); tick = ex.config.tick_s
        limit = 3*travel/speed+2*tick
        row.update(status='descending', steps=1, requestedM=travel, endpoint=end, budgetSimS=limit)
        ex._setpoints['lift'] = end
        ex._command({'lift': end})
        began = state['stamp']; previous = q0
        while True:
            ctl.check(); ex._wait(tick)
            state = ex._state(); q = float(state['joints']['lift'])
            row['actualDescentM'] = q0-q
            error = q-end
            if abs(error) <= STEP_M/8 or q < end:
                # Endpoint arrival is not contact. A new frame decides whether
                # to hold or withdraw, including overshoot below the old estimate.
                servo.driver_after_contact(ctl, plan, motion)
                row['status'] = 'visual_contact'
                return
            if state['stamp']-began >= limit:
                sample = motion_sample(ctl, plan)
                delta = sample.get('heightAboveHeadM')
                row.update(followingErrorM=error, contactSample=sample, heightErrorM=delta)
                if error > STEP_M/8 and abs(q-previous) <= STEP_M/8:
                    servo.driver_after_contact(ctl, plan, motion)
                    row['status'] = 'contact'
                    return
                raise SkillFailed('driver descent tracking_timeout')
            previous = q
    except SkillFailed as exc:
        row['status'] = str(exc)
        raise
    finally:
        row['simS'] = max(0., ex._state()['stamp']-start)
        evidence = motion.get('tipEntryServo')
        if evidence:
            evidence['contactStatus'] = row['status']
            servo.emit(evidence, 'contact_outcome')


def summary(row):
    from steward.ego.driver_batch import compact
    out = {k:v for k,v in row.items() if k not in ('contactSample','endpointSample')}
    sample = row.get('contactSample') or row.get('endpointSample')
    if sample: out['sampledAt'] = sample.get('stamp')
    return compact(out)
