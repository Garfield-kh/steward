"""Conservative navigation advice from fresh Ego observations only."""
from steward.ego import capabilities
from steward.ego.planner import entity_box

ENABLED = capabilities.get('STEWARD_ASSEMBLY_NAVIGATION_ESCAPE') != '0'


def goal_evidence(scene, goal, radius):
    if not ENABLED:
        return None
    nearby = []
    for entity in scene.entities:
        if entity.id == scene.self_id or entity.pose.src not in ('observed', 'fused'):
            continue
        stamp = entity.pose.t
        if stamp is None or not 0 <= scene.stamp-stamp <= 20:
            continue
        if entity_box(entity)[0].distance(*goal) <= radius:
            nearby.append(entity.id)
    # A nearby observed entity is not a free-space certificate. Absence of one
    # always warns: the scene schema has no persisted camera coverage map.
    return dict(status='observed_nearby' if nearby else 'unknown',
        reason='nearby_fresh_observations' if nearby else 'navigation_goal_unobserved',
        nearbyEntities=nearby, freshnessS=20,
        nextAction=None if nearby else dict(skill='observe', targetXY=list(goal),
            message='先观察目标附近及前进方向，再重新预测导航。'))
