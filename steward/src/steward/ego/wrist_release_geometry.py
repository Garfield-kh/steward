"""Shared FK self masks and grasp dimensions for the ASSEMBLY_WRIST_PIPELINE release camera."""


def robot_boxes(controller, raw):
    est=controller.rt.estimator
    with controller.rt._lock:
        previous=est.joints,est.odometry
        try:
            est.joints,est.odometry=raw['header']['joints'],raw['header']['odometry']
            return list(est.body_boxes())
        finally:
            est.joints,est.odometry=previous


def prior(track):
    if track is None: return None
    for key in ('held_grasp_geometry','held_tool_feature','tool_feature'):
        f=getattr(track,key,None)
        if f and f.get('lengthM',0)>0 and f.get('radiusM',0)>0:
            return {k:f[k] for k in ('lengthM','radiusM','source') if k in f}
    return None
