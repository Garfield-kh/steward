"""Keep edge corrections in a registered body frame; no additional sensing."""
from copy import deepcopy
import math
import numpy as np
from steward.ego import capabilities
from steward.math3d import matrix_from_quat, quat_from_yaw


def enabled():
    return all(capabilities.get(n) != '0' for n in ('STEWARD_ASSEMBLY_EDGE_GEOMETRY', 'STEWARD_ASSEMBLY_EDGE_RETENTION', 'STEWARD_ASSEMBLY_PLACEMENT_SUPPORT_MARGIN', 'STEWARD_ASSEMBLY_DIRECTED_ORIENTATION', 'STEWARD_ASSEMBLY_PLACEMENT_BODY_FRAME'))


def long_axis(record):
    ref = record.get('contactReference', record['reference'])
    axis = matrix_from_quat(quat_from_yaw(ref['yaw']))[:, int(np.argmax(ref['size'][:2]))]
    if record.get('edgeGeometry', {}).get('turnOver'):
        r=matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
        axis=r @ np.diag([1.,-1.,-1.]) @ r.T @ axis
    return axis


def outward_axis(record):
    axis = long_axis(record)
    return axis if axis @ np.asarray(record['edgeGeometry']['outwardAxis']) >= 0 else -axis


def correction_rows(record, table, sigma):
    """Exact intention, tangent neighbours, other edges, then checked recovery."""
    from steward.ego.edge_retention import contact, candidates
    from steward.ego.assembly_edge import entity_box
    target = record['edgeCorrectionTarget']; box, _, _ = entity_box(table)
    if table.id != target['support']: return []
    tr = matrix_from_quat(quat_from_yaw(box.yaw))[:2,:2]
    offset = (np.asarray(target['spot'])-[box.cx,box.cy]) @ tr
    angle = target.get('yaw', record['reference']['yaw']+target.get('yawDeltaRad',0.))-record['reference']['yaw']
    dim, sign = target['edgeAxis'], target['edgeSign']; tangent = 1-dim
    env = np.asarray(record['edgeGeometry']['envelope']) @ matrix_from_quat(quat_from_yaw(angle)).T
    env = env[:,:2] @ tr; half = [box.hx,box.hy]
    lo, hi = -half[tangent]-env[:,tangent].min(), half[tangent]-env[:,tangent].max()
    rows=[]
    if lo <= hi:
        # Divide the free tangent interval using the registered footprint.
        step=max(float(np.ptp(env[:,tangent])), float(sigma), np.finfo(float).eps)
        alongs=[offset[tangent], np.clip(offset[tangent]-step,lo,hi), np.clip(offset[tangent]+step,lo,hi),lo,hi]
        for along in dict.fromkeys(float(x) for x in alongs):
            moved=offset.copy(); moved[tangent]=along
            c=contact(record,table,dim,sign,angle,moved,inward=True)
            if c:
                c.update(inwardOnly=True, correctionTarget=True, correctionRetry=bool(rows))
                rows.append(c)
    other=deepcopy(record);other.pop('edgeCorrectionTarget',None)
    for c in candidates(other,table,sigma):
        if (c['edgeAxis'],c['edgeSign']) != (dim,sign):
            c.update(correctionTarget=True,correctionRetry=True);rows.append(c)
    # Last resort only after all corrective paths were checked.
    original=(np.asarray(record['reference']['position'][:2])-[box.cx,box.cy]) @ tr
    c=contact(record,table,dim,sign,0.,original,inward=True)
    if c:
        c.update(correctionRecovery=True,inwardOnly=True);rows.append(c)
    return rows


def own_records(scene, record):
    """Identity links only: proximity must never erase an unrelated obstacle."""
    ids={record['reference']['entity'], *record.get('aliases', [])}
    ids.update(p['entity'] for p in record.get('parts', []))
    for ent in scene.entities:
        value=ent.state_vars.get('orientation')
        info=(value.value or {}) if value else {}
        for key in ('stationaryAssembly', 'assembly'):
            saved=info.get(key) or {}
            if saved.get('reference',{}).get('entity') == record['reference']['entity']:
                ids.add(ent.id)
    return ids


def neighbour_options(record, table, sigma):
    """Only used after the normal initial edge candidates; no extra observation."""
    from steward.ego.edge_retention import candidates
    if record.get('edgeCorrectionTarget'): return []
    key=lambda c:tuple([c['edgeAxis'],c['edgeSign'],*c['offset'],c['yaw']])
    original={key(c) for c in candidates(record,table,sigma)}
    expanded=deepcopy(record);expanded['edgeNeighbourSearch']=True
    return [c for c in candidates(expanded,table,sigma) if key(c) not in original]
