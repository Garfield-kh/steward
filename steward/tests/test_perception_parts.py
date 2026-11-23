"""Offline point clouds and tracker regressions; no world, renderer or SAM service."""
import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator, Track, _small_observations_apart, _voxelize
from steward.ego.perception import _raised_cluster_indices, box_corners, box_of


def cloud(x=1.0, y=0.0):
    xx, yy, zz = np.meshgrid(np.linspace(-.008, .008, 6),
                            np.linspace(-.008, .008, 6), np.linspace(0, .022, 4))
    return np.c_[xx.ravel() + x, yy.ravel() + y, zz.ravel() + .75]


def track(i, y):
    points = cloud(y=y)
    pos, size, yaw = box_of(points)
    return Track(id=f'e:{i}', category_counts={'toy_screw': 3.0}, pos=pos,
                 var=np.full(3, .01**2), yaw=yaw, yaw_var=.01, size=size,
                 observations=3, first_seen=0., last_seen=1., voxels=_voxelize(points), span=box_corners(pos, size, yaw))


@pytest.mark.parametrize('method', ['_merge_duplicates', '_merge_same_place',
                                    '_merge_same_spot', '_merge_same_surface'])
def test_nearby_hardware_survives_every_short_distance_merge(method):
    est = EgoEstimator(steward_bot().to_dict())
    est.tracks = {t.id: t for t in [track(i, i * .08) for i in range(4)]}
    fn = getattr(est, method)
    fn(np.array([0., 0., 1.2])) if method in ('_merge_same_place', '_merge_same_surface') else fn()
    assert len(est.tracks) == 4


def test_two_noisy_views_of_one_screw_still_merge():
    est = EgoEstimator(steward_bot().to_dict())
    est.tracks = {t.id: t for t in (track(1, 0), track(2, .006))}
    est._merge_duplicates()
    assert len(est.tracks) == 1


def test_completion_cannot_hide_observed_separation():
    a, b = track(1, 0), track(2, .08)
    a.size[:] = b.size[:] = [.1, .05, .03]
    assert _small_observations_apart(a.span, b.span)
    assert not _small_observations_apart(a.span, a.span + [.004, .004, 0])
    assert not _small_observations_apart(None, b.span)


@pytest.mark.parametrize('count', [2, 3, 4, 5])
def test_one_mask_of_several_screws_splits_at_each_gap(count):
    points = np.vstack([cloud(y=i * .08) for i in range(count)])
    pieces = _raised_cluster_indices(points, 20)
    assert pieces is not None and len(pieces) == count
    assert sum(len(p) for p in pieces) == len(points)
    for p in pieces:
        assert np.ptp(points[p, 1]) < .02


def test_one_connected_part_is_not_split():
    assert _raised_cluster_indices(cloud(), 20) is None


def test_repeated_sensor_regions_keep_four_named_tracks(monkeypatch):
    from steward.ego.perception import Segment

    est = EgoEstimator(steward_bot().to_dict())
    regions = []
    for i in range(4):
        points = cloud(y=i * .08)
        regions.append(Segment(points=points, color=np.array([.55, .58, .62]),
                               pixels=len(points), invalid_frac=0., border=False,
                               range=1.2, bottom_seen=True, semantics={'toy_screw': .7},
                               span=box_corners(*box_of(points))))
    monkeypatch.setattr('steward.ego.estimator.segments', lambda *a, **kw: regions)
    for stamp in range(1, 5):
        frame = {'header': {'stamp': stamp, 'odometry': est.odometry, 'joints': {}},
                 'depth': np.ones((40, 60)), 'fovy': 60.}
        est.update_rgbd(frame)
        assert len(est.tracks) == 4
        assert {t.category for t in est.tracks.values()} == {'toy_screw'}


@pytest.mark.parametrize('labels', [None, {'toy_screw': .7}])
def test_noisy_screw_regions_keep_recall_and_do_not_create_duplicates(monkeypatch, labels):
    from steward.ego.perception import Segment

    est = EgoEstimator(steward_bot().to_dict())
    regions = []
    for i in range(4):
        points = cloud(y=i * .08)
        points[:, 0] = 1.0 + (points[:, 0] - 1.0) * (.035 / .016)
        points[:, 2] = .75 + (points[:, 2] - .75) * (.03 / .022)
        regions.append(Segment(points=points, color=np.array([.55, .58, .62]),
                               pixels=len(points), invalid_frac=0., border=False,
                               range=1.2, bottom_seen=True, semantics=labels,
                               span=box_corners(*box_of(points))))
    monkeypatch.setattr('steward.ego.estimator.segments', lambda *a, **kw: regions)
    for stamp in range(1, 7):
        est.update_rgbd({'header': {'stamp': stamp, 'odometry': est.odometry, 'joints': {}},
                         'depth': np.ones((40, 60)), 'fovy': 60.})
        assert len(est.tracks) == 4
        assert {t.category for t in est.tracks.values()} == {'toy_screw'}
        assert all(max(t.size[:2]) < .06 for t in est.tracks.values())


def sheet_track(i=1, y=0., observations=5, score=.8, width=.07):
    xx, yy = np.meshgrid(np.linspace(-.06, .06, 12), np.linspace(-width / 2, width / 2, 10))
    points = np.c_[xx.ravel() + 1., yy.ravel() + y, np.full(xx.size, .755)]
    pos, size, yaw = box_of(points)
    return Track(id=f'e:{i}', category_counts={'manual': 2.5, 'picture': 2.5},
                 pos=pos, size=size, yaw=yaw, yaw_var=.01, var=np.full(3, .01**2),
                 observations=observations, first_seen=0., last_seen=1.,
                 voxels=_voxelize(points), span=box_corners(pos, size, yaw),
                 mean_rgb=np.array([.9, .9, .9]), bottom_seen=False,
                 semantic_counts={'manual': score * observations, 'picture': score * observations},
                 semantic_views=observations)


def test_ambiguous_named_sheets_export_with_thickness_without_widening_or_duplicates():
    est = EgoEstimator(steward_bot().to_dict())
    a, b = sheet_track(1), sheet_track(2, y=.14)
    est.tracks = {t.id: t for t in (a, b)}
    assert not any(e.id in est.tracks for e in est.to_scene().entities)
    for t in (a, b):
        est._rebuild_geometry(t, np.array([0., 0., 1.2]), 1.2)
        assert t.size[2] == pytest.approx(.004)
        assert max(t.size[:2]) < .14
        assert min(t.size[:2]) < .09
        assert t.top() == pytest.approx(.755)
    for method in ('_merge_duplicates', '_merge_same_place', '_merge_same_spot', '_merge_same_surface'):
        fn = getattr(est, method)
        fn(np.array([0., 0., 1.2])) if method in ('_merge_same_place', '_merge_same_surface') else fn()
        assert len(est.tracks) == 2
    assert {e.id for e in est.to_scene().entities if e.id in est.tracks} == {'e:1', 'e:2'}
    duplicate = sheet_track(3, y=.004)
    est.tracks[duplicate.id] = duplicate
    est._rebuild_geometry(duplicate, np.array([0., 0., 1.2]), 1.2)
    est._merge_duplicates()
    assert len(est.tracks) == 2


@pytest.mark.parametrize('kwargs', [{'observations': 1}, {'score': .3}, {'score': 0.}, {'width': .02}])
def test_sheet_completion_does_not_publish_weak_single_or_line_detections(kwargs):
    est = EgoEstimator(steward_bot().to_dict())
    t = sheet_track(**kwargs)
    est.tracks[t.id] = t
    est._rebuild_geometry(t, np.array([0., 0., 1.2]), 1.2)
    # Unnamed planes can take the old scissors path; its existing cleanup must
    # still reject them, rather than the sheet rule publishing a new object.
    est._drop_impossible(np.array([0., 0., 1.2]))
    if t.id in est.tracks:
        assert t.size[2] == pytest.approx(0.)
    assert not any(e.id == t.id for e in est.to_scene().entities)


def test_confident_single_sheet_keeps_existing_thickness_completion():
    est = EgoEstimator(steward_bot().to_dict())
    t = sheet_track(observations=1)
    t.semantic_counts = {'picture': .8}
    t.category_counts = {'picture': 1.}
    est.tracks[t.id] = t
    est._rebuild_geometry(t, np.array([0., 0., 1.2]), 1.2)
    assert t.category == 'picture'
    assert t.size[2] == pytest.approx(.004)
    assert t.top() == pytest.approx(.755)
    assert max(t.size[:2]) < .14
    assert any(e.id == t.id for e in est.to_scene().entities)


def measured_track(i, category, points, observations=6):
    pos, size, yaw = box_of(points)
    return Track(id=f'e:{i}', category_counts={category: float(observations)},
                 pos=pos, size=size, yaw=yaw, yaw_var=.01, var=np.full(3, .007**2),
                 observations=observations, first_seen=0., last_seen=1.,
                 voxels=_voxelize(points), span=box_corners(pos, size, yaw),
                 semantic_counts={category: .8 * observations}, semantic_views=observations)


@pytest.mark.parametrize('blob_first', [False, True])
def test_old_background_smear_cannot_swallow_repeated_named_fuselage(blob_first):
    est = EgoEstimator(steward_bot().to_dict())
    xx, yy, zz = np.meshgrid(np.linspace(.91, 1.09, 19), np.linspace(-.02, .02, 5),
                            np.linspace(.75, .81, 7))
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    part = measured_track(1, 'airplane_fuselage', points)
    # Same visible part plus a long background strip: more observations cannot
    # make the strip the object. Both iteration orders must preserve the part.
    background = points.copy(); background[:, 0] -= .8
    blob = measured_track(2, 'unknown', np.vstack([points, background]), 100)
    tracks = [blob, part] if blob_first else [part, blob]
    est.tracks = {t.id: t for t in tracks}
    est._merge_same_surface(np.array([0., 0., 1.2]))
    assert list(est.tracks) == [part.id]
    assert part.category == 'airplane_fuselage'
    assert part.size[0] < .2


def test_composite_screw_mask_is_removed_but_individuals_and_solid_wing_survive():
    est = EgoEstimator(steward_bot().to_dict())
    parts = [track(i, i * .08) for i in range(4)]
    composite_points = np.vstack([cloud(y=i * .08) for i in range(4)])
    composite = measured_track(9, 'airplane_wing', composite_points, 30)
    est.tracks = {t.id: t for t in [*parts, composite]}
    assert est._assembly_composite(composite_points)
    est._drop_assembly_composites()
    assert set(est.tracks) == {t.id for t in parts}
    xx, yy = np.meshgrid(np.linspace(.99, 1.01, 6), np.linspace(0, .24, 80))
    wing = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, .765)]
    assert not est._assembly_composite(wing)
    assert not est._assembly_composite(composite_points + [0, 0, .09])
    assert not est._assembly_composite(cloud())
    parts[0].held = True
    est.tracks = {t.id: t for t in parts[:2]}
    assert not est._assembly_composite(np.vstack([cloud(), cloud(y=.08)]))


def test_printed_wheel_fragments_fold_into_sheet_but_raised_wheel_stays():
    est = EgoEstimator(steward_bot().to_dict())
    sheet = sheet_track(width=.09)
    sheet.category_counts = {'picture': 5.}
    xx, yy = np.meshgrid(np.linspace(-.015, .015, 6), np.linspace(-.015, .015, 6))
    points = np.c_[xx.ravel() + 1., yy.ravel(), np.full(xx.size, .755)]
    left = measured_track(2, 'car_wheel', points + [-.035, 0., 0.])
    right = measured_track(3, 'car_wheel', points + [.035, 0., 0.])
    raised = measured_track(4, 'car_wheel', points + [0., 0., .016])
    for t in (left, right, raised):
        t.size[2] = .016  # prior-completed size must not decide coplanarity
    est.tracks = {t.id: t for t in (sheet, left, right, raised)}
    est._drop_printed_fragments()
    assert set(est.tracks) == {sheet.id, raised.id}
    assert sheet.semantic_counts == {'manual': 4., 'picture': 4.}


@pytest.mark.parametrize('score,height_shift', [(.3, 0.), (.8, .012)])
def test_weak_or_non_coplanar_sheet_cannot_delete_wheel(score, height_shift):
    est = EgoEstimator(steward_bot().to_dict())
    sheet = sheet_track(score=score)
    sheet.category_counts = {'picture': 5.}
    points = cloud(); points[:, 2] = .755 + height_shift
    wheel = measured_track(2, 'car_wheel', points)
    est.tracks = {t.id: t for t in (sheet, wheel)}
    est._drop_printed_fragments()
    assert len(est.tracks) == 2


def test_sensor_sheet_and_printed_wheels_export_one_sheet(monkeypatch):
    from steward.ego.perception import Segment
    est = EgoEstimator(steward_bot().to_dict())
    xx, yy = np.meshgrid(np.linspace(.94, 1.06, 24), np.linspace(-.045, .045, 18))
    points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, .755)]
    regions = [Segment(points=points, color=np.array([.5, .5, .5]), pixels=len(points),
                       invalid_frac=0., border=False, range=1.2, bottom_seen=True,
                       semantics={'picture': .8, 'scissors': .7}, span=box_corners(*box_of(points)))]
    for x in (.965, 1.035):
        part = points[(abs(points[:, 0] - x) < .015) & (abs(points[:, 1]) < .02)]
        regions.append(Segment(points=part, color=np.array([.1, .1, .1]), pixels=len(part),
                               invalid_frac=0., border=False, range=1.2, bottom_seen=True,
                               semantics={'car_wheel': .8}, span=box_corners(*box_of(part))))
    monkeypatch.setattr('steward.ego.estimator.segments', lambda *a, **kw: regions)
    for stamp in range(1, 7):
        est.update_rgbd({'header': {'stamp': stamp, 'odometry': est.odometry, 'joints': {}},
                         'depth': np.ones((40, 60)), 'fovy': 60.})
        assert len(est.tracks) == 1
        assert next(iter(est.tracks.values())).category == 'picture'
    assert sum(e.id in est.tracks for e in est.to_scene().entities) == 1


@pytest.mark.parametrize('method', ['_merge_duplicates', '_merge_same_place',
                                    '_merge_same_spot', '_merge_same_surface'])
def test_merged_close_screw_views_replace_old_elongated_geometry(method):
    est = EgoEstimator(steward_bot().to_dict())
    long = cloud(); long[:, 0] = 1. + (long[:, 0] - 1.) * (.055 / .016)
    old = measured_track(1, 'airplane_landing_gear', long, 20)
    old.mean_rgb = np.array([.55, .58, .62]); old.bottom_seen = True
    old.semantic_counts = {'toy_screw': 14.}
    for _ in range(6):
        old.recent.append((old.pos.copy(), old.size.copy(), 1.8))
    neighbour = track(2, .08)
    est.tracks = {t.id: t for t in (old, neighbour)}
    est._rebuild_geometry(old, np.array([0., 0., 1.2]), 1.8)
    assert old.category == 'airplane_landing_gear'
    for i in range(4):
        short = cloud(); short[:, 0] = 1. + (short[:, 0] - 1.) * (.035 / .016)
        # Same-spot and duplicate fusion require compatible names; surface/place
        # fusion also handles the historical screw -> gear disagreement.
        name = old.category if method in ('_merge_same_spot', '_merge_duplicates') else 'toy_screw'
        new = measured_track(i + 3, name, short, 1)
        new.first_seen = 2. + i
        new.semantic_counts = {'toy_screw': .7}
        new.recent.append((new.pos.copy(), new.size.copy(), 1.2))
        if name == 'airplane_landing_gear':
            new.size[0] = .08  # completion does not alter the recorded raw view
        est.tracks[new.id] = new
        fn = getattr(est, method)
        fn(np.array([0., 0., 1.2])) if method in ('_merge_same_place', '_merge_same_surface') else fn()
        est._rebuild_geometry(old, np.array([0., 0., 1.2]), 1.2)
    assert set(est.tracks) == {old.id, neighbour.id}
    assert old.category == 'toy_screw'
    assert max(old.size[:2]) < .04


def test_merging_real_gear_views_does_not_turn_it_into_a_screw():
    est = EgoEstimator(steward_bot().to_dict())
    points = cloud(); points[:, 0] = 1. + (points[:, 0] - 1.) * (.08 / .016)
    points[:, 1] *= .03 / .016
    a = measured_track(1, 'airplane_landing_gear', points)
    b = measured_track(2, 'airplane_landing_gear', points + [.004, 0., 0.])
    for t in (a, b):
        t.mean_rgb = np.array([.7, .72, .75]); t.bottom_seen = True
        for _ in range(3):
            t.recent.append((t.pos.copy(), t.size.copy(), 1.2))
    est.tracks = {t.id: t for t in (a, b)}
    est._merge_same_surface(np.array([0., 0., 1.2]))
    assert len(est.tracks) == 1
    assert next(iter(est.tracks.values())).category == 'airplane_landing_gear'
