"""Per-frame clip planes; caller owns model/renderer lock through all render passes."""
from contextlib import contextmanager


@contextmanager
def clip_planes(model, sensor):
    previous = model.vis.map.znear
    try:
        if 'renderNearM' in sensor:
            model.vis.map.znear = float(sensor['renderNearM']) / float(model.stat.extent)
        yield
    finally:
        model.vis.map.znear = previous
