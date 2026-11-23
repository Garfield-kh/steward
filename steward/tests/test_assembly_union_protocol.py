"""Self-contained assembly regressions using bundled inputs and synthetic fixtures."""
import copy
import math

import pytest


def assembly_body(body, part, released=10, observed=11):
    merged = copy.deepcopy(body)
    merged['sizeM'] = [.23, .20, .09]
    merged['orientation'] = {'posture': {'status': 'unknown', 'upright': None},
        'assembly': {'status': 'stationary_envelope_observed', 'observedAt': observed,
                     'reference': {'entity': body['id'], 'position': body['position'],
                                   'size': body['sizeM'], 'yaw': math.radians(body['yawDeg'])},
                     'parts': [{'entity': part['id'], 'releasedAt': released,
                                'status': 'envelope_observed', 'observedAt': observed}]}}
    return merged
