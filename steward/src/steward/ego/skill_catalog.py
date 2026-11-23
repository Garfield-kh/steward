"""Public skill names advertised by each runtime profile.

Dispatch tests exercise every entry against the runtime and executor. Internal
controller primitives are deliberately absent from the model-facing catalogue.
"""
from steward.assembly_profile import enabled

_COMMON = ('regrasp_upright', 'assemble_onto', 'fasten_onto', 'seat_screw',
           'drive_screw', 'flip_assembly')
_ORDINARY = ('navigate', 'pick', 'place', 'pour', 'open', 'close')
PUBLIC_SKILLS = (('complete_assembly', *_COMMON, *_ORDINARY) if enabled()
                 else (*_COMMON, 'insert_screw_into_hole', *_ORDINARY))
ESIM_SKILLS = tuple(skill for skill in PUBLIC_SKILLS if skill != 'complete_assembly')
