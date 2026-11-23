"""Opt-in settings for the toy assembly runtime.

Select the profile before starting either service. Explicit environment values
win over the profile; world fastening and no-slip settings remain independent.
"""
import os

_PROFILE = os.environ.get("STEWARD_ASSEMBLY_PROFILE", "")
if _PROFILE not in ("", "off", "toy"):
    raise ValueError("STEWARD_ASSEMBLY_PROFILE must be off or toy")

TOY_DEFAULTS = {
    "SUBGOAL_SUCCESS": "0",
    "STEWARD_WRIST_CAMERA": "1",
    "STEWARD_RGBD_STORED": "1",
    "STEWARD_PERCEPTION_HZ": "5",
    "STEWARD_ASSEMBLY_DRIVE_BUDGET_S": "600",
    "STEWARD_ASSEMBLY_LANDING_CHECK": "0",
    "STEWARD_ASSEMBLY_DESTINATION_CHECK": "0",
    "STEWARD_ASSEMBLY_RELEASE_OBSERVATION": "0",
    "STEWARD_ASSEMBLY_COMPONENT_LANDING_CHECK": "0",
    **{n: "0" for n in ('STEWARD_ASSEMBLY_SESSION_FACES', 'STEWARD_ASSEMBLY_INSERTION_SEARCH', 'STEWARD_ASSEMBLY_SESSION_INVENTORY', 'STEWARD_ASSEMBLY_RELEASE_CONDITIONS', 'STEWARD_ASSEMBLY_RELEASE_RETENTION', 'STEWARD_ASSEMBLY_RELEASE_PLANE', 'STEWARD_ASSEMBLY_FIRST_INTERFACE', 'STEWARD_ASSEMBLY_SESSION_IDENTITY')},
}


def enabled() -> bool:
    """Whether this process selects the toy assembly implementation."""
    return _PROFILE == "toy"


def apply_defaults() -> None:
    """Apply the selected profile once, preserving every explicit override."""
    if enabled():
        for name, value in TOY_DEFAULTS.items():
            os.environ.setdefault(name, value)


def harness_environment(file_values: dict[str, str]) -> dict[str, str]:
    """Merge Harness dotenv values while keeping an active toy process configuration.

    The default profile retains the upstream dotenv precedence. In a toy process,
    inherited simulation settings must also reach its Node child unchanged.
    """
    result = os.environ | file_values
    if enabled():
        result.update({key: value for key, value in os.environ.items()
                       if key.startswith("STEWARD_") or key == "SUBGOAL_SUCCESS"})
    return result


def validate_toy_model(scenario: dict) -> None:
    """Refuse a cached assembly scene built with a different robot configuration."""
    from steward.assets.generators import steward_bot
    from steward.assets.library import template_hash

    expected = template_hash(steward_bot().to_dict())
    robots = {entity['id']: entity for entity in scenario['sceneState']['entities']
              if entity['id'] in scenario['robots']}
    for robot in robots.values():
        asset = robot.get('asset') or {}
        if asset.get('id') == 'steward-bot' and asset.get('version') != expected:
            raise ValueError('toy-assembly has a cached robot from a different configuration; '
                             'run build-toy-assembly with the current STEWARD_ASSEMBLY_PROFILE '
                             'and camera settings in a separate data root before up')
