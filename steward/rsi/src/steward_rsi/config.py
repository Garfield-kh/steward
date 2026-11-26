"""Read and validate ``rsi.toml``.

Relative paths in the file are resolved against the directory that holds the file. API keys are
wrapped in :class:`Secret`, whose ``repr``/``str`` never show the value.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from pathlib import Path

PROVIDERS = ("claude", "openai", "deepseek")
BACKENDS = ("gpu", "npu")
TASKS = ("car", "airplane")

PROVIDER_DEFAULTS = {
    "claude": {"model": "claude-opus-5-5", "base_url": "https://api.anthropic.com",
               "api_key_env": "ANTHROPIC_API_KEY", "effort": "high", "send_images": True},
    "openai": {"model": "REPLACE_WITH_OPENAI_MODEL", "base_url": "https://api.openai.com/v1",
               "api_key_env": "OPENAI_API_KEY", "effort": "high", "send_images": True},
    "deepseek": {"model": "deepseek-flash", "base_url": "https://api.deepseek.com",
                 "api_key_env": "DEEPSEEK_API_KEY", "effort": "high", "send_images": True,
                 "max_tokens": 32000},
}


class ConfigError(ValueError):
    """The configuration file is missing, unreadable or has an invalid value."""


class Secret:
    """An API key. Only :meth:`reveal` returns the value; printing the object shows ``<redacted>``."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(<redacted>)"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._value)


@dataclasses.dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str
    api_key: Secret
    effort: str
    send_images: bool
    max_images: int
    max_tokens: int
    timeout_s: float
    max_retries: int
    thinking: str  # Claude only: "" (omit) or "adaptive"
    anthropic_version: str


@dataclasses.dataclass
class MachineConfig:
    backend: str
    devices: list[int]
    episodes_per_device: int
    wall_minutes: float
    python: str
    base_port: int
    segmentation_configs: list[str]
    env: dict[str, str]


@dataclasses.dataclass
class TaskConfig:
    name: str
    task_file: Path
    seeds: list[int]
    threshold_mm: float
    strict_mm: float
    start_state_dir: Path | None
    kit_body: str
    phase: str


@dataclasses.dataclass
class RunConfig:
    sample_every_s: float
    frame_every_min: float
    startup_timeout_s: float
    build_timeout_s: float
    isolate_harness: bool
    robot_harness_template: Path
    pause_on_perception_fault: bool


@dataclasses.dataclass
class Config:
    path: Path
    advisor: ProviderConfig
    machine: MachineConfig
    task: TaskConfig
    run: RunConfig
    repo_root: Path
    output_dir: Path
    language: str

    @property
    def steward_dir(self) -> Path:
        return self.repo_root / "steward"


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _get(table: dict, key: str, kind, default, where: str):
    value = table.get(key, default)
    if value is None:
        return None
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ConfigError(f"{where}.{key} must be {kind.__name__}, got {type(value).__name__}")
    return value


def _path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else (base / path).resolve()


def _provider(data: dict, name: str) -> ProviderConfig:
    defaults = PROVIDER_DEFAULTS[name]
    table = _section(data, name)
    where = f"[{name}]"
    max_images = _get(table, "max_images", int, 6, where)
    if max_images < 0:
        raise ConfigError(f"{where}.max_images must be >= 0")
    thinking = _get(table, "thinking", str, "", where)
    if thinking not in ("", "adaptive"):
        raise ConfigError(f'{where}.thinking must be "" or "adaptive"')
    return ProviderConfig(
        name=name,
        model=_get(table, "model", str, defaults["model"], where),
        base_url=_get(table, "base_url", str, defaults["base_url"], where).rstrip("/"),
        api_key_env=_get(table, "api_key_env", str, defaults["api_key_env"], where),
        api_key=Secret(_get(table, "api_key", str, "", where)),
        effort=_get(table, "effort", str, defaults["effort"], where),
        send_images=_get(table, "send_images", bool, defaults["send_images"], where),
        max_images=max_images,
        max_tokens=_get(table, "max_tokens", int, defaults.get("max_tokens", 16000), where),
        timeout_s=_get(table, "timeout_s", float, 300.0, where),
        max_retries=_get(table, "max_retries", int, 3, where),
        thinking=thinking,
        anthropic_version=_get(table, "anthropic_version", str, "2023-06-01", where),
    )


def parse(data: dict, path: Path) -> Config:
    """Validate an already-decoded TOML document; ``path`` locates relative paths."""
    base = path.parent.resolve()
    advisor_table = _section(data, "advisor")
    provider = _get(advisor_table, "provider", str, "deepseek", "[advisor]")
    if provider not in PROVIDERS:
        raise ConfigError(f"[advisor].provider must be one of {', '.join(PROVIDERS)}; got {provider!r}")
    language = _get(advisor_table, "language", str, "English", "[advisor]")

    paths = _section(data, "paths")
    repo_root = _path(base, _get(paths, "repo_root", str, "../..", "[paths]"))
    output_dir = _path(base, _get(paths, "output_dir", str, "runs", "[paths]"))

    m = _section(data, "machine")
    backend = _get(m, "backend", str, "gpu", "[machine]")
    if backend not in BACKENDS:
        raise ConfigError(f"[machine].backend must be gpu or npu; got {backend!r}")
    devices = _get(m, "devices", list, [0, 1, 2, 3], "[machine]")
    if not devices or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in devices):
        raise ConfigError("[machine].devices must be a non-empty list of non-negative integers")
    if len(set(devices)) != len(devices):
        raise ConfigError("[machine].devices must not repeat a device")
    per_device = _get(m, "episodes_per_device", int, 1, "[machine]")
    if per_device < 1:
        raise ConfigError("[machine].episodes_per_device must be >= 1")
    wall = _get(m, "wall_minutes", float, 75.0, "[machine]")
    if wall <= 0:
        raise ConfigError("[machine].wall_minutes must be > 0")
    seg = _get(m, "segmentation_configs", list, [], "[machine]")
    if seg and len(seg) != len(devices):
        raise ConfigError("[machine].segmentation_configs must be empty or have one entry per device")
    env = _get(m, "env", dict, {}, "[machine]")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ConfigError("[machine].env values must be strings")
    base_port = _get(m, "base_port", int, 31000, "[machine]")
    if not 1024 <= base_port <= 65000:
        raise ConfigError("[machine].base_port must be between 1024 and 65000")
    python = _get(m, "python", str, "", "[machine]") or str(repo_root / "steward" / ".venv" / "bin" / "python")
    machine = MachineConfig(backend=backend, devices=devices, episodes_per_device=per_device, wall_minutes=wall,
                            python=str(_path(base, python)) if "/" in python else python,
                            base_port=base_port, segmentation_configs=[str(_path(base, s)) for s in seg], env=env)

    t = _section(data, "task")
    name = _get(t, "name", str, "airplane", "[task]")
    if name not in TASKS:
        raise ConfigError(f"[task].name must be car or airplane; got {name!r}")
    default_task_file = str(Path(__file__).resolve().parent / "tasks" / f"{name}.md")
    seeds = _get(t, "seeds", list, [0, 1, 2, 3], "[task]")
    if not seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise ConfigError("[task].seeds must be a non-empty list of integers")
    threshold = _get(t, "threshold_mm", float, 20.0, "[task]")
    strict = _get(t, "strict_mm", float, 6.0, "[task]")
    if threshold <= 0 or strict <= 0:
        raise ConfigError("[task].threshold_mm and strict_mm must be > 0")
    task = TaskConfig(name=name, task_file=_path(base, _get(t, "task_file", str, "", "[task]") or default_task_file),
                      seeds=seeds, threshold_mm=threshold, strict_mm=strict,
                      start_state_dir=_path(base, _get(t, "start_state_dir", str, "", "[task]")),
                      kit_body=_get(t, "kit_body", str, "", "[task]"),
                      phase=_get(t, "phase", str, "phase1-sam", "[task]"))

    r = _section(data, "run")
    template = _get(r, "robot_harness_template", str, "", "[run]")
    run = RunConfig(sample_every_s=_get(r, "sample_every_s", float, 30.0, "[run]"),
                    frame_every_min=_get(r, "frame_every_min", float, 10.0, "[run]"),
                    startup_timeout_s=_get(r, "startup_timeout_s", float, 900.0, "[run]"),
                    build_timeout_s=_get(r, "build_timeout_s", float, 1800.0, "[run]"),
                    isolate_harness=_get(r, "isolate_harness", bool, True, "[run]"),
                    robot_harness_template=_path(base, template) if template
                    else repo_root / "steward" / "var" / "dsh" / "robot-1",
                    pause_on_perception_fault=_get(r, "pause_on_perception_fault", bool, False, "[run]"))

    return Config(path=path, advisor=_provider(data, provider), machine=machine, task=task, run=run,
                  repo_root=repo_root, output_dir=output_dir, language=language)


def load(path: str | os.PathLike) -> Config:
    """Read ``path`` and return the validated configuration."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path} (copy rsi.example.toml to rsi.toml)")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from None
    return parse(data, path)


def resolve_api_key(provider: ProviderConfig, environ: dict | None = None) -> tuple[Secret, str]:
    """Return the key and where it came from (``"rsi.toml"`` or ``"env:NAME"``).

    A key written in the file takes precedence over the environment variable. The error message
    names the variable, never a value.
    """
    environ = os.environ if environ is None else environ
    if provider.api_key:
        return provider.api_key, "rsi.toml"
    value = environ.get(provider.api_key_env, "") if provider.api_key_env else ""
    if value:
        return Secret(value), f"env:{provider.api_key_env}"
    raise ConfigError(f"no API key for advisor {provider.name}: set [{provider.name}].api_key in rsi.toml "
                      f"or export {provider.api_key_env or '<api_key_env>'}")
