"""Catalogue of remote perception and reconstruction services (``config/perception-services.toml``).

The file lists, per service, base URLs in preference order, its health and documentation paths
and its endpoints. :func:`load` validates the whole file and fails on the first malformed entry;
:func:`resolve` picks the first base URL whose health endpoint answers with a JSON body, because
the gateway's domain name is intermittently blocked on the network path while its IP address is
not. Nothing in the Ego Runtime calls these services yet; see steward.md §13.7.6.
"""

from __future__ import annotations

import json
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "perception-services.toml"
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")


class ServiceConfigError(ValueError):
    """The services file is malformed; the message names the file and the offending key."""


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str


@dataclass(frozen=True)
class Service:
    name: str
    title: str
    kind: str
    urls: tuple[str, ...]
    docs: str
    health: str
    timeout_s: float
    endpoints: dict[str, Endpoint]
    openapi: str | None = None
    capabilities: str | None = None

    def url(self, base: str, endpoint: str, **params: str) -> tuple[str, str]:
        """Method and absolute URL of ``endpoint`` under ``base``, with ``{name}`` path parameters filled in."""
        ep = self.endpoints[endpoint]
        return ep.method, base.rstrip("/") + ep.path.format(**params)


@dataclass(frozen=True)
class Health:
    service: str
    url: str | None
    ok: bool
    detail: str


def _path(where: str, value: object) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ServiceConfigError(f"{where}: expected a path starting with '/', got {value!r}")
    return value


def _service(name: str, raw: dict, source: Path) -> Service:
    where = f"{source}: services.{name}"
    for key in ("title", "kind", "docs", "health"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ServiceConfigError(f"{where}.{key}: required string")
    urls = raw.get("urls")
    if not isinstance(urls, list) or not urls or not all(isinstance(u, str) and u.startswith(("http://", "https://")) for u in urls):
        raise ServiceConfigError(f"{where}.urls: required non-empty list of http(s) URLs")
    timeout = raw.get("timeout_s")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ServiceConfigError(f"{where}.timeout_s: required positive number")
    endpoints: dict[str, Endpoint] = {}
    for key, spec in (raw.get("endpoints") or {}).items():
        method, _, path = str(spec).partition(" ")
        if method not in METHODS:
            raise ServiceConfigError(f"{where}.endpoints.{key}: expected '<METHOD> /path', got {spec!r}")
        endpoints[key] = Endpoint(method, _path(f"{where}.endpoints.{key}", path))
    if not endpoints:
        raise ServiceConfigError(f"{where}.endpoints: at least one endpoint required")
    optional = {k: _path(f"{where}.{k}", raw[k]) for k in ("openapi", "capabilities") if k in raw}
    return Service(name=name, title=raw["title"], kind=raw["kind"], urls=tuple(u.rstrip("/") for u in urls),
                   docs=raw["docs"], health=_path(f"{where}.health", raw["health"]), timeout_s=float(timeout),
                   endpoints=endpoints, **optional)


def load(path: Path = DEFAULT_PATH) -> dict[str, Service]:
    """Every service in ``path``, keyed by name. Raises :class:`ServiceConfigError` on a malformed entry."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ServiceConfigError(f"{path}: {exc}") from None
    services = data.get("services")
    if not isinstance(services, dict) or not services:
        raise ServiceConfigError(f"{path}: no [services.<name>] tables")
    return {name: _service(name, raw, path) for name, raw in services.items()}


def check(service: Service, timeout: float = 8.0) -> Health:
    """Probe each base URL's health endpoint in order; the first JSON answer with status 200 wins.

    A blocked domain answers with an HTML block page or resets the connection, so a non-JSON body
    counts as unreachable rather than as a service fault."""
    failures = []
    for base in service.urls:
        try:
            with urllib.request.urlopen(base + service.health, timeout=timeout) as resp:
                body = json.loads(resp.read())
            status = body.get("status", "ok") if isinstance(body, dict) else "ok"
            return Health(service.name, base, True, str(status))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            failures.append(f"{base}: {getattr(exc, 'code', '') or type(exc).__name__}")
    return Health(service.name, None, False, "; ".join(failures))


def resolve(service: Service, timeout: float = 8.0) -> str:
    """The first reachable base URL of ``service``; raises ``ConnectionError`` listing every attempt."""
    health = check(service, timeout)
    if health.url is None:
        raise ConnectionError(f"{service.name}: no reachable URL ({health.detail})")
    return health.url
