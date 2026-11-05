"""The perception services file: validation, URL building and fallback to the next base URL."""

from __future__ import annotations

import pytest

from steward import perception_services as ps
from steward.net import Binary, JsonServer

VALID = {"title": '"t"', "kind": '"k"', "docs": '"d"', "health": '"/h"', "urls": '["http://h"]', "timeout_s": "5"}


def _write(tmp_path, fields: dict[str, str], endpoint: str = '"GET /p"'):
    lines = ["[services.a]", *(f"{k} = {v}" for k, v in fields.items()), "[services.a.endpoints]", f"x = {endpoint}"]
    path = tmp_path / "s.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_shipped_file_lists_every_catalogued_service():
    services = ps.load()
    assert set(services) == {"sam3", "novelanything", "nvs", "slarm", "sam3d"}
    for svc in services.values():
        assert len(svc.urls) >= 2, f"{svc.name} needs a fallback URL"
    method, url = services["sam3"].url(services["sam3"].urls[0], "propagate", sid="s1")
    assert (method, url) == ("POST", "https://linxlab-research.com/sam3/api/sessions/s1/propagate")


@pytest.mark.parametrize("key, value", [("urls", "[]"), ("urls", '["ftp://x"]'), ("timeout_s", "0"), ("health", '"h"')])
def test_malformed_field_fails_naming_the_key(tmp_path, key, value):
    with pytest.raises(ps.ServiceConfigError, match=f"services.a.{key}"):
        ps.load(_write(tmp_path, VALID | {key: value}))


@pytest.mark.parametrize("endpoint", ['"FETCH /p"', '"GET p"'])
def test_malformed_endpoint_fails_naming_the_endpoint(tmp_path, endpoint):
    with pytest.raises(ps.ServiceConfigError, match="services.a.endpoints.x"):
        ps.load(_write(tmp_path, VALID, endpoint))


def test_check_skips_an_unreachable_or_non_json_base_url():
    blocked = JsonServer({("GET", "/svc/h"): lambda q, b: Binary(b"<html>blocked</html>", "text/html")}).start()
    healthy = JsonServer({("GET", "/svc/h"): lambda q, b: {"status": "ready"}}).start()
    try:
        svc = ps.Service(name="svc", title="t", kind="k", docs="d", health="/h", timeout_s=5,
                         urls=("http://127.0.0.1:9/svc", f"{blocked.url}/svc", f"{healthy.url}/svc"),
                         endpoints={"x": ps.Endpoint("GET", "/x")})
        health = ps.check(svc, timeout=2)
        assert health.ok and health.url == f"{healthy.url}/svc" and health.detail == "ready"
        assert ps.resolve(svc, timeout=2) == f"{healthy.url}/svc"
    finally:
        blocked.stop()
        healthy.stop()


def test_resolve_raises_when_nothing_answers():
    svc = ps.Service(name="svc", title="t", kind="k", docs="d", health="/h", timeout_s=5,
                     urls=("http://127.0.0.1:9/svc",), endpoints={"x": ps.Endpoint("GET", "/x")})
    with pytest.raises(ConnectionError, match="svc: no reachable URL"):
        ps.resolve(svc, timeout=1)
