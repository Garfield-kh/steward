"""Advisor adapters: Anthropic Messages, OpenAI Chat Completions, DeepSeek (OpenAI-compatible).

Each adapter builds one request from a system prompt, a user prompt and optional images, sends it
through a replaceable transport with a timeout and bounded retries, and returns the reply text.
Every :class:`ProviderError` message is passed through :func:`redact`, so the API key never
appears in it. A refusal, an empty reply or a reply cut off by the token limit is an error, never
returned as advice.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .config import ProviderConfig, Secret

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
# Finish reasons that mean the reply text is incomplete (Anthropic stop_reason, OpenAI finish_reason).
TRUNCATED = {"length", "max_tokens", "model_context_window_exceeded"}

# transport(url, headers, body, timeout_s) -> (status, response bytes)
Transport = Callable[[str, dict, bytes, float], tuple[int, bytes]]


class ProviderError(RuntimeError):
    """A request failed; the message holds the status and a redacted response excerpt."""


@dataclasses.dataclass
class Image:
    media_type: str
    data: bytes
    label: str = ""

    @classmethod
    def from_file(cls, path: Path, label: str = "") -> "Image":
        suffix = path.suffix.lower()
        media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix)
        if media is None:
            raise ValueError(f"unsupported image type: {path.name}")
        return cls(media, path.read_bytes(), label or path.name)

    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclasses.dataclass
class Reply:
    text: str
    model: str
    finish: str
    usage: dict


def redact(text: str, secrets: list[Secret | str]) -> str:
    for secret in secrets:
        value = secret.reveal() if isinstance(secret, Secret) else secret
        if value:
            text = text.replace(value, "<redacted>")
    return text


def urllib_transport(url: str, headers: dict, body: bytes, timeout_s: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class Provider:
    """Common send/retry logic; subclasses define the URL, headers, body and reply parsing."""

    def __init__(self, cfg: ProviderConfig, key: Secret, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.key = key
        self.transport = transport or urllib_transport
        self.sleep = sleep

    def url(self) -> str:
        raise NotImplementedError

    def headers(self) -> dict:
        raise NotImplementedError

    def body(self, system: str, user: str, images: list[Image], max_tokens: int) -> dict:
        raise NotImplementedError

    def parse(self, payload: dict) -> Reply:
        """Reply of a 200 response; raises through :meth:`fail` on a refusal."""
        raise NotImplementedError

    def fail(self, message: str) -> ProviderError:
        """A :class:`ProviderError` whose message has the API key redacted."""
        return ProviderError(redact(message, [self.key]))

    def complete(self, system: str, user: str, images: list[Image] | None = None,
                 max_tokens: int | None = None) -> Reply:
        """Send one request and return a reply with complete, non-empty text.

        Raises :class:`ProviderError` when the request fails, the model refuses, the reply is cut
        off by the token limit, or the reply has no text.
        """
        reply = self._send(system, user, images, max_tokens)
        if reply.finish in TRUNCATED:
            where = "before any text" if not reply.text.strip() else f"after {len(reply.text)} characters"
            raise self.fail(f"{self.cfg.name}: the reply was cut off by the token limit {where} "
                            f"(finish={reply.finish}, usage={json.dumps(reply.usage)}); raise "
                            f"[{self.cfg.name}].max_tokens or lower effort")
        if not reply.text.strip():
            raise self.fail(f"{self.cfg.name}: the reply has no text (finish={reply.finish or 'none'}, "
                            f"usage={json.dumps(reply.usage)})")
        return reply

    def _send(self, system: str, user: str, images: list[Image] | None, max_tokens: int | None) -> Reply:
        images = list(images or []) if self.cfg.send_images else []
        images = images[: self.cfg.max_images]
        body = json.dumps(self.body(system, user, images, max_tokens or self.cfg.max_tokens)).encode()
        last = ""
        for attempt in range(self.cfg.max_retries + 1):
            try:
                status, raw = self.transport(self.url(), self.headers(), body, self.cfg.timeout_s)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last = f"network error: {type(error).__name__}: {error}"
            else:
                if status == 200:
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        raise self.fail(f"{self.cfg.name}: response is not JSON: {raw[:300]!r}") from None
                    if not isinstance(payload, dict):
                        raise self.fail(f"{self.cfg.name}: response is not a JSON object: {raw[:300]!r}")
                    return self.parse(payload)
                excerpt = raw[:500].decode("utf-8", "replace")
                last = f"HTTP {status}: {excerpt}"
                if status not in RETRY_STATUS:
                    break
            if attempt < self.cfg.max_retries:
                self.sleep(min(60.0, 2.0 * 2 ** attempt))
        raise self.fail(f"{self.cfg.name} request to {self.url()} failed: {last}")


class ClaudeProvider(Provider):
    """Anthropic Messages API: ``POST {base_url}/v1/messages``."""

    def url(self) -> str:
        return f"{self.cfg.base_url}/v1/messages"

    def headers(self) -> dict:
        return {"x-api-key": self.key.reveal(), "anthropic-version": self.cfg.anthropic_version,
                "content-type": "application/json"}

    def body(self, system, user, images, max_tokens):
        content = []
        for image in images:
            if image.label:
                content.append({"type": "text", "text": f"Image: {image.label}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": image.media_type,
                                                         "data": image.b64()}})
        content.append({"type": "text", "text": user})
        body = {"model": self.cfg.model, "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": content}]}
        if self.cfg.effort:
            body["output_config"] = {"effort": self.cfg.effort}
        if self.cfg.thinking:
            body["thinking"] = {"type": self.cfg.thinking}
        return body

    def parse(self, payload):
        text = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text")
        finish = payload.get("stop_reason") or ""
        if finish == "refusal":
            raise self.fail(f"claude declined the request (stop_reason=refusal, "
                            f"details={json.dumps(payload.get('stop_details'))})")
        return Reply(text=text, model=payload.get("model", self.cfg.model), finish=finish,
                     usage=payload.get("usage") or {})


class OpenAIProvider(Provider):
    """OpenAI Chat Completions: ``POST {base_url}/chat/completions``."""

    token_field = "max_completion_tokens"

    def url(self) -> str:
        return f"{self.cfg.base_url}/chat/completions"

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key.reveal()}", "Content-Type": "application/json"}

    def body(self, system, user, images, max_tokens):
        parts = []
        for image in images:
            if image.label:
                parts.append({"type": "text", "text": f"Image: {image.label}"})
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{image.media_type};base64,{image.b64()}"}})
        parts.append({"type": "text", "text": user})
        body = {"model": self.cfg.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": parts if images else user}],
                self.token_field: max_tokens}
        if self.cfg.effort:
            body["reasoning_effort"] = self.cfg.effort
        return body

    def parse(self, payload):
        choices = payload.get("choices") or []
        if not choices:
            raise self.fail(f"{self.cfg.name}: response has no choices")
        message = choices[0].get("message") or {}
        if message.get("refusal"):
            raise self.fail(f"{self.cfg.name} declined the request: {str(message['refusal'])[:500]}")
        return Reply(text=message.get("content") or "", model=payload.get("model", self.cfg.model),
                     finish=choices[0].get("finish_reason") or "", usage=payload.get("usage") or {})


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek Chat Completions (OpenAI-compatible): ``POST {base_url}/chat/completions``.

    DeepSeek takes ``max_tokens``; ``reasoning_content`` in the reply is not returned as advice.
    """

    token_field = "max_tokens"


ADAPTERS = {"claude": ClaudeProvider, "openai": OpenAIProvider, "deepseek": DeepSeekProvider}


def make(cfg: ProviderConfig, key: Secret, transport: Transport | None = None, **kwargs) -> Provider:
    return ADAPTERS[cfg.name](cfg, key, transport, **kwargs)
