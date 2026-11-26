import base64
import json
import unittest

from helpers import config

from steward_rsi import providers
from steward_rsi.config import Secret

KEY = "sk-test-SECRET-0123456789"


class FakeHTTP:
    """Records requests and replays canned (status, payload) responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body), "timeout": timeout})
        status, payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


PNG = providers.Image("image/png", b"\x89PNG fake", "ep0 final-kit")
OPENAI_REPLY = {"model": "m", "choices": [{"message": {"content": "advice", "reasoning_content": "hidden"},
                                           "finish_reason": "stop"}], "usage": {"total_tokens": 9}}


def make(provider, http, **section):
    cfg = config({"advisor": {"provider": provider}, provider: section})
    return providers.make(cfg.advisor, Secret(KEY), http, sleep=lambda s: None)


class ClaudeTest(unittest.TestCase):
    def test_request_and_reply(self):
        http = FakeHTTP((200, {"model": "claude-opus-5-5", "stop_reason": "end_turn", "usage": {"output_tokens": 3},
                               "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}]}))
        p = make("claude", http, send_images=True, thinking="adaptive", base_url="https://gw.example/")
        reply = p.complete("sys", "question", [PNG])
        call = http.calls[0]
        self.assertEqual(call["url"], "https://gw.example/v1/messages")
        self.assertEqual(call["headers"]["x-api-key"], KEY)
        self.assertEqual(call["headers"]["anthropic-version"], "2023-06-01")
        body = call["body"]
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["system"], "sys")
        self.assertEqual(body["output_config"], {"effort": "high"})
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertNotIn("temperature", body)
        content = body["messages"][0]["content"]
        image = next(c for c in content if c["type"] == "image")
        self.assertEqual(image["source"], {"type": "base64", "media_type": "image/png",
                                           "data": base64.b64encode(PNG.data).decode()})
        self.assertEqual(content[-1], {"type": "text", "text": "question"})
        self.assertEqual(reply.text, "hello")
        self.assertEqual(reply.finish, "end_turn")

    def test_refusal_is_an_error(self):
        http = FakeHTTP((200, {"stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []}))
        with self.assertRaises(providers.ProviderError):
            make("claude", http).complete("s", "u")

    def test_refusal_details_are_redacted(self):
        http = FakeHTTP((200, {"stop_reason": "refusal", "content": [],
                               "stop_details": {"explanation": f"request carried {KEY}"}}))
        with self.assertRaises(providers.ProviderError) as ctx:
            make("claude", http).complete("s", "u")
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))

    def test_truncated_text_is_an_error(self):
        http = FakeHTTP((200, {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "1. The worst"}]}))
        with self.assertRaises(providers.ProviderError) as ctx:
            make("claude", http).complete("s", "u")
        self.assertIn("cut off", str(ctx.exception))


class OpenAITest(unittest.TestCase):
    def test_request_and_reply(self):
        http = FakeHTTP((200, OPENAI_REPLY))
        p = make("openai", http, model="my-model", effort="medium", max_images=1)
        reply = p.complete("sys", "question", [PNG, PNG])
        call = http.calls[0]
        self.assertEqual(call["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {KEY}")
        body = call["body"]
        self.assertEqual(body["model"], "my-model")
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(body["max_completion_tokens"], 16000)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "sys"})
        parts = body["messages"][1]["content"]
        urls = [x["image_url"]["url"] for x in parts if x["type"] == "image_url"]
        self.assertEqual(len(urls), 1)  # max_images
        self.assertTrue(urls[0].startswith("data:image/png;base64,"))
        self.assertEqual(reply.text, "advice")

    def test_no_effort_field_when_empty(self):
        http = FakeHTTP((200, OPENAI_REPLY))
        make("openai", http, effort="").complete("s", "u")
        self.assertNotIn("reasoning_effort", http.calls[0]["body"])


class DeepSeekTest(unittest.TestCase):
    def test_request_and_reply(self):
        http = FakeHTTP((200, OPENAI_REPLY))
        reply = make("deepseek", http).complete("sys", "question", [PNG])
        call = http.calls[0]
        self.assertEqual(call["url"], "https://api.deepseek.com/chat/completions")
        body = call["body"]
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["max_tokens"], 32000)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual([x["type"] for x in body["messages"][1]["content"]], ["text", "image_url", "text"])
        self.assertEqual(reply.text, "advice")  # reasoning_content is not advice

    def test_images_disabled(self):
        http = FakeHTTP((200, OPENAI_REPLY))
        make("deepseek", http, send_images=False).complete("s", "u", [PNG])
        self.assertEqual(http.calls[0]["body"]["messages"][1]["content"], "u")


class RetryAndRedactionTest(unittest.TestCase):
    def test_retries_then_succeeds(self):
        http = FakeHTTP((503, b"busy"), (429, b"slow down"), (200, OPENAI_REPLY))
        self.assertEqual(make("deepseek", http, max_retries=3).complete("s", "u").text, "advice")
        self.assertEqual(len(http.calls), 3)

    def test_network_error_retried(self):
        http = FakeHTTP((0, OSError("connection reset")), (200, OPENAI_REPLY))
        self.assertEqual(make("deepseek", http).complete("s", "u").text, "advice")

    def test_gives_up_after_limit(self):
        http = FakeHTTP(*[(500, b"down")] * 3)
        with self.assertRaises(providers.ProviderError):
            make("deepseek", http, max_retries=2).complete("s", "u")
        self.assertEqual(len(http.calls), 3)

    def test_client_error_not_retried_and_key_redacted(self):
        echo = json.dumps({"error": f"invalid key {KEY}"}).encode()
        http = FakeHTTP((401, echo), (200, OPENAI_REPLY))
        with self.assertRaises(providers.ProviderError) as ctx:
            make("openai", http).complete("s", "u")
        self.assertEqual(len(http.calls), 1)
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))


class EmptyAndRefusalTest(unittest.TestCase):
    def test_openai_refusal_is_an_error(self):
        reply = {"choices": [{"message": {"content": None, "refusal": f"I cannot help with {KEY}"},
                              "finish_reason": "stop"}]}
        with self.assertRaises(providers.ProviderError) as ctx:
            make("openai", FakeHTTP((200, reply))).complete("s", "u")
        self.assertIn("declined", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))

    def test_empty_reply_is_an_error_for_every_provider(self):
        empty_chat = {"choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]}
        empty_claude = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "..."}]}
        for provider, reply in (("openai", empty_chat), ("deepseek", empty_chat), ("claude", empty_claude)):
            with self.subTest(provider=provider):
                with self.assertRaises(providers.ProviderError) as ctx:
                    make(provider, FakeHTTP((200, reply))).complete("s", "u")
                self.assertIn("no text", str(ctx.exception))

    def test_truncated_chat_text_is_an_error(self):
        reply = {"choices": [{"message": {"content": "partial advice"}, "finish_reason": "length"}]}
        for provider in ("openai", "deepseek"):
            with self.subTest(provider=provider):
                with self.assertRaises(providers.ProviderError):
                    make(provider, FakeHTTP((200, reply))).complete("s", "u")


class TruncationTest(unittest.TestCase):
    def test_empty_reply_cut_by_token_limit(self):
        reply = {"model": "m", "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                 "usage": {"completion_tokens_details": {"reasoning_tokens": 4000}}}
        with self.assertRaises(providers.ProviderError) as ctx:
            make("deepseek", FakeHTTP((200, reply))).complete("s", "u")
        self.assertIn("max_tokens", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
