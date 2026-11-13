"""Probe a running Ego API for face-match deadlines and blocking, without moving the robot.

Uses only Python's standard library. Stop other agent requests before running; the robot must
already have observed the spread puzzle and its lid. HTTP observations cannot measure CPU use.
"""

from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
import select
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit


def request(api: str, path: str, timeout: float, body: dict | None = None) -> dict:
    """Return response fields and monotonic elapsed seconds, including HTTP/transport failures."""
    url = urlsplit(api)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    start = time.monotonic()
    result: dict = {}
    try:
        connection.request("GET" if body is None else "POST", path,
                           None if body is None else json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        result["http"] = response.status
        raw = response.read().decode()
        try:
            result["body"] = json.loads(raw)
        except ValueError:
            result["raw"] = raw[:2000]
    except (OSError, http.client.HTTPException) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    result["elapsed_s"] = round(time.monotonic() - start, 3)
    return result


def discover(summary: dict, reference: str | None, entities: list[str] | None) -> dict:
    """Select exact puzzle_piece/lid categories; reject missing or ambiguous selections."""
    objects = {obj["id"]: obj for obj in summary["objects"]}
    pieces = entities or sorted(obj["id"] for obj in objects.values()
                               if obj.get("category") == "puzzle_piece" and obj.get("existence", 0) >= 0.5)
    lids = [reference] if reference else [obj["id"] for obj in objects.values()
                                         if obj.get("category") == "lid" and obj.get("existence", 0) >= 0.5]
    if not pieces or len(lids) != 1:
        raise ValueError(f"发现 {len(pieces)} 片拼图、{len(lids)} 个参考图候选。请先观察拼图及盒盖；"
                         "多盒盖时用 --reference，类别识别有误时用 --entities 指定 Ego ID。")
    selected = [*pieces, lids[0]]
    if len(set(selected)) != len(selected) or any(eid not in objects for eid in selected):
        raise ValueError("所选 ID 重复或不在当前 Ego summary 中。")
    return {"entities": pieces, "reference": lids[0]}


def input_records(api: str, body: dict, timeout: float) -> dict:
    """Count applied input records; these do not establish correlation completion or success."""
    response = request(api, "/ego/inputs", timeout)
    if response.get("http") != 200 or "inputs" not in response.get("body", {}):
        return {"error": response}
    return {"count": sum(item.get("path") == "/ego/match-face" and item.get("body") == body
                         for item in response["body"]["inputs"])}


def trial(api: str, body: dict, deadline: float, recovery: float, baseline: float,
          threshold: float, mode: str) -> dict:
    """Disconnect one match at its deadline and wait for concurrent and subsequent state probes.

    State recovery and applied input counts can expose the old lock-held request continuing after
    disconnect. They cannot establish CPU use or settlement when computation does not hold that lock.
    """
    before = input_records(api, body, recovery)
    url = urlsplit(api)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=recovery)
    probe: dict = {}
    started = time.monotonic()
    match: dict = {}

    def probe_state() -> None:
        time.sleep(min(0.25, deadline / 4))
        probe.update(request(api, "/ego/summary", recovery))
        probe["finished_after_start_s"] = round(time.monotonic() - started, 3)

    try:
        connection.request("POST", "/ego/match-face", json.dumps(body),
                           {"Content-Type": "application/json"})
        started = time.monotonic()
        worker = threading.Thread(target=probe_state)
        worker.start()
        end = started + deadline
        # Waiting for response headers covers the server's synchronous computation. The socket
        # deadline also bounds a response whose headers arrive before its body is complete.
        if not select.select([connection.sock], [], [], deadline)[0]:
            match["outcome"] = "client_timeout" if mode == "timeout" else "client_cancelled"
        else:
            connection.sock.settimeout(max(0.001, end - time.monotonic()))
            response = connection.getresponse()
            match.update(outcome="response", http=response.status)
            match["body"] = json.loads(response.read())
    except (TimeoutError, socket.timeout):
        match["outcome"] = "client_timeout" if mode == "timeout" else "client_cancelled"
    except (OSError, ValueError, http.client.HTTPException) as exc:
        match.update(outcome="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        match["elapsed_s"] = round(time.monotonic() - started, 3)
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # The peer may already have closed the response connection.
        connection.close()
    disconnected = time.monotonic()
    after_disconnect = request(api, "/ego/summary", recovery)
    if "worker" in locals():
        worker.join()
    after = input_records(api, body, recovery)
    valid = match.get("http") == 200 and any(
        result.get("matches") for result in match.get("body", {}).get("results", []))
    cutoff = max(baseline * 3, baseline + threshold)
    interrupted = match["outcome"] in ("client_timeout", "client_cancelled")
    recovered = after_disconnect.get("http") == 200 and probe.get("http") == 200
    delayed = after_disconnect["elapsed_s"] > cutoff
    grew = "count" in before and "count" in after and after["count"] > before["count"]
    return {
        "mode": mode, "deadline_s": deadline, "match": match,
        "concurrent_summary": probe, "summary_after_disconnect": after_disconnect,
        "input_records_before": before, "input_records_after": after,
        "blocking_threshold_s": cutoff,
        "state_blocking_observed": probe.get("http") == 200 and probe["elapsed_s"] > cutoff,
        "handler_continued_after_disconnect_evidence": interrupted and delayed and grew and recovered,
        "cpu_still_computing": "unknown: HTTP cannot measure server CPU",
        "settlement_after_disconnect": "unknown: input records and responsive state do not prove correlation stopped" if interrupted else "response received",
        "valid_matches_received": bool(valid), "recovered": recovered,
        "finished_after_disconnect_s": round(time.monotonic() - disconnected, 3),
    }


def main() -> int:
    """Run sequential probes and save all measurements, including incomplete diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:7420", help="Ego API URL, not World UI")
    parser.add_argument("--reference", help="Override reference Ego ID")
    parser.add_argument("--entities", nargs="+", help="Override puzzle Ego IDs")
    parser.add_argument("--px", type=int, choices=range(4, 33), default=24)
    parser.add_argument("--timeout", type=float, default=15, help="Match client deadline in seconds")
    parser.add_argument("--cancel-after", type=float, default=2, help="Close the second request after seconds")
    parser.add_argument("--recovery-timeout", type=float, default=120, help="Per-probe wait for recovery")
    parser.add_argument("--delay-threshold", type=float, default=1, help="Extra seconds above baseline")
    parser.add_argument("--output", type=Path, default=Path("match-diagnosis.json"))
    args = parser.parse_args()
    url = urlsplit(args.api)
    if url.scheme != "http" or not url.hostname or url.path not in ("", "/") or url.query or url.fragment:
        parser.error("--api 必须是 http://host:port，不带路径、查询或片段")
    for value in (args.timeout, args.cancel_after, args.recovery_timeout, args.delay_threshold):
        if not 0 < value < float("inf"):
            parser.error("所有时间参数必须是有限正数")
    report: dict = {"api": args.api, "trials": [],
                    "limitations": "请停止其他模型/工具请求。延迟和完成记录不能证明 CPU 持续计算；"
                                    "取消没有返回匹配结果时，无法排除观测不足等失败。"}
    code = 0
    try:
        baseline = [request(args.api, "/ego/summary", args.recovery_timeout) for _ in range(3)]
        report["baseline"] = baseline
        if any(item.get("http") != 200 for item in baseline):
            raise ValueError("基线状态查询失败，请检查 Ego API 地址和服务状态。")
        body = {**discover(baseline[-1]["body"], args.reference, args.entities), "px": args.px}
        report["request"] = body
        report["selected_appearances"] = {
            obj["id"]: obj.get("appearance") for obj in baseline[-1]["body"]["objects"]
            if obj["id"] in [*body["entities"], body["reference"]]
        }
        median = statistics.median(item["elapsed_s"] for item in baseline)
        print(f"发现 {len(body['entities'])} 片拼图；参考图 {body['reference']}；基线 {median:.3f}s", flush=True)
        print(json.dumps(body, ensure_ascii=False), flush=True)
        for mode, deadline in (("timeout", args.timeout), ("cancel", args.cancel_after)):
            print(f"开始 {mode}：{deadline}s 后断开匹配请求；等待状态恢复……", flush=True)
            result = trial(args.api, body, deadline, args.recovery_timeout, median, args.delay_threshold, mode)
            report["trials"].append(result)
            print(f"{mode}: {result['match']['outcome']}；状态阻塞={result['state_blocking_observed']}；"
                  f"断开后处理继续的证据={result['handler_continued_after_disconnect_evidence']}；"
                  f"恢复={result['recovered']}", flush=True)
            print(f"  并发状态查询 {result['concurrent_summary'].get('elapsed_s', '未启动')}s；"
                  f"断开后状态查询 {result['summary_after_disconnect']['elapsed_s']}s", flush=True)
            if not result["recovered"]:
                raise ValueError("状态未在等待上限内恢复；不再派发匹配，请检查服务。")
            if result["match"]["outcome"] in ("client_timeout", "client_cancelled"):
                if not result["handler_continued_after_disconnect_evidence"]:
                    report["remaining_trials_skipped"] = "状态可用但无法从 HTTP 确认匹配已停止，避免派发重叠匹配。"
                    print(report["remaining_trials_skipped"], flush=True)
                    code = 2
                    break
            if result["match"]["outcome"] == "error" or (
                result["match"]["outcome"] == "response" and not result["valid_matches_received"]
            ):
                raise ValueError("匹配未产生有效候选，请查看 JSON 错误并补充观测后重试。")
        print("false 表示本次未观察到证据，不表示问题已修复；CPU 是否持续计算仍为 unknown。", flush=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
        print(f"诊断不完整：{exc}", flush=True)
        code = 2
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"结果：{args.output.resolve()}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
