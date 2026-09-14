#!/usr/bin/env python3
"""Measure provider-reported low-effort reasoning length without formal scoring."""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import (  # noqa: E402
    Provider,
    Tokenizer,
    append_jsonl,
    digest,
    load_json,
    public_text,
    read_jsonl,
    write_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc: BaseException) -> str:
    return re.sub(r"(?i)(bearer\s+)[^\s,}\"]+", r"\1[REDACTED]", str(exc))[:2000]


def usage_reasoning_tokens(result: dict) -> int | None:
    usage = result.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    for container, key in ((details, "reasoning_tokens"), (usage, "reasoning_tokens")):
        value = container.get(key) if isinstance(container, dict) else None
        if isinstance(value, (int, float)):
            return int(value)
    return None


def metadata_from_exception(exc: BaseException) -> dict:
    metadata = getattr(exc, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/models_opencode_go_reasoning_length_diagnostic_v1.json",
    )
    parser.add_argument(
        "--histories",
        default="work/p0_calibration_data_glm_20260911/histories.jsonl",
    )
    parser.add_argument(
        "--prompt",
        default="prompts/compress_neutral_calibration.txt",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--technical-budget", type=int, default=131072)
    args = parser.parse_args()

    config_path = ROOT / args.config
    histories_path = ROOT / args.histories
    prompt_path = ROOT / args.prompt
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    configured_budget = int(config["compressor"]["provider_output_budget_tokens"])
    if args.technical_budget != configured_budget:
        raise SystemExit(
            f"technical budget must match frozen config: {configured_budget}"
        )
    histories = read_jsonl(histories_path)
    if len(histories) < 8:
        raise SystemExit(f"expected at least 8 histories, found {len(histories)}")
    if args.limit != 10:
        raise SystemExit("this diagnostic is fixed to exactly 10 calls")

    selected: list[tuple[dict, int, str]] = [
        (history, 0, "unique") for history in histories[:8]
    ]
    selected.extend((histories[i], 1, "repeat") for i in range(2))

    run_start = utc_now()
    config["run_start_utc"] = run_start
    tokenizer = Tokenizer(config["compressor"]["tokenizer"])
    prompt_template = prompt_path.read_text(encoding="utf-8")
    system = prompt_template.format(budget=1024, tokenizer=tokenizer.name)
    history_digest = digest(histories)
    config_digest = digest(load_json(config_path))
    prompt_digest = digest(system)

    write_json(
        out / "config_snapshot.json",
        {**config, "api_key_env_value": "[REDACTED_NOT_STORED]"},
    )
    write_json(
        out / "run_manifest.json",
        {
            "run_start_utc": run_start,
            "diagnostic": "measure provider-reported low-effort reasoning length",
            "formal_scoring": False,
            "formal_R1": False,
            "future_queries_loaded": False,
            "no_tools_inside_model_calls": True,
            "no_web_inside_model_calls": True,
            "provider_name": config["provider_name"],
            "provider_model_id": config["compressor"]["model"],
            "reasoning_effort": config["compressor"]["reasoning_effort"],
            "technical_output_ceiling_tokens": args.technical_budget,
            "ordinary_output_limit_applied": False,
            "wire_output_parameter": config["compressor"]["output_parameter"],
            "native_memory_target_tokens": 1024,
            "memory_definition": config["memory_definition"],
            "full_HF_tokenizer_revision": config["compressor"]["full_HF_tokenizer_revision"],
            "tokenizer": tokenizer.name,
            "GET_v1_models_sha256": config["models_endpoint_sha256"],
            "histories_path": str(histories_path.relative_to(ROOT)),
            "histories_sha256": history_digest,
            "prompt_path": str(prompt_path.relative_to(ROOT)),
            "prompt_sha256": prompt_digest,
            "config_sha256": config_digest,
            "sample_plan": {
                "calls": 10,
                "unique_histories": 8,
                "repeat_calls": 2,
                "repeat_history_ids": [histories[0]["history_id"], histories[1]["history_id"]],
            },
            "authorization": {
                "project_total_usd_cap": config["project_total_usd_cap"],
                "deepseek_v4_1_flash_model_cap": config["model_caps_usd"]["deepseek-v4.1-flash"],
                "auto_topup": False,
                "paid_fallback_provider": False,
            },
        },
    )

    rows_path = out / "calls.jsonl"
    provider = None
    before = None
    completed = 0
    failures = 0
    try:
        provider = Provider(config, tokenizer, out, mock=False)
        before = provider.ledger.summary()
        for index, (history, replicate, sample_kind) in enumerate(selected, start=1):
            history_id = history["history_id"]
            user = "BEGIN_MEMORY_DATA\n" + public_text(history) + "\nEND_MEMORY_DATA"
            request_metadata = {
                "phase": config["phase"],
                "diagnostic": "reasoning_length_only",
                "completion_budget_class": "unbounded_diagnostic",
                "max_completion_tokens": args.technical_budget,
                "reasoning_effort": config["compressor"]["reasoning_effort"],
                "memory_definition": "final_memory_body_only",
                "include_reasoning_tokens": False,
                "include_think_blocks": False,
                "fresh_request_each_stage": True,
                "reasoning_forwarded_to_next_stage": False,
                "no_future_query_visible_to_compressor": True,
                "native_memory_tokens": 1024,
                "ordinary_output_limit_applied": False,
                "experimental_path": [
                    "P0_REASONING_LENGTH_DIAGNOSTIC_V1",
                    history_id,
                    sample_kind,
                    replicate,
                ],
            }
            input_tokens = tokenizer.count(system + user)
            started = time.monotonic()
            base = {
                "call_index": index,
                "history_id": history_id,
                "sample_kind": sample_kind,
                "replicate": replicate,
                "status": "TECHNICAL_FAILURE",
                "run_start_utc": run_start,
                "provider_name": config["provider_name"],
                "provider_model_id": config["compressor"]["model"],
                "reasoning_effort": config["compressor"]["reasoning_effort"],
                "provider_output_budget_tokens": args.technical_budget,
                "max_completion_tokens": args.technical_budget,
                "ordinary_output_limit_applied": False,
                "native_memory_tokens": 1024,
                "full_HF_tokenizer_revision": config["compressor"]["full_HF_tokenizer_revision"],
                "GET_v1_models_sha256": config["models_endpoint_sha256"],
                "full_request_native_tokens_preflight": input_tokens,
                "provider_input_tokens": None,
                "provider_output_tokens": None,
                "reasoning_tokens": None,
                "reasoning_tokens_is_lower_bound": None,
                "cache_read_tokens": None,
                "visible_response_tokens": None,
                "final_visible_token_count": None,
                "finish_reason": None,
                "latency_seconds": None,
            }
            try:
                result = provider.call(
                    "compressor",
                    system,
                    user,
                    max_tokens=1024,
                    replicate=replicate,
                    namespace=[
                        "P0_REASONING_LENGTH_DIAGNOSTIC_V1",
                        history_id,
                        sample_kind,
                        replicate,
                    ],
                    request_metadata=request_metadata,
                    provider_output_budget=args.technical_budget,
                    visible_output_limit=args.technical_budget,
                )
                finish_reason = result.get("finish_reason")
                reasoning_tokens = usage_reasoning_tokens(result)
                base.update(
                    {
                        "status": "COMPLETED",
                        "model": result.get("model"),
                        "returned_model_field": result.get("returned_model_field"),
                        "provider_input_tokens": result.get("provider_input_tokens"),
                        "provider_output_tokens": result.get("provider_output_tokens"),
                        "cache_read_tokens": result.get("cache_read_tokens", 0),
                        "full_request_native_tokens": result.get("full_request_native_tokens"),
                        "visible_response_tokens": result.get("visible_response_tokens"),
                        "final_visible_token_count": result.get("visible_response_tokens"),
                        "reasoning_tokens": reasoning_tokens,
                        "reasoning_tokens_is_lower_bound": finish_reason == "length",
                        "finish_reason": finish_reason,
                        "request_key": result.get("request_key"),
                        "opencode_session_id": result.get("opencode_session_id"),
                        "reasoning_content_present": result.get("reasoning_content_present", False),
                        "usage": result.get("usage"),
                        "protocol_violation": "TECHNICAL_INVALID" if finish_reason == "length" else None,
                    }
                )
                completed += 1
            except Exception as exc:  # preserve all 10 diagnostics after one failure
                metadata = metadata_from_exception(exc)
                base.update(
                    {
                        "error": safe_error(exc),
                        "error_type": type(exc).__name__,
                        "returned_model_field": metadata.get("returned_model_field"),
                        "provider_input_tokens": metadata.get("provider_input_tokens"),
                        "provider_output_tokens": metadata.get("provider_output_tokens"),
                        "cache_read_tokens": metadata.get("cache_read_tokens"),
                        "full_request_native_tokens": metadata.get("full_request_native_tokens", input_tokens),
                        "finish_reason": metadata.get("finish_reason"),
                        "request_key": metadata.get("request_key"),
                        "opencode_session_id": metadata.get("opencode_session_id"),
                    }
                )
                failures += 1
            base["latency_seconds"] = time.monotonic() - started
            append_jsonl(rows_path, base)
            print(
                f"[{index}/10] {history_id} {sample_kind} status={base['status']} "
                f"reasoning_tokens={base.get('reasoning_tokens')} "
                f"provider_output_tokens={base.get('provider_output_tokens')} "
                f"finish_reason={base.get('finish_reason')}"
            , flush=True)
    finally:
        after = provider.ledger.summary() if provider is not None else None
        write_json(
            out / "cost_summary.json",
            {
                "run_start_utc": run_start,
                "provider_model_id": config["compressor"]["model"],
                "before": before,
                "after": after,
                "completed_calls": completed,
                "failed_calls": failures,
                "hard_project_cap_usd": config["project_total_usd_cap"],
                "hard_model_cap_usd": config["model_caps_usd"]["deepseek-v4.1-flash"],
            },
        )

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
