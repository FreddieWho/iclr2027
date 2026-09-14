#!/usr/bin/env python3
"""Live C1-only protocol probe for the frozen V3 compression prompt."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import Provider, Tokenizer, append_jsonl, digest, load_json, public_text, read_jsonl, write_json  # noqa: E402


def reasoning_tokens(result: dict) -> int | None:
    usage = result.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens", usage.get("reasoning_tokens"))
    return int(value) if isinstance(value, (int, float)) else None


def error_metadata(exc: BaseException) -> dict:
    value = getattr(exc, "metadata", {})
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models_opencode_go_p0_reasoning_v3_naturalstop.json")
    parser.add_argument("--histories", default="work/p0_calibration_data_glm_20260911/histories.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-id", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    histories_path = ROOT / args.histories
    config = load_json(config_path)
    prompt_path = ROOT / config["compressor_prompt"]
    prompt_template = prompt_path.read_text(encoding="utf-8")
    histories = read_jsonl(histories_path)
    if not histories:
        raise SystemExit("No histories available for the C1-only technical probe")
    if args.history_id:
        matches = [item for item in histories if item.get("history_id") == args.history_id]
        if not matches:
            raise SystemExit(f"history_id not found: {args.history_id}")
        history = matches[0]
    else:
        history = histories[0]

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    run_start = datetime.now(timezone.utc).isoformat()
    config["run_start_utc"] = run_start
    tokenizer = Tokenizer(config["compressor"]["tokenizer"])
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    histories_sha = hashlib.sha256(histories_path.read_bytes()).hexdigest()
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    write_json(
        out / "run_manifest.json",
        {
            "run_start_utc": run_start,
            "purpose": "C1-only V3 prompt and natural-stop technical validation",
            "formal_p0": False,
            "formal_scoring": False,
            "reader_called": False,
            "queries_loaded": False,
            "history_id": history["history_id"],
            "history_sha256": digest(history),
            "histories_file_sha256": histories_sha,
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": config_sha,
            "prompt_path": str(prompt_path.relative_to(ROOT)),
            "prompt_sha256": prompt_sha,
            "provider_name": config["provider_name"],
            "provider_model_id": config["compressor"]["model"],
            "reasoning_effort": config["compressor"]["reasoning_effort"],
            "provider_output_budget_tokens": config["compressor"]["provider_output_budget_tokens"],
            "full_HF_tokenizer_revision": config["compressor"]["full_HF_tokenizer_revision"],
            "GET_v1_models_sha256": config["models_endpoint_sha256"],
            "targets_native_tokens": config["final_memory_budgets_native_tokens"],
            "no_future_query_visible_to_compressor": True,
            "no_tools_inside_model_calls": True,
            "no_web_inside_model_calls": True,
            "cost_cap_policy": {
                "project_hard_cap_usd": config["project_total_usd_cap"],
                "deepseek_v4_1_flash_hard_cap_usd": config["model_caps_usd"]["deepseek-v4.1-flash"],
                "auto_topup": False,
                "paid_fallback_provider": False,
            },
        },
    )

    provider = Provider(config, tokenizer, out, mock=False)
    ledger_before = provider.ledger.summary()
    user = "BEGIN_MEMORY_DATA\n" + public_text(history) + "\nEND_MEMORY_DATA"
    rows_path = out / "calls.jsonl"
    rows = []
    for target in config["final_memory_budgets_native_tokens"]:
        system = prompt_template.format(budget=int(target), tokenizer=tokenizer.name)
        full_request_tokens = tokenizer.count(system + user)
        started = time.monotonic()
        row = {
            "history_id": history["history_id"],
            "target_memory_tokens": int(target),
            "run_start_utc": run_start,
            "provider_name": config["provider_name"],
            "provider_model_id": config["compressor"]["model"],
            "reasoning_effort": config["compressor"]["reasoning_effort"],
            "provider_output_budget_tokens": config["compressor"]["provider_output_budget_tokens"],
            "max_completion_tokens": config["compressor"]["provider_output_budget_tokens"],
            "full_HF_tokenizer_revision": config["compressor"]["full_HF_tokenizer_revision"],
            "GET_v1_models_sha256": config["models_endpoint_sha256"],
            "full_request_native_tokens_preflight": full_request_tokens,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "reasoning_tokens": None,
            "cache_read_tokens": None,
            "native_memory_tokens": None,
            "visible_response_tokens": None,
            "finish_reason": None,
            "request_key": None,
            "status": "TECHNICAL_FAILURE",
        }
        try:
            result = provider.call(
                "compressor",
                system,
                user,
                max_tokens=min(int(target) * 2, config["compressor"]["max_visible_output_tokens"]),
                replicate=0,
                namespace=["P0_REASONING_V3_C1_PROMPT_PROBE", history["history_id"], int(target)],
                request_metadata={
                    "phase": config["phase"],
                    "diagnostic": "C1_prompt_natural_stop_and_native_memory_ceiling",
                    "completion_budget_class": "technical_validation",
                    "max_completion_tokens": config["compressor"]["provider_output_budget_tokens"],
                    "reasoning_effort": config["compressor"]["reasoning_effort"],
                    "memory_definition": "final_memory_body_only",
                    "include_reasoning_tokens": False,
                    "include_think_blocks": False,
                    "fresh_request_each_stage": True,
                    "reasoning_forwarded_to_next_stage": False,
                    "native_memory_tokens": tokenizer.count(public_text(history)),
                    "no_future_query_visible_to_compressor": True,
                    "experimental_path": ["P0_REASONING_V3_C1_PROMPT_PROBE", history["history_id"], int(target)],
                },
                provider_output_budget=config["compressor"]["provider_output_budget_tokens"],
                visible_output_limit=int(target),
            )
            visible_tokens = int(result.get("visible_response_tokens", tokenizer.count(result.get("text", ""))))
            finish = result.get("finish_reason")
            if finish != "stop":
                status = "TECHNICAL_INVALID_FINISH_REASON"
            elif not result.get("text", "").strip():
                status = "TECHNICAL_INVALID_EMPTY_MEMORY"
            elif visible_tokens > int(target):
                status = "TECHNICAL_INVALID_MEMORY_OVER_BUDGET"
            else:
                status = "TECHNICAL_PASS"
            row.update(
                {
                    "status": status,
                    "model": result.get("model"),
                    "returned_model_field": result.get("returned_model_field"),
                    "provider_input_tokens": result.get("provider_input_tokens"),
                    "provider_output_tokens": result.get("provider_output_tokens"),
                    "reasoning_tokens": reasoning_tokens(result),
                    "cache_read_tokens": result.get("cache_read_tokens", 0),
                    "full_request_native_tokens": result.get("full_request_native_tokens"),
                    "native_memory_tokens": visible_tokens,
                    "visible_response_tokens": visible_tokens,
                    "finish_reason": finish,
                    "request_key": result.get("request_key"),
                    "opencode_session_id": result.get("opencode_session_id"),
                    "usage": result.get("usage"),
                    "cost_usd_or_conservative": result.get("cost_usd_or_conservative"),
                    "reasoning_content_present": result.get("reasoning_content_present", False),
                }
            )
        except Exception as exc:
            metadata = error_metadata(exc)
            row.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1200],
                    "returned_model_field": metadata.get("returned_model_field"),
                    "provider_input_tokens": metadata.get("provider_input_tokens"),
                    "provider_output_tokens": metadata.get("provider_output_tokens"),
                    "cache_read_tokens": metadata.get("cache_read_tokens"),
                    "finish_reason": metadata.get("finish_reason"),
                    "request_key": metadata.get("request_key"),
                }
            )
        row["latency_seconds"] = time.monotonic() - started
        append_jsonl(rows_path, row)
        rows.append(row)
        print(
            f"target={target} status={row['status']} native_memory_tokens={row.get('native_memory_tokens')} "
            f"reasoning_tokens={row.get('reasoning_tokens')} finish_reason={row.get('finish_reason')}",
            flush=True,
        )

    write_json(
        out / "summary.json",
        {
            "run_start_utc": run_start,
            "status": "TECHNICAL_PASS" if all(row["status"] == "TECHNICAL_PASS" for row in rows) else "TECHNICAL_BLOCKED",
            "formal_p0": False,
            "formal_scoring": False,
            "reader_called": False,
            "provider_model_id": config["compressor"]["model"],
            "reasoning_effort": config["compressor"]["reasoning_effort"],
            "provider_output_budget_tokens": config["compressor"]["provider_output_budget_tokens"],
            "rows": rows,
            "ledger_before": ledger_before,
            "ledger_after": provider.ledger.summary(),
            "hard_project_cap_usd": config["project_total_usd_cap"],
            "hard_model_cap_usd": config["model_caps_usd"]["deepseek-v4.1-flash"],
        },
    )
    return 0 if all(row["status"] == "TECHNICAL_PASS" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
