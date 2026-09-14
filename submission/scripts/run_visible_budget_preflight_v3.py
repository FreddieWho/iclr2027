#!/usr/bin/env python3
"""Run the frozen C1-only VISIBLE_BUDGET_PROTOCOL_V3 feasibility preflight."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import (  # noqa: E402
    Ledger,
    Provider,
    Tokenizer,
    append_jsonl,
    digest,
    load_json,
    load_model_config,
    public_text,
    read_jsonl,
    reported_reasoning_tokens,
    write_json,
)


def classify_compliance(finish_reason, body, visible_tokens, budget):
    if finish_reason != "stop" or not isinstance(body, str) or not body.strip():
        return "TECHNICAL_INVALID"
    if not isinstance(visible_tokens, int) or visible_tokens < 0:
        return "TECHNICAL_INVALID"
    if visible_tokens > int(budget):
        return "BUDGET_NONCOMPLIANT"
    return "COMPLIANT"


def summarize_candidates(rows, candidate_budgets, expected_history_ids, threshold, preferred_pair):
    expected_ids = list(expected_history_ids)
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("duplicate history IDs in frozen sample")
    seen = set()
    for row in rows:
        key = (row.get("history_id"), int(row.get("target_memory_tokens", -1)))
        if key in seen:
            raise ValueError(f"duplicate history-candidate cell: {key}")
        seen.add(key)

    candidate_summary = {}
    expected = set(expected_ids)
    for budget in candidate_budgets:
        budget = int(budget)
        items = [row for row in rows if int(row.get("target_memory_tokens", -1)) == budget]
        observed_ids = {row.get("history_id") for row in items}
        complete = observed_ids == expected and len(items) == len(expected_ids)
        if not complete:
            candidate_summary[str(budget)] = {
                "candidate_budget_native_tokens": budget,
                "planned_calls": len(expected_ids),
                "attempted_calls": len(items),
                "compliant_calls": sum(row.get("budget_compliance") == "COMPLIANT" for row in items),
                "denominator": len(expected_ids),
                "rate": None,
                "status": "INCOMPLETE",
                "missing_history_ids": sorted(expected - observed_ids),
                "unexpected_history_ids": sorted(observed_ids - expected),
            }
            continue
        compliant = sum(row.get("budget_compliance") == "COMPLIANT" for row in items)
        rate = compliant / len(expected_ids) if expected_ids else 0.0
        candidate_summary[str(budget)] = {
            "candidate_budget_native_tokens": budget,
            "planned_calls": len(expected_ids),
            "attempted_calls": len(items),
            "compliant_calls": compliant,
            "denominator": len(expected_ids),
            "rate": rate,
            "status": "PASS" if rate >= float(threshold) else "FAIL",
            "missing_history_ids": [],
            "unexpected_history_ids": [],
        }

    pair = [int(value) for value in preferred_pair]
    selected = pair if all(
        candidate_summary.get(str(budget), {}).get("status") == "PASS" for budget in pair
    ) else None
    return {
        "minimum_natural_compliance_rate": float(threshold),
        "candidates": candidate_summary,
        "selected_formal_budgets_native_tokens": selected,
    }


def load_credential(name):
    if os.environ.get(name):
        return os.environ[name]
    env_path = ROOT / ".env"
    if not env_path.is_file():
        raise RuntimeError("authorized credential is missing")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, separator, value = line.partition("=")
        if not separator or key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value:
            os.environ[name] = value
            return value
    raise RuntimeError("authorized credential is missing")


def fetch_models_catalog(config, output_dir, plan):
    role = config["compressor"]
    if plan.get("models_catalog_snapshot_path"):
        snapshot_path = ROOT / plan["models_catalog_snapshot_path"]
        response_path = ROOT / plan["models_catalog_response_path"]
        previous = load_json(snapshot_path)
        raw = response_path.read_bytes()
        response_sha = hashlib.sha256(raw).hexdigest()
        if response_sha != previous.get("response_sha256"):
            raise RuntimeError("frozen model catalog response hash mismatch")
        parsed = json.loads(raw)
        url = previous["endpoint"]
        snapshot = {
            **previous,
            "snapshot_reused": True,
            "source_snapshot_path": plan["models_catalog_snapshot_path"],
            "source_response_path": plan["models_catalog_response_path"],
        }
    else:
        import requests

        key = load_credential(role["api_key_env"])
        url = config.get("models_endpoint") or role["base_url"].rstrip("/") + "/models"
        response = requests.get(
            url,
            headers={"Authorization": "Bearer " + key, "User-Agent": role.get("user_agent", "iclr-memory-pilot/0.1")},
            timeout=(15, 60),
        )
        response.raise_for_status()
        raw = response.content
        parsed = response.json()
        response_sha = hashlib.sha256(raw).hexdigest()
        snapshot = {
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": response.status_code,
            "snapshot_reused": False,
        }
    entries = parsed.get("data", parsed.get("models", [])) if isinstance(parsed, dict) else parsed
    model_ids = sorted(
        item.get("id") or item.get("name")
        for item in entries
        if isinstance(item, dict) and (item.get("id") or item.get("name"))
    )
    (output_dir / "provider_models_response.json").write_bytes(raw)
    snapshot.update({
        "endpoint": url,
        "response_sha256": response_sha,
        "model_ids": model_ids,
        "c1_model_available": role["model"] in model_ids,
        "requested_r1_model_id": "glm-3-flash",
        "requested_r1_model_available": "glm-3-flash" in model_ids,
        "configured_r1_model_id": config.get("reader", {}).get("model"),
        "configured_r1_model_available": config.get("reader", {}).get("model") in model_ids,
    })
    write_json(output_dir / "provider_models_snapshot.json", snapshot)
    if not snapshot["c1_model_available"]:
        raise RuntimeError("configured C1 model is absent from the current provider catalog")
    return snapshot


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_error(exc):
    value = str(exc)
    return value[:1200]


def run(plan_path):
    plan_path = Path(plan_path)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan = load_json(plan_path)
    config_path = ROOT / plan["model_config"]
    config = load_model_config(config_path)
    data_path = ROOT / plan["histories_path"]
    output_dir = ROOT / plan["run_output_dir"]
    artifact_path = ROOT / plan["canonical_artifact"]
    output_dir.mkdir(parents=True, exist_ok=False)

    run_start = datetime.now(timezone.utc).isoformat()
    plan_sha = _sha256(plan_path)
    config_source_sha = _sha256(config_path)
    histories_sha = _sha256(data_path)
    histories = read_jsonl(data_path)
    sample = plan["sample_design"]
    generation = plan["history_generation"]
    inference = plan["inference"]
    accounting = plan["accounting"]
    budgets = [int(value) for value in sample["candidate_budgets_native_tokens"]]
    preferred = [int(value) for value in sample["preferred_formal_pair_native_tokens"]]
    if len(histories) != int(generation["n"]):
        raise RuntimeError("history count differs from the frozen sample design")
    history_ids = [item.get("history_id") for item in histories]
    if not all(isinstance(value, str) and value for value in history_ids) or len(history_ids) != len(set(history_ids)):
        raise RuntimeError("history IDs must be nonempty and unique")
    if budgets != [int(value) for value in config["visible_budget_protocol_v3"]["preflight_budget_feasibility"]["candidate_budgets"]]:
        raise RuntimeError("frozen candidate budgets differ from V3 protocol")
    if int(config["compressor"]["provider_output_budget_tokens"]) != int(inference["provider_output_budget_tokens"]):
        raise RuntimeError("provider output budget differs from the frozen V3 sample design")
    if config["compressor"]["model"] != inference["provider_model_id"]:
        raise RuntimeError("C1 model differs from the frozen V3 sample design")
    if config["compressor"]["reasoning_effort"] != inference["reasoning_effort"]:
        raise RuntimeError("C1 reasoning effort differs from the frozen V3 sample design")

    tokenizer = Tokenizer(config["compressor"]["tokenizer"])
    prompt_path = ROOT / config["compressor_prompt"]
    prompt_template = prompt_path.read_text(encoding="utf-8")
    user_text = {item["history_id"]: "BEGIN_MEMORY_DATA\n" + public_text(item) + "\nEND_MEMORY_DATA" for item in histories}
    candidate_orders = {}
    request_specs = []
    order_rng = random.Random(int(sample.get("candidate_order_seed", generation["seed"])))
    for history in histories:
        order = list(budgets)
        order_rng.shuffle(order)
        candidate_orders[history["history_id"]] = order
        for budget in order:
            system = prompt_template.format(budget=budget, tokenizer=tokenizer.name)
            full_request_tokens = tokenizer.count(system + user_text[history["history_id"]])
            if full_request_tokens > int(config["authorized_max_input_tokens"]):
                raise RuntimeError("full request exceeds the authorized native input-token limit")
            request_specs.append({
                "history": history,
                "budget": budget,
                "system": system,
                "user": user_text[history["history_id"]],
                "full_request_native_tokens_preflight": full_request_tokens,
                "history_native_tokens": tokenizer.count(public_text(history)),
            })
    expected_calls = int(generation["n"]) * len(budgets)
    if len(request_specs) != int(sample["scheduled_calls"]) or len(request_specs) != expected_calls:
        raise RuntimeError("scheduled call count differs from the frozen sample design")

    try:
        catalog = fetch_models_catalog(config, output_dir, plan)
    except Exception as exc:
        failure = {
            "run_start_utc": run_start,
            "status": "BLOCKED_MODELS_CATALOG",
            "error": _safe_error(exc),
            "model_generation_requests_sent": 0,
            "formal_p0": False,
            "scientific_scoring": False,
        }
        write_json(output_dir / "setup_failure.json", failure)
        return 2

    config["run_start_utc"] = run_start
    config["models_endpoint_sha256"] = catalog["response_sha256"]
    config["compressor"]["models_endpoint_sha256"] = catalog["response_sha256"]
    config["usd_cap"] = float(accounting["p0_and_project_usd_cap"])
    config["phase_usd_cap"] = float(accounting["p0_and_project_usd_cap"])
    config["project_total_usd_cap"] = float(accounting["p0_and_project_usd_cap"])
    config["model_caps_usd"] = {
        **config.get("model_caps_usd", {}),
        "deepseek-v4.1-flash": float(accounting["deepseek_v4_1_flash_cap_usd"]),
        "glm-5.3-flash": float(accounting["glm_5_3_flash_cap_usd"]),
    }
    config["auto_topup"] = False
    config["paid_fallback_provider"] = False
    config["transport_attempts"] = int(sample["transport_attempts"])
    config["session_policy"] = "isolated_between_history_budget_cells"
    config["opencode_session_id"] = "iclr-memory-pilot/p0-visible-budget-v3/20260913"
    config["ledger_path"] = str((output_dir / "cost_ledger.sqlite").resolve())
    compressor = config["compressor"]
    pricing = plan["pricing_snapshot"]
    compressor["input_usd_per_million"] = float(pricing["input_usd_per_million"])
    compressor["output_usd_per_million"] = float(pricing["output_usd_per_million"])
    compressor["cache_read_usd_per_million"] = float(pricing["cache_read_usd_per_million"])
    compressor["price_verified_date"] = pricing["verified_date"]
    compressor["price_source_url"] = pricing["source_url"]
    compressor["price_basis"] = pricing["basis"]

    prior_cost = float(accounting["prior_project_actual_plus_uncertain_usd"])
    ledger = Ledger(config["ledger_path"], config)
    rid = ledger.reserve("prior_project_actual_plus_uncertain_carry_forward", prior_cost, 0, 0)
    ledger.settle(rid, prior_cost, 0, 0)

    input_price = float(compressor["input_usd_per_million"])
    output_price = float(compressor["output_usd_per_million"])
    input_overhead = int(compressor.get("input_reservation_overhead_tokens", 256))
    preflight_reserved_upper = sum(
        ((spec["full_request_native_tokens_preflight"] + input_overhead) * input_price
         + int(inference["provider_output_budget_tokens"]) * output_price) / 1_000_000
        for spec in request_specs
    )
    formal_c1_calls = int(plan.get("additional_formal_p0_c1_calls_upper", 16))
    worst_request = max(spec["full_request_native_tokens_preflight"] for spec in request_specs) + input_overhead
    formal_c1_reserved_upper = formal_c1_calls * (
        worst_request * input_price + int(inference["provider_output_budget_tokens"]) * output_price
    ) / 1_000_000
    c1_worst_cumulative = prior_cost + preflight_reserved_upper + formal_c1_reserved_upper
    if c1_worst_cumulative > float(accounting["deepseek_v4_1_flash_cap_usd"]):
        raise RuntimeError("P0 upper-bound reservation exceeds the amended C1 model cap")
    if prior_cost + preflight_reserved_upper > float(config["usd_cap"]):
        raise RuntimeError("P0 preflight upper-bound reservation exceeds the authorized project cap")

    config["visible_budget_protocol_v3"]["preflight_budget_feasibility"].update({
        "sample_design_status": "FROZEN",
        "sample_size_per_candidate": int(sample["sample_size_per_candidate"]),
        "preflight_histories_path": plan["histories_path"],
        "preflight_histories_sha256": histories_sha,
        "preflight_status": "RUNNING",
        "selected_formal_budgets_native_tokens": None,
        "formal_budgets_frozen": False,
    })
    write_json(output_dir / "preflight_plan.json", plan)
    write_json(output_dir / "resolved_config_before_calls.json", config)
    manifest = {
        "run_start_utc": run_start,
        "status": "RUNNING",
        "phase": "P0_VISIBLE_BUDGET_PREFLIGHT_V3",
        "formal_p0": False,
        "formal_scoring": False,
        "provider_name": config["provider_name"],
        "provider_model_id": compressor["model"],
        "reasoning_effort": compressor["reasoning_effort"],
        "provider_output_budget_tokens": compressor["provider_output_budget_tokens"],
        "pricing_snapshot": pricing,
        "GET_v1_models_sha256": catalog["response_sha256"],
        "models_endpoint": catalog["endpoint"],
        "models_catalog_snapshot_reused": catalog.get("snapshot_reused", False),
        "full_HF_tokenizer_revision": compressor["full_HF_tokenizer_revision"],
        "tokenizer_spec": compressor["tokenizer"],
        "tokenizer_name": tokenizer.name,
        "history_generation": generation,
        "histories_path": plan["histories_path"],
        "histories_sha256": histories_sha,
        "history_ids": history_ids,
        "query_file_opened": False,
        "no_future_query_visible_to_compressor": True,
        "no_tools_inside_model_calls": True,
        "no_web_inside_model_calls": True,
        "candidate_budgets_native_tokens": budgets,
        "preferred_formal_pair_native_tokens": preferred,
        "minimum_natural_compliance_rate": float(sample["minimum_natural_compliance_rate"]),
        "sample_size_per_candidate": int(sample["sample_size_per_candidate"]),
        "scheduled_call_count": len(request_specs),
        "candidate_order_by_history": candidate_orders,
        "one_call_per_history_and_candidate": True,
        "transport_attempts": int(sample["transport_attempts"]),
        "rejection_sampling": False,
        "model_config_path": plan["model_config"],
        "model_config_sha256": config_source_sha,
        "resolved_config_sha256": digest(config),
        "prompt_path": str(prompt_path.relative_to(ROOT)),
        "prompt_sha256": _sha256(prompt_path),
        "plan_path": str(plan_path.relative_to(ROOT)),
        "plan_sha256": plan_sha,
        "accounting": {
            **accounting,
            "preflight_reserved_upper_usd": round(preflight_reserved_upper, 8),
            "max_formal_p0_c1_calls_reserved": formal_c1_calls,
            "formal_p0_c1_reserved_upper_usd": round(formal_c1_reserved_upper, 8),
            "c1_worst_cumulative_upper_usd": round(c1_worst_cumulative, 8),
            "ledger_path": str(Path(config["ledger_path"]).relative_to(ROOT)),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    load_credential(compressor["api_key_env"])
    provider = Provider(config, tokenizer, output_dir, mock=False)
    rows_path = output_dir / "calls.jsonl"
    rows = []
    hard_block = None
    for spec in request_specs:
        history = spec["history"]
        budget = spec["budget"]
        started = time.monotonic()
        row = {
            "run_start_utc": run_start,
            "history_id": history["history_id"],
            "history_sha256": digest(history),
            "target_memory_tokens": budget,
            "provider_name": config["provider_name"],
            "provider_model_id": compressor["model"],
            "reasoning_effort": compressor["reasoning_effort"],
            "provider_output_budget_tokens": compressor["provider_output_budget_tokens"],
            "full_HF_tokenizer_revision": compressor["full_HF_tokenizer_revision"],
            "GET_v1_models_sha256": catalog["response_sha256"],
            "native_history_tokens": spec["history_native_tokens"],
            "full_request_native_tokens_preflight": spec["full_request_native_tokens_preflight"],
            "query_visible_to_compressor": False,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "reasoning_tokens": None,
            "reasoning_token_source": "UNAVAILABLE",
            "cache_read_tokens": None,
            "finish_reason": None,
            "visible_native_tokens": None,
            "budget_compliance": "API_ERROR",
            "status": "API_ERROR",
        }
        try:
            result = provider.call(
                "compressor",
                spec["system"],
                spec["user"],
                max_tokens=budget,
                replicate=0,
                namespace=["P0_VISIBLE_BUDGET_PREFLIGHT_V3", history["history_id"], budget],
                request_metadata={
                    "phase": "P0_VISIBLE_BUDGET_PREFLIGHT_V3",
                    "diagnostic": "one_scheduled_call_per_history_candidate",
                    "completion_budget_class": "technical_preflight",
                    "reasoning_effort": compressor["reasoning_effort"],
                    "native_memory_tokens": spec["history_native_tokens"],
                    "target_memory_tokens": budget,
                    "memory_definition": "final_memory_body_only",
                    "include_reasoning_tokens": False,
                    "include_think_blocks": False,
                    "no_future_query_visible_to_compressor": True,
                    "experimental_path": ["P0_VISIBLE_BUDGET_PREFLIGHT_V3", history["history_id"], budget],
                },
                provider_output_budget=int(inference["provider_output_budget_tokens"]),
                visible_output_limit=budget,
            )
            visible = tokenizer.count(result.get("text", ""))
            status = classify_compliance(result.get("finish_reason"), result.get("text"), visible, budget)
            reasoning = reported_reasoning_tokens(result.get("usage"))
            reasoning_source = "PROVIDER_REPORTED" if reasoning is not None else "ESTIMATED_COMPLETION_MINUS_VISIBLE_NATIVE"
            if reasoning is None and result.get("provider_output_tokens") is not None:
                reasoning = max(0, int(result["provider_output_tokens"]) - visible)
            row.update({
                "status": status,
                "budget_compliance": status,
                "model": result.get("model"),
                "returned_model_field": result.get("returned_model_field"),
                "full_request_native_tokens": result.get("full_request_native_tokens"),
                "provider_input_tokens": result.get("provider_input_tokens"),
                "provider_output_tokens": result.get("provider_output_tokens"),
                "total_completion_tokens": result.get("total_completion_tokens", result.get("provider_output_tokens")),
                "reasoning_tokens": reasoning,
                "reasoning_token_source": reasoning_source,
                "cache_read_tokens": result.get("cache_read_tokens"),
                "finish_reason": result.get("finish_reason"),
                "visible_native_tokens": visible,
                "actual_compression_rate": (1.0 - visible / spec["history_native_tokens"]) if spec["history_native_tokens"] else None,
                "request_key": result.get("request_key"),
                "opencode_session_id": result.get("opencode_session_id"),
                "usage": result.get("usage"),
                "cost_usd_or_conservative": result.get("cost_usd_or_conservative"),
                "memory_body": result.get("text", ""),
                "memory_body_sha256": hashlib.sha256(result.get("text", "").encode("utf-8")).hexdigest(),
            })
        except Exception as exc:
            metadata = getattr(exc, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            row.update({
                "error_type": type(exc).__name__,
                "error": _safe_error(exc),
                "returned_model_field": metadata.get("returned_model_field"),
                "provider_input_tokens": metadata.get("provider_input_tokens"),
                "provider_output_tokens": metadata.get("provider_output_tokens"),
                "cache_read_tokens": metadata.get("cache_read_tokens"),
                "finish_reason": metadata.get("finish_reason"),
                "request_key": metadata.get("request_key"),
            })
            message = str(exc)
            if any(marker in message for marker in (
                "BUDGET_BLOCK", "OUTPUT_CAP_BLOCK", "CONTEXT_BLOCK", "CONFIG_BLOCK",
                "HTTP_400", "HTTP_401", "HTTP_403", "HTTP_404",
            )):
                hard_block = type(exc).__name__ + ": " + message[:300]
        row["latency_seconds"] = time.monotonic() - started
        append_jsonl(rows_path, row)
        rows.append(row)
        print(
            f"history={history['history_id']} target={budget} status={row['status']} "
            f"visible={row.get('visible_native_tokens')} finish={row.get('finish_reason')}",
            flush=True,
        )
        if hard_block:
            break

    summary = summarize_candidates(
        rows,
        budgets,
        history_ids,
        float(sample["minimum_natural_compliance_rate"]),
        preferred,
    )
    complete = len(rows) == len(request_specs)
    selected = summary["selected_formal_budgets_native_tokens"] if complete else None
    if not complete:
        selected = None
        summary["selected_formal_budgets_native_tokens"] = None
    if complete and selected:
        preflight_status = "PASS"
    elif complete:
        preflight_status = "NO_PREFERRED_PAIR_PASSED"
    else:
        preflight_status = "INCOMPLETE"
    final_summary = {
        "artifact_version": "VISIBLE_BUDGET_PREFLIGHT_V3_V1",
        "run_start_utc": run_start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": preflight_status,
        "formal_p0": False,
        "formal_scoring": False,
        "provider_name": config["provider_name"],
        "provider_model_id": compressor["model"],
        "returned_model_fields": sorted({row.get("returned_model_field") for row in rows if row.get("returned_model_field")}),
        "reasoning_effort": compressor["reasoning_effort"],
        "provider_output_budget_tokens": compressor["provider_output_budget_tokens"],
        "GET_v1_models_sha256": catalog["response_sha256"],
        "full_HF_tokenizer_revision": compressor["full_HF_tokenizer_revision"],
        "histories_path": plan["histories_path"],
        "histories_sha256": histories_sha,
        "history_ids": history_ids,
        "sample_size_per_candidate": len(history_ids),
        "candidate_budgets_native_tokens": budgets,
        "minimum_natural_compliance_rate": float(sample["minimum_natural_compliance_rate"]),
        "scheduled_calls": len(request_specs),
        "attempted_calls": len(rows),
        "transport_attempts_per_cell": 1,
        "retry_or_rejection_sampling": False,
        "no_future_query_visible_to_compressor": True,
        "no_tools_inside_model_calls": True,
        "no_web_inside_model_calls": True,
        "summary": summary,
        "selected_formal_budgets_native_tokens": selected,
        "formal_budgets_frozen": bool(selected),
        "hard_block": hard_block,
        "scientific_score_rows": 0,
        "accounting": ledger.summary(),
        "run_output_dir": str(output_dir.relative_to(ROOT)),
        "calls_path": str((output_dir / "calls.jsonl").relative_to(ROOT)),
        "run_manifest_path": str((output_dir / "run_manifest.json").relative_to(ROOT)),
        "canonical_artifact": str(artifact_path.relative_to(ROOT)),
    }
    if selected:
        final_summary["frozen_config_path"] = str((output_dir / "frozen_formal_config.json").relative_to(ROOT))
    write_json(output_dir / "summary.json", final_summary)
    write_json(artifact_path, final_summary)
    artifact_hash = _sha256(artifact_path)

    if selected:
        frozen = copy.deepcopy(config)
        feasibility = frozen["visible_budget_protocol_v3"]["preflight_budget_feasibility"]
        feasibility.update({
            "sample_design_status": "FROZEN",
            "sample_size_per_candidate": len(history_ids),
            "preflight_histories_path": plan["histories_path"],
            "preflight_histories_sha256": histories_sha,
            "observed_natural_compliance_rates": {
                budget: value["rate"] for budget, value in summary["candidates"].items()
            },
            "preflight_status": "PASS",
            "preflight_artifact": str(artifact_path.relative_to(ROOT)),
            "preflight_artifact_sha256": artifact_hash,
            "selected_formal_budgets_native_tokens": selected,
            "formal_budgets_frozen": True,
        })
        frozen["final_memory_budgets_native_tokens"] = selected
        frozen["compressor"]["final_memory_calls"]["target_memory_tokens"] = selected
        frozen["reasoning_budget_amendment"]["C1_compressor"]["final_memory_calls"]["target_memory_tokens"] = selected
        frozen["active_status"] = "PREFLIGHT_PASS_FORMAL_P0_READY"
        write_json(output_dir / "frozen_formal_config.json", frozen)
    return 0 if complete and selected else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="configs/visible_budget_preflight_v3_plan_20260913.json")
    args = parser.parse_args()
    return run(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
