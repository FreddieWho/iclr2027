import contextlib
import hashlib
import io
import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_visible_budget_preflight_v3 import classify_compliance, summarize_candidates


class VisibleBudgetPreflightTests(unittest.TestCase):
    def test_accepts_only_nonempty_natural_stop_within_native_budget(self):
        self.assertEqual(classify_compliance("stop", "memory", 9, 10), "COMPLIANT")
        self.assertEqual(classify_compliance("length", "memory", 9, 10), "TECHNICAL_INVALID")
        self.assertEqual(classify_compliance("stop", "", 0, 10), "TECHNICAL_INVALID")
        self.assertEqual(classify_compliance("stop", "memory", 11, 10), "BUDGET_NONCOMPLIANT")

    def test_provider_errors_count_as_failures_but_missing_cells_are_incomplete(self):
        ids = [f"h{i}" for i in range(8)]
        rows = [
            {"history_id": history_id, "target_memory_tokens": 2048,
             "budget_compliance": "COMPLIANT" if i < 7 else "API_ERROR"}
            for i, history_id in enumerate(ids)
        ]
        summary = summarize_candidates(rows, [1024, 2048, 4096], ids, 0.90, [2048, 4096])
        self.assertEqual(summary["candidates"]["2048"]["denominator"], 8)
        self.assertEqual(summary["candidates"]["2048"]["compliant_calls"], 7)
        self.assertEqual(summary["candidates"]["2048"]["rate"], 0.875)
        self.assertEqual(summary["candidates"]["1024"]["status"], "INCOMPLETE")
        self.assertIsNone(summary["selected_formal_budgets_native_tokens"])

    def test_preferred_pair_freezes_only_when_both_candidates_meet_threshold(self):
        ids = [f"h{i}" for i in range(8)]
        rows = [
            {"history_id": history_id, "target_memory_tokens": budget,
             "budget_compliance": "COMPLIANT"}
            for history_id in ids
            for budget in (2048, 4096)
        ]
        summary = summarize_candidates(rows, [2048, 4096], ids, 0.90, [2048, 4096])
        self.assertEqual(summary["selected_formal_budgets_native_tokens"], [2048, 4096])

        rows[-1]["budget_compliance"] = "BUDGET_NONCOMPLIANT"
        summary = summarize_candidates(rows, [2048, 4096], ids, 0.90, [2048, 4096])
        self.assertIsNone(summary["selected_formal_budgets_native_tokens"])

    def test_duplicate_history_candidate_cell_is_rejected(self):
        rows = [
            {"history_id": "h1", "target_memory_tokens": 2048,
             "budget_compliance": "COMPLIANT"},
            {"history_id": "h1", "target_memory_tokens": 2048,
             "budget_compliance": "COMPLIANT"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_candidates(rows, [2048], ["h1"], 0.90, [2048])

    def test_frozen_preflight_runner_writes_receipt_without_network(self):
        import run_visible_budget_preflight_v3 as preflight
        from core import Ledger, append_jsonl, load_json, write_json
        from run_pilot import validate_visible_budget_protocol

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "work") as temp_dir:
            temp = Path(temp_dir)
            histories_path = temp / "histories.jsonl"
            for i in range(8):
                append_jsonl(histories_path, {"history_id": f"p0-test-{i}", "text": f"synthetic fact {i}"})
            plan = {
                "model_config": "configs/models_opencode_go_p0_visible_budget_v3.json",
                "histories_path": str(histories_path),
                "run_output_dir": str(temp / "run"),
                "canonical_artifact": str(temp / "artifact.json"),
                "history_generation": {
                    "n": 8,
                    "seed": 1,
                    "requested_native_tokens": 32768,
                    "tokenizer": "test",
                    "tokenizer_revision": "test-rev",
                },
                "sample_design": {
                    "sample_size_per_candidate": 8,
                    "candidate_budgets_native_tokens": [1024, 1536, 2048, 3072, 4096],
                    "minimum_natural_compliance_rate": 0.90,
                    "preferred_formal_pair_native_tokens": [2048, 4096],
                    "scheduled_calls": 40,
                    "transport_attempts": 1,
                },
                "inference": {
                    "provider_model_id": "deepseek-v4.1-flash",
                    "reasoning_effort": "low",
                    "provider_output_budget_tokens": 65536,
                },
                "pricing_snapshot": {
                    "verified_date": "2026-09-13",
                    "source_url": "https://dev.opencode.ai/docs/go/",
                    "basis": "test",
                    "input_usd_per_million": 0.3,
                    "output_usd_per_million": 1.2,
                    "cache_read_usd_per_million": 0.006,
                },
                "accounting": {
                    "prior_project_actual_plus_uncertain_usd": 0.001,
                    "p0_and_project_usd_cap": 18.0,
                    "deepseek_v4_1_flash_cap_usd": 8.0,
                    "glm_5_3_flash_cap_usd": 15.0,
                },
            }
            plan_path = temp / "plan.json"
            write_json(plan_path, plan)
            seen = []

            def fake_catalog(config, output_dir, frozen_plan):
                return {
                    "response_sha256": "a" * 64,
                    "endpoint": "https://example.invalid/models",
                    "snapshot_reused": True,
                }

            class FakeProvider:
                def __init__(self, config, tokenizer, output_dir, mock=False):
                    self.config = config
                    self.tokenizer = tokenizer
                    self.ledger = Ledger(config["ledger_path"], config)

                def call(self, role, system, user, max_tokens, replicate, **kwargs):
                    request_metadata = kwargs["request_metadata"]
                    seen.append(tuple(request_metadata["experimental_path"]))
                    full_input = self.tokenizer.count(system + user)
                    rid = self.ledger.reserve(f"fake-{len(seen)}", 0.0001, full_input, 8)
                    self.ledger.settle(rid, 0.0001, full_input, 8)
                    return {
                        "text": "memory",
                        "finish_reason": "stop",
                        "model": self.config["compressor"]["model"],
                        "returned_model_field": self.config["compressor"]["model"],
                        "provider_input_tokens": full_input,
                        "provider_output_tokens": 8,
                        "total_completion_tokens": 8,
                        "reasoning_tokens": 1,
                        "cache_read_tokens": 0,
                        "usage": {"prompt_tokens": full_input, "completion_tokens": 8, "reasoning_tokens": 1},
                        "request_key": f"fake-request-{len(seen)}",
                        "opencode_session_id": f"fake-session-{len(seen)}",
                        "cost_usd_or_conservative": 0.0001,
                    }

            with patch.object(preflight, "fetch_models_catalog", fake_catalog), \
                    patch.object(preflight, "Provider", FakeProvider), \
                    patch.object(preflight, "load_credential", return_value="test-key"), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = preflight.run(plan_path)

            self.assertEqual(result, 0)
            self.assertEqual(len(seen), 40)
            self.assertEqual(len(set(seen)), 40)
            frozen = load_json(temp / "run/frozen_formal_config.json")
            validate_visible_budget_protocol(frozen)
            self.assertEqual(frozen["final_memory_budgets_native_tokens"], [2048, 4096])

    def test_reuses_only_hash_verified_frozen_models_catalog(self):
        import run_visible_budget_preflight_v3 as preflight
        from core import write_json

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "work") as temp_dir:
            temp = Path(temp_dir)
            response_path = temp / "models.json"
            snapshot_path = temp / "snapshot.json"
            raw = json.dumps({"data": [{"id": "deepseek-v4.1-flash"}, {"id": "glm-5.3-flash"}]}).encode()
            response_path.write_bytes(raw)
            write_json(snapshot_path, {
                "observed_utc": "2026-09-13T00:00:00Z",
                "endpoint": "https://example.invalid/models",
                "http_status": 200,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
            })
            output = temp / "run"
            output.mkdir()
            config = {
                "compressor": {"model": "deepseek-v4.1-flash"},
                "reader": {"model": "glm-5.3-flash"},
            }
            plan = {
                "models_catalog_snapshot_path": str(snapshot_path),
                "models_catalog_response_path": str(response_path),
            }
            result = preflight.fetch_models_catalog(config, output, plan)
            self.assertTrue(result["snapshot_reused"])
            self.assertTrue(result["c1_model_available"])
            self.assertEqual(result["response_sha256"], hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
