# V4 Fast Route Decision — Preflight

> **HISTORICAL — continuation 前快照，已被取代。** `NO_GO_TECHNICAL` 为当时技术停机状态；当前决策为 `INCONCLUSIVE_NO_GO_SIGNAL`（`reports/V4_ROUTE_DECISION.json`）。

**Current state:** The original B=4,096 P0 and the one permitted B=8,192 adjustment both failed the C1 technical gate. The amended run produced 3/6 natural-stop final memories and only 1/6 in `[4,096, 16,384]`. R1 calibration was skipped; P1/P2/R2 were not run. Current route decision: `NO_GO_TECHNICAL`; no scientific route comparison was performed. This package selects a research route; it is not a strict equal-native-token endpoint proof.

## Frozen scope

- C1: OpenCode Go `deepseek-v4.1-flash`; `thinking.type=disabled`; `reasoning_effort` omitted; temperature `0`, `top_p=0.95`; the single predeclared adjustment is now frozen at final target 8,192 and intermediate target 16,384 DeepSeek-native tokens. Provider completion budgets are 16,384 final / 24,576 intermediate. Only `finish_reason=stop` is accepted. No post-hoc truncation or size-based regeneration.
- R1: `glm-5.3-flash`, initial `reasoning_effort=low`, 12,000 provider completion budget, visible response ≤512 GLM-native tokens. Promote once to high only if oracle accuracy is below 0.90; then use high for every formal R1 condition. R2 is `deepseek-v4.1-flash` and will read only frozen memories, matching the final R1 effort.
- Paths: raw (reader control), direct H→B, staged2 H→2B→B, and rewrite H→B→B. P0 has two non-statistical histories; P1 has 12 histories; P2 has 24 new histories. Each history has eight questions; history is the statistical unit.
- Compressor calls receive only `history.text`. They do not receive the query file, future questions, answers, or evidence labels. The compressor manifest records `query_file_opened=false`; reader work begins only after all memories in that phase are frozen.

## Runtime and provenance locks

| Item | Frozen value |
|---|---|
| Live provider model-list snapshot SHA-256 | `8f4a71947dacd5fccaf8353064cab5a7c932a5f332a2631d6ab38f028a6bbd2d` |
| Model-list response artifact | `work/v4_fast_decision_20260913/preflight/provider_models_live_20260913.json` |
| DeepSeek tokenizer revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| GLM tokenizer revision | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |
| V4 runtime config SHA-256 after amendment | `6efe541eccb62508bdffbaaf73c518030d57ea4b7e30706fa045608805bf0415` |
| V4 runner SHA-256 after safe-resume, retry, and provenance forecast fixes | `1d5f5f8da59abe6329a12ee48f3092dd34fd38f882ca09693fd3a96c145548c0` |
| C1 prompt SHA-256 | `253a1e813308b0500b574a6698e1b6bf57a02cfe0c8e0b4afd849f4597d3d0bb` |
| Batch reader prompt SHA-256 | `fff465eab75aceaa282ce29735d84f838fa657ec78aac5b74847bdd569aaf763` |
| Batch retry prompt SHA-256 | `e40803dd3b67def595de3ea1f43af7a35caf9cf6e4c69fd4d1e646f7f28596bf` |
| P0 dataset manifest SHA-256 | `b60cb971b66f5bb99d19c2c0b075566116f39a0ab7ae923ac2228c26a0a0addb` |

P0 histories are `syn-2026091301-00000` and `syn-2026091301-00001`, generated with seed `2026091301`. Their measured lengths are 32,724 and 32,742 DeepSeek-native tokens. They do not reuse the V3 seed/history. The full revisions, request hashes, returned model field, usage, finish reason, per-call UTC start, and cost are recorded with the rows.

## Cost and context gate

The current V3 ledger carries `$1.37211145` actual-plus-uncertain usage, including the still-reserved interrupted request. Its split by model is unknown. For model-cap checks, the full prior amount is conservatively reserved against each model independently; for the project cap it is counted once. At the original pre-amendment decision, no cap had been raised; the later approved ceilings and their full forecast are recorded below.

The forecast uses current OpenCode Go peak rates, no cache discount, and each request's full provider output budget as a conservative upper bound. The official rate table lists GLM-5.3-Flash at `$0.15/$0.50` per million input/output tokens and DeepSeek V4.1 Flash at up to `$0.30/$1.20`; peak rates are used regardless of the current time. Source: [OpenCode Go usage and model pricing](https://dev.opencode.ai/docs/go/).

| Initial B=4,096 forecast, before live P0; includes allowed R1 high promotion | Calls | Future max usage value | Including conservative prior | Cap |
|---|---:|---:|---:|---:|
| C1 DeepSeek compression | 190 | `$4.6508` | — | — |
| R2 DeepSeek batch reread | 72 | `$1.3257` | — | — |
| DeepSeek combined | 262 | `$5.9765` | `$7.3486` | `$8.00` |
| R1 GLM batch scoring | 158 | `$1.7601` | `$3.1322` | `$15.00` |
| Project total | — | — | `$9.1087` | `$18.00` |

Before P0, the initial B=4,096 batch plan fit every authorized cap under this maximum-completion forecast. It left `$0.6514` of the DeepSeek cap after the conservative prior and planned upper-bound usage, `$11.8678` of the GLM cap, and `$8.8913` of the project cap. Largest single-request reservations were about `$0.0297` for C1 and `$0.0112` for R1. The machine-readable calculation is [v4_preflight_cost_20260913.json](/home/huyudi/012_conference/iclr2027/submission/artifacts/v4_preflight_cost_20260913.json). The post-P0 fallback recost below supersedes this forecast for any continuation.

Contingencies are not free: if batch parsing exceeds 5%, the R1 per-question maximum-completion scenario reaches `$15.2149` including the conservative prior, about `$0.2149` above the GLM cap. An R1 mode switch therefore requires a fresh forecast using measured P0 usage before P1/P2. The one-time B=8,192 P0 size fallback also exceeds the DeepSeek cap under the same full-output upper bound; if P0 triggers it, recalculate from measured P0 receipts first. Do not raise caps: stop before the affected requests if the updated forecast does not fit. R2 remains batch unless its own parser evidence and a separate recost justify otherwise.

Context checks pass for C1 (1,048,576), R1 (65,536), and R2 (1,048,576) with the frozen prompts and provider budgets. The P0 cost ledger will enforce the `$18` project cap plus `$8` DeepSeek and `$15` GLM caps before each request; uncertain charges remain reserved.

## V3 boundary and history

V3 is frozen as `INCOMPLETE_PROTOCOL_DEVIATION / SUPERSEDED_FOR_ROUTE_SELECTION`. Its four completed outputs, interrupted request/reservation, 35 unattempted cells, and early-stop deviation remain unchanged and excluded from V4 statistics. The prior 1,024-token no-thinking probe was technically invalid (`finish_reason=length`) and also sent `reasoning_effort=low`; it does not verify the exact V4 wire policy. V4 C1 used `thinking.type=disabled` with `reasoning_effort` omitted. No V3 result is merged with V4.

## Live endpoint check and quarantined attempt

A credential-safe, read-only `GET /v1/models` returned HTTP 200. The exact response bytes are saved at `work/v4_fast_decision_20260913/preflight/provider_models_live_20260913.json`; both `deepseek-v4.1-flash` and `glm-5.3-flash` were present. Its SHA-256 is frozen above and in the runtime configuration.

The first C1 attempt was launched inside the restricted sandbox and left one reservation (`$0.0247104`) for `syn-2026091301-00000 / direct / step 1`, with no provider usage, finish-reason, or response receipt. The attempt is recorded at `work/v4_fast_decision_20260913/p0_b4096/compression_manifest.json` and remains counted conservatively against the caps. It was not replayed: the completed P0 manifest records that cell as `TECHNICAL_INVALID_UNCERTAIN` and excludes it from scoring.

The sandbox denied the configured local-proxy socket. The read-only model-list check succeeded only after network escalation with proxy discovery disabled. P0 provider calls used that scoped direct route; this changed transport only, not model, prompt, token budget, or experimental path. The credential is loaded from the ignored `.env` entry `PILOT_OPENCODE_GO_API_KEY`; its value is never written to artifacts. Provider-call rows record their own `run_start_utc` and usage metadata.

## Original P0 outcome and old-cap stop (historical, 13:33 UTC)

The C1 batch finished at `2026-09-13T13:25:42Z`; its manifest is `work/v4_fast_decision_20260913/p0_b4096_live/compression_manifest.json`. Final memory outcomes were: natural stop `2/6` (direct 2,387 tokens; rewrite 1,880 tokens), `TECHNICAL_INVALID_LENGTH` `3/6` (two intermediate calls at 16,384 completion tokens and one final call at 12,288), and `TECHNICAL_INVALID_UNCERTAIN` `1/6`. Only `1/6` final memories fell in `[2,048, 8,192]`; query-blindness checks passed. The gate requires at least `5/6` natural-stop outputs and a majority in that interval, so P0 fails technically before reader calibration.

The master prompt permits one adjustment to B=8,192 / intermediate=16,384 with final/intermediate completion budgets 16,384/24,576. Under the old caps, a measured-usage recost forecast DeepSeek at `$9.0454` including the V3 prior and observed P0 usage, exceeding the old `$8.00` cap by `$1.0454`; this produced the historical old-cap stop. No fallback, R1, P1, P2, or R2 request had been issued as of 13:33 UTC. This is not the current execution state.

## Approved resource amendment (current)

At 15:52 UTC the user authorized higher total usage-value ceilings and continuation. The frozen caps are project `$55`, DeepSeek `$13`, and GLM `$42`; these are total USD-equivalent Go usage values, not an auto-top-up or cash budget. `use_balance=false`, auto-top-up is disabled, and no paid fallback is permitted. The OpenCode Go documentation, rechecked at 16:00 UTC, lists DeepSeek V4.1 Flash limits of `$3` per 5 hours, `$7.50` weekly, and `$15` monthly; GLM-5.3-Flash limits are `$12`, `$30`, and `$60`. The local DeepSeek ceiling remains below its provider monthly limit. Provider usage outside this project is not observable, so a window limit can still pause requests; on rejection, preserve state and resume after natural reset without changing model or protocol. Source: [OpenCode Go usage limits and pricing](https://dev.opencode.ai/docs/go/).

The full-completion stress forecast includes the one-time P0 rerun, P1/P2, R1 promotion (including a retried low-effort P0 calibration), a complete R1 batch pass plus single-question fallback, one retry per reader call, and R2 batch plus one retry. It is `$10.4596` DeepSeek, `$33.0829` GLM, and `$42.1705` project total, leaving `$2.5404`, `$8.9171`, and `$12.8295` under the respective local caps. The provider sidecar's 4,000-call, 150M-input-token, and 50M-output-token ceilings cover a stress total of 3,207 calls / 104.96M input / 39.91M output including existing reservations. R2 per-question mode is excluded: its DeepSeek forecast exceeds the documented model monthly limit and amended cap. Machine-readable contract: [v4_resource_amendment_contract_20260913.json](/home/huyudi/012_conference/iclr2027/submission/artifacts/v4_resource_amendment_contract_20260913.json); estimator output: [v4_resource_preflight_amendment_20260913.json](/home/huyudi/012_conference/iclr2027/submission/artifacts/v4_resource_preflight_amendment_20260913.json).

## B=8,192 adjustment result and stop

The isolated run is `work/v4_fast_decision_20260913/p0_b8192_resource_amendment_live/`. Final statuses: natural stop `3/6`, `TECHNICAL_INVALID_LENGTH` `2/6`, and `TECHNICAL_INVALID_UNCERTAIN` `1/6`. Only one natural-stop final is inside the required `[4,096, 16,384]` interval (minimum four); the natural-stop minimum is five. Query-blindness passed. The uncertain staged2 request followed an HTTP 500, remained reserved at `$0.0394614`, and was quarantined without replay. P0 actual-plus-uncertain cost was `$0.1872174`; total project ledger after this run was `$1.72011715`.

Because the sole C1 adjustment still fails, the protocol requires stopping before R1, P1, or P2. This is `NO_GO_TECHNICAL`, not a scientific Memory no-go. Detailed rows and hashes: [P0 adjustment result](/home/huyudi/012_conference/iclr2027/submission/artifacts/v4_p0_b8192_adjustment_result_20260913.json); [compression manifest](/home/huyudi/012_conference/iclr2027/submission/work/v4_fast_decision_20260913/p0_b8192_resource_amendment_live/compression_manifest.json).
