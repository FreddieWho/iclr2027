# U10 claim correction / R01–R02–O02

All results are correction/reanalysis on historically exposed banks, not independent new-task confirmation. Four-arm contrasts retain every quartet and parent; bootstrap uses 2,000 common parent resamples, quartet-weighted point estimates. T1 distances=6, T2 distances=3.

|Task|Seed|n / parents|raw0|raw flip|distance0|distance flip|I (95% parent CI)|
|---|---|---|---|---|---|---|---|
|T1|11|2724/322|0.096182|0.248899|0.202643|0.748899|0.393539 [0.340966,0.445332]|
|T1|23|2724/322|0.068649|0.209985|0.192364|0.773128|0.439427 [0.387492,0.484586]|
|T1|47|2724/322|0.074156|0.192731|0.196035|0.727239|0.412628 [0.360747,0.460049]|
|T2|11|3443/369|0.151612|0.605867|0.732501|0.997676|-0.189079 [-0.244406,-0.130480]|
|T2|23|3443/369|0.159454|0.639559|0.688934|0.997096|-0.171943 [-0.223202,-0.117699]|
|T2|47|3443/369|0.131281|0.643334|0.738019|0.999710|-0.250363 [-0.303832,-0.196781]|

P-A is feature advantage under the same flip protocol. P-B is retained-label subset comparison, not oracle-query savings. Positive J-scale interaction holds for T1; T2 has negative I despite near-perfect distance+flip J. Direction strata are retained in JSON; this correction does not redefine the T2 task or effect scale.

## Typed normalization repair
Training-only shared x/y moments across T1 vertices or T2 endpoints, with separate query/center moments. Eighteen typed models (2 tasks × 3 seeds × clean/flip/f25) retrained from scratch, fixed Adam .01, 300 epochs; all learning curves and initialization hashes retained. No test epoch selection.

|Task|Seed|old flip J|fixed flip J|
|---|---|---|---|
|T1|11|0.223201|0.276065|
|T1|23|0.204846|0.178047|
|T1|47|0.253304|0.268355|
|T2|11|0.325588|0.339820|
|T2|23|0.330235|0.318908|
|T2|47|0.338949|0.340401|

Full raw-coordinate→normalization→checkpoint-restored network audit: old max logit differences T1=21.5411/T2=23.5805; fixed T1=3.05176e-5/T2=0 (float32 summation), fixed predicted-label differences=0. The independent production regression compares with rtol=1e-5, atol=1e-6 and catches deliberately indexwise bad normalization. Source U01 uses true scalar moments; all six saved source typed checkpoints have max difference=0 over G8, hence no source retrain.

Typed remains below distance in this recipe. Parameter counts are recorded but are not matched; optimization/architecture cannot be eliminated as explanations. No new U10 margin sweep or chirality experiment was run (NOT_RUN).

Evidence: `artifacts/e1a933_review/data_u10/U10_CORRECTED.json`, `SYMMETRY_PIPELINE_AUDIT.json`, `*_scores.npz`, task samples/edits and all training curves; `data_source_eval/SOURCE_TYPED_PIPELINE.json`.
