# SensorGuard resume-claim audit

## Evidence standard

This audit assesses four proposed resume bullets against the current implementation, tests, measured experiments, and the [official Intel Berkeley Research Lab data page](https://db.csail.mit.edu/labdata/labdata.html). “Verified” applies only to the scope tested here. Artificially injected faults support controlled recall measurements, not historical accuracy. Unmodified readings remain unlabeled unless independently reviewed.

The official page describes 54 deployed motes, their coordinates, per-mote epoch counters, occasional missing/truncated data, and voltage variation strongly correlated with temperature. The local source has 61 observed IDs, including additional IDs; the project does not force the observed count to 54. The official [mote coordinate file](https://db.csail.mit.edu/labdata/mote_locs.txt) supplies physical positions for IDs 1–54.

## Claim assessments

| Proposed resume claim | Verdict | Evidence and limitation | Truthful wording |
|---|---|---|---|
| “Built a sensor-data quality pipeline using public Intel Berkeley Lab data from 54 sensors to identify timestamp issues, gaps, duplicate readings, and implausible values with reproducible cleaning.” | **PARTIALLY VERIFIED** | The streaming profiler and explicit cleaning workflow are implemented. A full profile measured 2,313,682 rows, 61 IDs, zero repeated `(timestamp, sensor_id)` keys, 227,392 per-sensor input-order reversals, and 231,380 intervals over 300 seconds. It found 526 trailing rows with blank sensor and measurement positions; their cause is unknown. Cleaning smoke processing retained 20,000/20,000 rows and preserved quality annotations/provenance. Because the local file contains 61 observed IDs, saying the analyzed file contains only 54 sensors is inaccurate. | “Built a streaming data-quality and provenance pipeline for the Intel Berkeley Lab sensor dataset, with explicit, reproducible policies for missing fields, ordering, gaps, duplicates, and configurable plausibility checks.” |
| “Applied an autoencoder and Isolation Forest to flag potential sensor faults and drift, using nearby sensors as reference signals and comparing results with rule-based checks.” | **VERIFIED (bounded experiment scope)** | The experiment call path now adds causal, backward-aligned features from physically nearby motes using the official coordinates. The run supplied those features to the Isolation Forest and autoencoder and used a validation-calibrated contextual rule. Self-reference and future peer readings are excluded; availability is reported. Coverage rows existed for 32 sensors, but at least one peer feature was available for only 23; mean per-feature availability was 40.27%. Paired 54-mote experiments used the same sampled rows, split, seed, and injected faults. Context did not improve injected-pair recall in this run; it reduced the combined detector’s descriptive alert rate on unlabeled pairs. This is a controlled implementation/evaluation result, not evidence that context improves real-world detection. | “Compared rule-based, Isolation Forest, and autoencoder sensor anomaly detectors, including a causal physical-neighbor feature ablation, on seeded synthetic faults.” |
| “Evaluated detection behaviour using injected faults with known outcomes and real battery-failure cases, considering detected faults alongside false alarms on valid readings.” | **PARTIALLY VERIFIED** | The held-out test copy contained three seeded injected events modifying 11 sensor-feature-reading pairs; injection labels were not used for fitting or validation threshold calibration. In the broader paired run, 7,695 feature-reading pairs were evaluated: 11 injected positives and 7,684 unlabeled pairs. Precision, F1, false-positive rate, and true-negative counts are therefore undefined and intentionally omitted. Repository and official dataset documentation searches found no independently verified battery failure, service, or replacement labels. The voltage analysis reports candidates only. | “Measured detector recall on seeded synthetic sensor faults and summarized alerts on unlabeled historical readings; investigated temperature-adjusted voltage-decline candidates without verified battery-failure labels.” |
| “Added automated tests, GitHub Actions CI, and a Streamlit dashboard to display flagged readings and their reasons for human review.” | **VERIFIED (implemented/local checks; hosted CI pending)** | The full current suite passed 60 tests, Ruff lint/format checks passed, and `pip check` found no broken requirements. Streamlit `AppTest` rendered all six sections with the newly generated real profile and contextual experiment artifacts. A local server started at port 8503, `curl.exe http://127.0.0.1:8503/_stcore/health` returned `ok`, and the process was stopped. The base commit had passing GitHub Actions; CI for the new commit is pending. Reviewer feedback does not itself establish model accuracy. | “Built a Streamlit sensor-investigation dashboard with human alert review, backed by automated tests and GitHub Actions CI.” |

## Reproducible Phase 5 evaluation

Commands used from the repository root, with the existing `.venv`:

```powershell
$python = ".venv/Scripts/python.exe"
$sensors = (1..54) -join ","
& $python scripts/run_experiment.py --input data.txt/data.txt --output-dir artifacts/phase5-baseline-20261027 --sensors $sensors --max-per-sensor 150 --seed 42 --epochs 2
& $python scripts/run_experiment.py --input data.txt/data.txt --output-dir artifacts/phase5-contextual-20261027 --sensors $sensors --max-per-sensor 150 --seed 42 --epochs 2 --contextual --peer-radius-meters 8 --peer-minimum 3 --peer-tolerance 120s
& $python scripts/compare_contextual.py --baseline-dir artifacts/phase5-baseline-20261027 --contextual-dir artifacts/phase5-contextual-20261027 --output artifacts/phase5-contextual-comparison-20261027.csv
```

The runner hashed the source and verified identical sampled rows and fault manifests before comparison. It selected up to 150 source-order readings for each of motes 1–54, retained 7,693 rows in the dense time window, then split chronologically into 4,615 train, 1,539 validation, and 1,539 held-out test rows. Three synthetic faults modified 11 feature-reading pairs. The unit is a sensor-feature-reading pair (with a `record` marker for record-level faults). Metrics count each alert pair once. Models were fit on unmodified training rows; thresholds were calibrated using unmodified validation scores only; injections were applied to the held-out copy after contextual features were prepared.

| Detector | Baseline injected recall | Baseline alerts / rate on unlabeled pairs | Contextual injected recall | Contextual alerts / rate on unlabeled pairs |
|---|---:|---:|---:|---:|
| Rules | 11/11 (1.000) | 1,001 / 13.03% | 11/11 (1.000) | 1,037 / 13.50% |
| Isolation Forest | 0/11 (0.000) | 194 / 2.52% | 0/11 (0.000) | 48 / 0.62% |
| Autoencoder | 1/11 (0.091) | 302 / 3.93% | 1/11 (0.091) | 87 / 1.13% |
| Combined rules + IF + AE | 11/11 (1.000) | 1,293 / 16.83% | 11/11 (1.000) | 1,072 / 13.95% |

The contextual test had features available for 32 of the 54 selected sensors, with mean per-feature-reading availability 40.27%. The quality-rule-only recall was 5/11 in both runs; statistical-rule recall was 6/11 in both. The combined alert-volume reduction is descriptive and does not establish fewer false positives: the 7,684 non-injected pairs are unlabeled. The synthetic distribution has only three events and should not be generalized to natural faults.

Outputs are local ignored artifacts: `artifacts/phase5-baseline-20261027/`, `artifacts/phase5-contextual-20261027/`, the reviewed comparison CSV `artifacts/phase5-contextual-comparison-reviewed-20261027.csv`, a 20,000-row cleaning smoke run, and the separate streaming voltage investigation `artifacts/phase5-voltage-streaming-20261027/` (299 evaluated windows, 296 cutoff candidates). Existing artifacts and user data are not overwritten.

## Cleaning behavior and data profile

`scripts/clean_dataset.py` was run on the original source with `--max-rows 20000 --chunk-size 5000`. It reported 20,000 input rows, all 20,000 retained under the default flag policy, and 559 policy actions (`flag`). It generated a retained CSV, an all-input provenance CSV, and JSON/Markdown summaries. The raw file is unchanged. Policy options can explicitly preserve, flag, drop, or impute missing measurements; imputation requires externally supplied training-only medians and is not interpolation. This smoke run does not replace the separate full-dataset profile.

The current full profiler run measured 2,313,682 records in 56.45 seconds with a 329.9 MiB peak working set. Literal empty positional fields map all rows to eight columns; the terminal blank-ID block has 526 rows beginning at source line 2,313,157 and remains unassigned with raw-line provenance. Sensor IDs observed were 1–58 plus 6485, 33117, and 65407. Timestamp range was 2004-02-28 00:58:15.315133 to 2004-04-05 11:02:32.715337. Missing/unparsed counts: temperature 901, humidity 902, light 93,878, voltage 526. Repeated timestamp/sensor keys: zero. Exact per-feature ranges, configured screening counts, and all per-sensor counts are in the ignored report `artifacts/phase5-profile-20261027.md`.

## Historical battery evidence

No independently documented historical battery-failure cases, maintenance records, or verified battery labels were found in the repository or official data documentation. The official documentation states that voltage variation is strongly correlated with temperature and that these lithium-ion cells generally maintain fairly constant voltage. The full streaming investigation evaluated 299 seven-day sensor windows; 296 crossed its exploratory adjusted-slope cutoff of −0.005 V/day. This unusually broad screen is evidence that the candidate rule is not specific enough to diagnose failures, not evidence of 296 battery events. Temperature-adjusted declines remain observational candidates: the linear adjustment is not causal and load, shared environmental/time effects, calibration, sampling gaps, or truncation can explain patterns. No real-event precision or recall is supportable.

## Remaining scope and release state

The bounded evaluation does not establish production thresholds, natural-fault accuracy, or battery-failure detection. Physical proximity does not prove that sensors share local conditions. Independent labels mapped to stable sensor/time/feature observations, documented battery interventions, and a larger set of representative fault events are needed for stronger evaluation. This audit must be tied to a specific reviewed commit and hosted CI run before being described as released.
