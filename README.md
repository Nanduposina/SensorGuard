# SensorGuard

SensorGuard is a reproducible data-quality and anomaly-investigation pipeline for the Intel Berkeley Research Lab sensor dataset. It keeps malformed and suspicious readings auditable, evaluates detectors against controlled synthetic faults, and provides an interactive Streamlit dashboard. The historical data is unlabeled: naturally occurring patterns are candidates for investigation, not verified failures.

CI runs on pushes and pull requests through [GitHub Actions](./.github/workflows/ci.yml). A repository-specific badge can be added after publishing this folder to a known GitHub owner/repository.

## What it does

- Streams plain text or gzip records and preserves raw line evidence.
- Validates timestamps, sensor identifiers, values, duplicates, ordering and sampling gaps.
- Produces configurable rule findings, Isolation Forest scores and PyTorch autoencoder reconstruction errors.
- Uses chronological train/validation/test partitions; fits preprocessing on training rows and calibrates model thresholds on unmodified validation data (not independently verified fault-free).
- Injects seeded synthetic faults into a separate held-out copy and records the changed readings and ground truth.
- Reports feature-reading pair metrics against synthetic labels, no-injection test alert rates, fault-type results and exploratory voltage cases.
- Serves cached data and precomputed results in a six-section Streamlit dashboard; the UI does not train models.

## Architecture

```mermaid
flowchart LR
  Raw[Plain or gzip sensor file] --> Stream[Streaming ingestion]
  Stream --> Quality[Validation and quality flags]
  Quality --> Profile[Profile and clean export]
  Quality --> Split[Chronological train / validation / test]
  Split --> Rules[Rule checks]
  Split --> IF[Isolation Forest]
  Split --> AE[PyTorch autoencoder]
  Split --> Inject[Seeded faults on test copy]
  Rules --> Merge[Evidence-preserving alert merge]
  IF --> Merge
  AE --> Merge
  Inject --> Metrics[Evaluation tables and plots]
  Merge --> Dashboard[Streamlit dashboard]
  Metrics --> Dashboard
  Dashboard --> Review[Local SQLite review store]
  Review --> Backup[Validated JSON backup / restore]
```

## Requirements and setup

Python 3.10 or newer and Git are required. The clean demo below was verified with Python 3.12.10 on Windows; CI runs Python 3.11 on Linux. PowerShell is used for the documented Windows commands. Install the package and development/dashboard tools from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
python -m pytest -q
```

Linux/macOS activation is `source .venv/bin/activate`. CPU inference and training are supported; CUDA is optional. CI uses Python 3.11 and synthetic data only.

The editable installation installs the dependencies declared in `pyproject.toml`, including PyTorch and Streamlit. PyTorch is a large download; allow several hundred megabytes for the environment. The demo writes bounded output under `artifacts/phase4-demo/` and a local SQLite review store under `.sensorguard/`; both are ignored by Git. On the verification machine repeated full profiles took 24.9–58.0 seconds with a 330 MiB peak working set; the four-sensor experiment took 49.1 seconds. Timings depend on hardware and disk speed.

## Data provenance and placement

The source is the [Intel Berkeley Research Lab sensor dataset](https://db.csail.mit.edu/labdata/labdata.html). The original records were collected by the Intel Research Berkeley lab deployment. Cite and acknowledge the dataset documentation when reusing it. This project does not redistribute the raw data. Place the local file at `data.txt/data.txt` or pass a path explicitly. The discovered local source was a 150,910,946-byte plain-text file; it remains unchanged and is excluded from Git.

The records contain date, time, per-sensor epoch sequence, mote ID, temperature, humidity, light and voltage. Epoch is not globally unique. The official page describes IDs 1–54 and notes that data from some motes may be missing or truncated. This local file contains all IDs 1–54, additional IDs 55–58, and isolated IDs 6485, 33117 and 65407. SensorGuard preserves these records and flags IDs outside the configured 1–54 range rather than forcing a 54-sensor count.

The initial profile counted 93,879 rows as short because whitespace collapsing removed empty columns. The raw file uses repeated spaces for missing columns and five trailing spaces on a terminal block of 526 rows. Preserving literal spaces maps all 2,313,682 records to eight positional columns: 92,976 rows have blank light, 373 have blank temperature/humidity/light, four other rows have unusual IDs/blank positions, and the final 526 have blank mote ID and all measurement slots. These 526 rows form the contiguous file suffix. They remain unassigned to any sensor; their physical cause cannot be established from the file. Every row retains its raw text and line number. Explicit empty measurements remain missing values, not discarded rows. The full profile measures 2,312,781 temperature, 2,312,780 humidity, 2,219,804 light and 2,313,156 voltage values. The one-pass profiler also measures 0 repeated (timestamp, sensor ID) keys, 227,392 input-order reversals, and 231,380 consecutive per-sensor intervals over 300 seconds. Under the documented screening bounds, it flags 388,857 temperature, 299,084 humidity, 0 light and 1 voltage values; these are heuristic findings, not confirmed faults. The profiler records elapsed time and peak working set in its report. Out-of-range readings remain in the data.

## Reproduce the data pipeline

Profile the dataset without retaining the complete file in memory. The profiler measures field counts, parsed timestamp coverage, sensor counts, feature ranges/missingness and input-order gaps. Its report goes to the ignored `artifacts/` directory.

```powershell
python scripts/profile_dataset.py "data.txt/data.txt"
python scripts/profile_dataset.py "C:\path\to\data.txt.gz"
```

The parser and chunk reader are available from `sensorguard.ingestion`; `validate_frame()` annotates a supplied DataFrame without deleting or repairing suspicious observations. There is no `clean_frame()` API and no automatic row-removal policy. `export_cleaned()` writes every supplied row, including raw-line provenance and issue evidence, to a derived CSV. “Duplicate” means a repeated `(timestamp, sensor ID)` key; rows that share that key can differ in epoch or measurements, so this is not a byte-identical-row test. Keep raw data immutable and treat generated exports as reproducible artifacts.

## Train detectors and export alerts

The training CLI materializes its input as a DataFrame. Do not pass the complete 2.3-million-row source on a memory-constrained machine. Instead, run the bounded experiment below; it saves the exact unmodified sample to `sampled_records.csv`. Then fit models and export predictions from that sample:

```powershell
python scripts/run_detectors.py artifacts/all_sensors_250/sampled_records.csv `
  --output artifacts/all_sensors_250/trained_predictions.csv `
  --model-dir artifacts/all_sensors_250/models `
  --epochs 3 `
  --threshold-quantile 0.99
```

The command saves `isolation_forest.joblib` and `autoencoder.pt`, and writes chronological test predictions. `sensorguard-detect` is the equivalent installed entry point. Missing values are imputed in model preprocessing; the source records and quality findings remain available in the sampled input.

## Controlled evaluation and metric meaning

Run a deterministic bounded sample experiment across every observed source ID. This includes IDs outside the official expected range; they are kept so the rule detector can surface them. Set the cap and epochs to fit available memory and runtime:

```powershell
$sensorIds = (1..58) + @(6485, 33117, 65407)
$sensors = $sensorIds -join ','
python scripts/run_experiment.py `
  --input "data.txt/data.txt" `
  --output-dir artifacts/all_sensors_250 `
  --sensors $sensors `
  --max-per-sensor 250 `
  --seed 42 `
  --epochs 3
```

The runner scans the source, retains at most the first N source-order rows per requested ID, removes isolated records outside a two-day neighborhood around the selected sample median, and sorts the bounded sample chronologically. It fits on unmodified training data, calibrates model thresholds on the unmodified validation score distribution, and injects faults only into a copy of the held-out test partition. Unmodified does not mean verified fault-free. The evaluation unit is a sensor-feature-reading pair (timestamp-only edits use a record marker). Threshold calibration uses validation data only; final test labels are not used to fit or calibrate. Because only injected modifications have known labels, precision, F1, false-positive rate, true negatives, and PR curves are withheld for this partially labeled set. The report instead gives injected-pair recall, missed injected pairs, and descriptive alert rates on unlabeled readings. It also includes per-fault, per-sensor, detector/feature and rule-contribution breakdowns, distinct alerted-reading counts, and delay for supported span faults. Combined alerts are deduplicated per pair and scored as OR decisions.

The rule, Isolation Forest, and autoencoder experiment path uses rule findings, training-fitted per-sensor baselines, per-sensor temporal deltas, time-of-day features (Isolation Forest), and multivariate reconstruction/scoring. The separate `contextual_deviations()` helper can compare a reading with prior peer values and has backward-only alignment; `align_context()` is used for dashboard comparisons. Neither peer comparison is called by `evaluate_track()` or supplied as an autoencoder/Isolation Forest input. The models therefore do not currently use nearby sensors as detection reference signals. Context comparisons are investigative displays/helpers, not model evidence in reported detector metrics.

The experiment directory contains `sampled_records.csv`, split/test records, injected observations and manifest, per-pair prediction details, detector metrics, no-injection alert baseline, per-fault and per-sensor tables, detector/feature/rule and distinct-reading alert breakdowns, threshold metadata, voltage candidates, plots, and `experiment_report.md`. Unmodified observations have no verified labels, so their alerts are descriptive rather than false-positive counts.

### Larger all-sensor bounded experiment and diagnosis

The most recent full one-pass profile processed 2,313,682 rows in 24.90 seconds with a measured peak working set of 330.0 MiB on this Windows environment. The larger all-sensor bounded evaluation scanned the source and selected up to 250 source-order records for each of the 61 observed IDs, then retained 12,977 rows in its dense time window (1,064 outside-window rows excluded). The chronological split was 7,786 train, 2,595 validation, and 2,596 test rows. It injected 3 events affecting 11 feature-reading pairs, evaluated 12,980 pairs, and completed the detector experiment in about 44 seconds; training/export on the saved sample took about 18 seconds. The saved evaluation threshold quantile was 0.99.

These figures supersede the earlier confusion table, which treated every unmodified reading as a verified negative. That assumption was not supported by the unlabeled source, so the old precision and apparent-FPR values are withdrawn as accuracy claims. The current run has 12,980 sensor-feature-reading pairs, 11 directly modified positives, and 12,969 unlabeled pairs. No real-world precision, F1 or false-positive rate can be established from this sample.

| Detector / ablation | Injected TP / FN | Injection recall | Alerts on unlabeled pairs | Unlabeled alert rate |
|---|---:|---:|---:|---:|
| Rules | 11 / 0 | 1.0000 | 3,014 | 0.2324 |
| Isolation Forest | 0 / 11 | 0.0000 | 818 | 0.0631 |
| Autoencoder | 1 / 10 | 0.0909 | 873 | 0.0673 |
| Quality-rule baseline | 5 / 6 | 0.4545 | 167 | 0.0129 |
| Statistical-rule baseline | 6 / 5 | 0.5455 | 2,847 | 0.2195 |
| Combined rules + IF + AE | 11 / 0 | 1.0000 | 3,754 | 0.2895 |

The 11 injected positives span three synthetic events (one spike, five drift points and five missing-block pairs). Thresholds use the training-fitted preprocessing and the 99th percentile of unmodified validation scores; the final test injections do not participate in fitting or threshold selection. The 12,977-row sample has chronological partitions of 7,786 / 2,595 / 2,596 train/validation/test records. The unlabeled alert rates are descriptive only: an alert outside an injected pair may be a false alert, a natural anomaly, or an unverified condition.

The earlier baseline run (same seed and split) emitted 4,870 rules alert pairs and 5,603 pairs in the all-detector union (including 11 injected pairs). The recalibrated run emits 3,025 rules pairs and 3,765 combined pairs, reductions of 37.9% and 32.8%. These are alert-volume reductions, not demonstrated specificity or precision improvements. The dominant statistical rule, `sensor_baseline_deviation`, fell from 4,673 to 2,812 pairs. Its former fixed robust-z threshold divided by a near-zero `1e-6` MAD floor and caused extreme scores on nearly constant sensor features. The rule now uses training-derived per-feature scale floors and per-feature robust-z cutoffs calibrated at the 99th percentile of validation scores. Floors remain configurable and are safeguards, not physical truth.

In the recalibrated run, the quality-rule ablation generated 172 pairs versus 2,853 for the statistical-rule ablation. Quality rules catch data-integrity and physical-limit conditions; statistical deviations dominate volume and require contextual review. Multivariate detector outputs are record-level scores projected onto present feature rows for investigation. The report and dashboard therefore expose both deduplicated feature-pair alerts and distinct sensor-timestamp readings, plus per-sensor, per-feature, per-rule, and detector contributions. Alert suppression/grouping has not been introduced: repeated findings must remain traceable to the source reading until an operational incident policy is defined.

Historical voltage patterns remain candidates only. Voltage is strongly related to temperature in the dataset; analysis adjusts for a per-sensor linear temperature relationship but cannot make a causal attribution. No independent battery-failure labels are available. The detector is an exploratory research pipeline, not an unattended or production-validated alarm system.

A smaller four-sensor smoke run (100 rows per ID, seed 42, two epochs) took 49.1 seconds and produced 386 sample rows, 78 test rows, and 3 injected events. Generated artifacts are local under `artifacts/` and excluded from Git. These runs exercise the full detector workflow on a bounded sample, not model training over every raw record.

The latest clean-install demo rerun used Python 3.12.10 and the four-sensor configuration above. It retained 386 rows (14 outside the dense time window excluded), split into 231 train / 77 validation / 78 test records, and injected three events that modified 11 feature-reading pairs. There were 390 evaluated test pairs: 11 known injected positives and 379 unlabeled pairs. The combined rules + Isolation Forest + autoencoder detector found all 11 injected pairs (recall 1.0) and flagged 185 of the 379 unlabeled pairs (48.8% descriptive alert rate). Rules alone also found 11/11 and flagged 80 unlabeled pairs (21.1%); Isolation Forest found 6/11 and flagged 98 (25.9%); the autoencoder found 6/11 and flagged 93 (24.5%). These are small synthetic-injection results. The 379 observations are not verified negatives, so precision, F1 and false-positive rate cannot be calculated. Threshold quantile was 0.99, fitted/calibrated without test labels; full measured details and exact splits are in the generated `experiment_metadata.json` and `metrics.csv`.

## Dashboard

```powershell
$env:SENSORGUARD_DATA_PATH = "data.txt/data.txt"
$env:SENSORGUARD_EXPERIMENT_DIR = "artifacts/all_sensors_250"
$env:SENSORGUARD_PROFILE_PATH = "artifacts/data_profile.md"
$env:SENSORGUARD_MODEL_DIR = "artifacts/all_sensors_250/models"
streamlit run app.py
```

Overview displays available profile and experiment outputs; Sensor explorer plots bounded sensor readings and peer references; Alert investigation filters and exports precomputed findings; Alert review collects independent labels; Model evaluation separates synthetic metrics from unlabeled case studies and matched human reviews; Sensor health summarizes completeness, gaps, alerts and candidate voltage signals. Missing artifacts or data produce actionable empty states. Set `SENSORGUARD_DATA_PATH`, `SENSORGUARD_EXPERIMENT_DIR`, `SENSORGUARD_PROFILE_PATH`, `SENSORGUARD_MODEL_DIR` and `SENSORGUARD_REVIEW_STORE` to configure paths. The data has no timezone metadata and the dashboard presents timestamps as recorded.

## Fresh-clone end-to-end demo (Windows PowerShell)

Download the Intel Berkeley dataset from the official [dataset page](https://db.csail.mit.edu/labdata/labdata.html) and place the original text file at `data.txt/data.txt` (or supply another path). SensorGuard does not download or redistribute the dataset. From a fresh clone, these commands install into a new environment, profile the source, run a deterministic bounded experiment, and launch the dashboard:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
python -m pip check
./scripts/demo.ps1
```

The demo script accepts `-InputPath`, `-Python`, `-Port`, `-OutputDirectory`, and `-ReviewStorePath`; for example, `./scripts/demo.ps1 -InputPath "D:\datasets\data.txt.gz" -Python .\.venv\Scripts\python.exe -Port 8502 -OutputDirectory artifacts\phase4-demo-run2 -ReviewStorePath .sensorguard\phase4-run2.sqlite3`. It refuses to proceed when the input is missing, profiles it to `<OutputDirectory>/data_profile.md`, and runs the experiment on sensors 1–4 with a 100-row per-sensor cap, seed 42 and two autoencoder epochs. Outputs are local under `<OutputDirectory>/experiment/`. It refuses to use an output folder containing files or an existing review database, so reruns must choose a new path and prior reports/reviews are not overwritten. No prior `phase2_improved` or `phase3-demo` artifacts are required.

When the dashboard opens at `http://localhost:8501`, use **Alert investigation** to filter unmodified alerts, then open **Alert review**. This bounded run contains multiple reviewable unmodified alerts: select different alerts and save one as **Confirmed anomaly**, one as **Normal behavior**, and one as **Uncertain / needs investigation**; add notes if useful. Revisit a saved alert by selecting it and update its decision. Use **Download review backup (JSON)** to save a backup outside the repository. To test restore, choose the backup file under **Import a SensorGuard review JSON**, click **Import review file**, and explicitly enable overwrite only if replacing matching IDs is intended. The page reports imported and replaced counts; malformed files and unconfirmed conflicts are rejected without partial writes. **Model evaluation** shows synthetic injected labels separately from the unmodified and human-reviewed historical alerts. Do not use injected test-copy alerts as historical review labels.

The script serves all six sections using the generated artifacts: Overview, Sensor explorer, Alert investigation, Alert review, Model evaluation and Sensor health. To rerun only a stage, use the profiler and experiment commands under the corresponding sections above and point `SENSORGUARD_EXPERIMENT_DIR` and `SENSORGUARD_PROFILE_PATH` to the outputs. The output directory may be deleted and regenerated; the raw dataset is never modified.

### Alert-review storage and interpretation

Open **Alert review**, filter the unmodified historical alert list, select an alert, inspect its sensor, recorded timestamp, feature value, detector, score/severity and explanation, then choose a review label. Add optional evidence notes and press **Save review decision**. A saved decision can be revisited and updated. The page excludes predictions from the synthetic-injection copy so generated test faults cannot be mistaken for historical findings. The Model evaluation page separately reports the class balance and confirmation rate among matched, binary-reviewed alerts; this selected-alert rate is descriptive, not recall or dataset-wide precision.

By default, feedback is stored in `.sensorguard/reviews.sqlite3`, separate from raw readings, prediction files and model artifacts. Set `SENSORGUARD_REVIEW_STORE` to use another local path. The `.sensorguard/` directory is ignored by Git. Use **Download review backup (JSON)** to export a versioned portable backup. Import validates the schema, labels, identities, timestamps and duplicate IDs before writing. If imported IDs already exist, import stops without writing anything unless the reviewer explicitly checks the overwrite confirmation. Store updates and imports use SQLite transactions.

Alert IDs are SHA-256 hashes of the recorded timestamp, sensor ID, feature, observed value and detector identity. Scores and explanations are omitted so recalibrating a detector does not change identity for the same reading and detector. A changed value, sensor, timestamp, feature or detector yields a different ID. Current prediction exports do not contain a unique raw-line identifier, so physically duplicated source rows with identical identity fields cannot be distinguished; they collapse to one reviewable identity. Reviews only match the current evaluation artifact when its alert ID matches exactly. Uncertain and unlabeled examples are excluded from binary reviewed-alert summaries. Feedback never retrains a detector, changes thresholds, or enters the synthetic-test metrics. Back up the JSON export somewhere separate from this working directory; local reviews are not synchronized or encrypted by SensorGuard.

For screenshots, launch the app, open each sidebar section and capture the browser view. Label any screenshots showing injected faults as synthetic evaluation results.

## Development and CI

```powershell
ruff check .
ruff format --check .
python -m pytest -q
```

GitHub Actions installs the package with dashboard/development extras, runs Ruff lint and format checks, executes the test suite and explicitly runs the small end-to-end experiment. The CI job does not use the local raw dataset, trained checkpoints or GPU. Use the same commands locally before submitting changes.

## Limitations and next steps

The rule limits are configurable heuristics; contextual differences can reflect legitimate local conditions. Peer comparisons are not integrated into the model/evaluation path. The bounded experiment is sensitive to sample window and injection choices, and its alerts on unlabeled readings cannot be called false alarms. The convenience detector-training script loads its complete input in memory. Better labeled field failures, calibrated operational thresholds, streaming model training, broader sensor/time coverage and independent battery ground truth are needed before operational use.
