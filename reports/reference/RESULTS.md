# Results and interpretation

Reference results from the supplied configuration and dataset.

## What was run

The study uses a training-only weekday/hour median baseline with a robust MAD spread,
and a 200-tree Isolation Forest using volume and cyclic hour/weekday inputs.
Thresholds are calibrated on unchanged validation data. The test period is used after
the protocol is frozen. Both models receive identical injected records.

- 85 paired test scenarios; 680 metric rows
  covering two methods and four calibration targets.
- Five injection seeds; 20 separate forest-seed checks.
- 1% target injection rate for the main study, plus 5% severe stress scenarios and a mixed scenario.
- Spikes, drops, shifts and flatlines at three recipe levels.
- Recipe levels change duration as well as magnitude for shifts; they are not a pure magnitude experiment.

## Data audit

The original download contains 48,204 rows.
After resolving repeated timestamps, the model uses 40,575 unique observed hours.
7,629 duplicate same-count rows were collapsed;
0 timestamp groups with conflicting counts were excluded.
There are 11,976 absent hours across the full retained span.
Gaps were retained, not filled with zeros. Source local timestamp labels were used without
guessing a daylight-saving conversion. Ambiguous repeated local times remain a limitation.

- train: 24,345 hours, 2012-10-02 09:00:00 to 2016-11-20 05:00:00.
- validation: 8,115 hours, 2016-11-20 06:00:00 to 2017-10-26 17:00:00.
- test: 8,115 hours, 2017-10-26 18:00:00 to 2018-09-30 23:00:00.

## Severe scenarios at the 1% validation alert target

Entries are across five injection seeds. Recall is per changed hour, not event-adjusted.
These labels identify deliberate changes; unchanged records may still contain real anomalies.

| family | method | recall_median | recall_min | recall_max | precision_median | ap_median |
| --- | --- | --- | --- | --- | --- | --- |
| spike | Typical-hour baseline | 0.939 | 0.915 | 0.976 | 0.453 | 0.884 |
| spike | Isolation Forest | 0.610 | 0.561 | 0.634 | 0.388 | 0.493 |
| drop | Typical-hour baseline | 0.707 | 0.659 | 0.780 | 0.384 | 0.621 |
| drop | Isolation Forest | 0.500 | 0.463 | 0.573 | 0.345 | 0.447 |
| shift | Typical-hour baseline | 0.667 | 0.583 | 0.726 | 0.377 | 0.527 |
| shift | Isolation Forest | 0.500 | 0.357 | 0.560 | 0.347 | 0.374 |
| flatline | Typical-hour baseline | 0.511 | 0.432 | 0.659 | 0.324 | 0.426 |
| flatline | Isolation Forest | 0.250 | 0.193 | 0.341 | 0.222 | 0.169 |

## Alerts on the unchanged test period

- Typical-hour baseline: 1.16% of unchanged test hours flagged.
- Isolation Forest: 0.99% of unchanged test hours flagged.

These rates are observed background alert burdens, not verified sensor-error false-positive rates.
A 1% calibration target on validation does not guarantee a 1% rate on later traffic.

## How to read the comparison

Across the 60 matched primary family/severity/seed comparisons, Isolation Forest had
higher recall in 10, lower recall in 48, and equal recall in 2.
This counts comparisons equally and is descriptive, not a statistical significance test.
Use family-specific results, precision, average precision and alert burden together.
The project does not assume that machine learning must beat a simpler rule.

## Reliability and limits

- Use summary.csv for injection-seed ranges and paired_recall_differences.csv for matched differences.
- Use model_seed_sensitivity.csv to isolate forest randomness on fixed severe injections.
- Use subgroups.csv for month/hour/weekday-weekend results with actual counts. Small groups are uncertain.
- Event recall reports whether a changed hour is flagged. Delay is measured only for detected events;
  missed events remain explicit in event recall.
- Flatlines can be difficult because neither core detector uses lag features.
- Synthetic perturbations may be unrealistic or remain plausible; results depend on their design.
- Mild changes do not necessarily correspond to genuine faults.
- One road, historical observations and unknown background anomalies limit generalisation.
- No real incident labels, live deployment, forecasting or causal diagnosis are claimed.

## Reproduce

Run the numbered stages in START_HERE.md. The frozen protocol, source hashes, data hashes,
package lockfile, injection logs and run manifest provide the record of this experiment.
Stochastic results should reproduce with the same locked environment and seeds; runtimes can vary.

## Figures

1. [Training-only daily patterns](figures/01_training_patterns.png)
2. [Recall by scenario](figures/02_recall_by_condition.png)
3. [A fixed spike example](figures/03_example_alerts.png)
4. [Precision–recall](figures/04_precision_recall.png)
5. [Monthly background alerts](figures/05_monthly_background_alerts.png)
6. [Threshold trade-off](figures/06_threshold_tradeoff.png)

## Sources

- [UCI dataset and attribution](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
- [IsolationForest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Avoiding leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [Precision and recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
