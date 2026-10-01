# How the study works

## The question

Traffic counts change with the time of day and day of the week. A count that looks ordinary at
midnight may be unusual during the morning commute. This study asks how well a simple rule based
on that context compares with Isolation Forest, and how the answer changes across anomaly recipes.

An alert is a reason to inspect a measurement. It does not identify an accident, road closure or
faulty sensor on its own.

## From source rows to observed hours

The UCI data contains hourly traffic volume alongside weather and holiday fields. Some timestamps
appear more than once because of weather descriptions. Adding their traffic counts would count
the same observation repeatedly.

`data.py` parses dates and counts, rejects invalid records, collapses repeated timestamp/count pairs
and excludes timestamps whose counts conflict. Missing hours stay absent; they are not replaced
with zero. A measured zero is retained. Source local time labels are used as supplied, without an
inferred daylight-saving conversion.

The supplied dataset has 48,204 rows, 7,629 repeated same-count rows and 40,575 retained hours.
There are 11,976 absent hours between the first and last retained timestamp. The audit records
these gaps, including a long gap, so the data should not be treated as a continuous hourly series.

## Keep the time periods separate

The ordered observations are split 60% / 20% / 20%:

| Period | Observed hours | Role |
| --- | ---: | --- |
| Training | 24,345 | Learn the models' patterns |
| Validation | 8,115 | Set score thresholds and check the experiment |
| Test | 8,115 | Measure performance on later data |

These fractions count observed hours, not elapsed calendar time. Models never fit on the later test
period. Thresholds come from unchanged validation scores, before test evaluation. After the validation
pilot, the project saves settings and hashes of the code, data and trained-model manifest. Evaluation
checks those records before proceeding.

## Typical-hour baseline

For each weekday/hour combination, the baseline learns a median count and a median absolute deviation
(MAD). The median is the middle count; MAD measures the typical distance from it. The anomaly score is:

```text
score = absolute(observed count − group median) / robust spread
robust spread = 1.4826 × MAD
```

For example, if the median is 120 and the spread is 30, a new count of 300 receives a score of 6.
It triggers an alert only if 6 exceeds the chosen threshold.

A group needs at least ten training observations. Sparse groups fall back to an hour-only group,
then to global statistics. A zero spread falls back to a broader positive spread, with a minimum of
one vehicle. This makes the score usable even when a small group contains identical counts.

## Isolation Forest

Isolation Forest repeatedly separates observations using random feature splits. Observations that
are separated in fewer steps tend to receive more unusual scores. The implementation uses 200 trees
and the same five inputs throughout the comparison:

- Current traffic volume.
- Sine and cosine of the hour of day.
- Sine and cosine of the weekday.

The sine/cosine pairs represent a circular clock, keeping 23:00 close to 00:00. No anomaly labels,
original pre-injection counts or future observations are passed to the model. The study assesses
readings after they arrive; it does not forecast the next hour.

The wrapper reverses `score_samples` so higher scores mean more unusual for both methods. Their raw
score scales still differ, so compare detection metrics rather than score magnitudes.

## Turn scores into alerts

The main threshold targets about 1% alerts on unchanged validation data. It is the 99th percentile
of validation scores, with alerts requiring a score strictly above it. This treats tied scores
conservatively. Additional thresholds target 0.5%, 2% and 5%.

A validation alert target is not an estimate of the true anomaly rate. The observed alert burden
can change in a later period.

## Create changes with known labels

The source has no verified anomaly labels, so each experiment changes a copy of the data and records
the original and altered values. Only hours whose numeric count actually changes receive a positive label.

| Family | Mild | Moderate | Severe |
| --- | --- | --- | --- |
| Spike | ×1.25 | ×1.75 | ×2.5 |
| Drop | Keep 75% | Keep 50% | Keep 10% |
| Shift | ±25% for 3 hours | ±50% for 6 hours | ±90% for 12 hours |
| Flatline | Repeat first count for 3 hours | For 6 hours | For 12 hours |

Counts remain nonnegative integers. Events cannot overlap or cross missing-hour gaps. A flatline's
first value is unchanged, so its window can contain more hours than its positive labels.

Five injection seeds change where alterations occur. The main comparison has four families × three
levels × five seeds = 60 scenarios. Twenty severe scenarios use a higher injection rate, and five
combine families, giving 85 paired scenarios. Each altered dataset is scored by both methods at all
four thresholds. A separate set of forest seeds checks model randomness while holding injections fixed.

Severity describes a recipe. For shifts, both magnitude and duration change; for flatlines, duration
changes. The levels do not isolate the effect of magnitude alone.

## Read the measurements

| Measure | Question it answers |
| --- | --- |
| Recall | What fraction of changed hours was flagged? |
| Benchmark precision | What fraction of alerts carried an injected-change label? |
| F1 | How do precision and recall balance at this threshold? |
| Average precision | How well do scores rank changed hours across thresholds? |
| Background alert rate | What fraction of unchanged hours was flagged? |
| Event recall | Was any changed hour in the event detected? |
| Detection delay | How long until the first detected changed hour, among detected events? |

With only 1% changed hours, a method that flags nothing can have 99% accuracy. That is why accuracy is
not the main result. Event detection also does not turn every hour in that event into a correct detection.

“Unchanged” does not mean “confirmed normal”: some source observations may already contain real issues.
Precision and false-positive counts therefore describe the synthetic benchmark. Background alerts are
not verified sensor-error false alarms.

## What reliability means here

The study compares seed ranges, different recipes, threshold choices, injection prevalence and monthly
alert rates. Ranges across five seeds describe variation; they are not confidence intervals for future
traffic. Small subgroups and events need particular care when interpreting percentages.

The baseline has higher median recall for the four severe recipes in the reference run. Isolation
Forest does better on mild spikes and mild shifts. Neither method uses historical lag features, which
helps explain why a flatline can be hard to detect when its repeated value remains plausible.

The results cover one station and one historical split. They do not establish incident detection
accuracy elsewhere. Useful follow-ups include weather features, temporal features, additional periods
and independently labelled incidents. After inspecting the test results, any redesign should be
reported as exploratory or assessed on a new held-out period.

## Read alongside the code

Start with `data.py`, then `models.py` and `features.py`, followed by `injection.py`, `evaluation.py`
and `experiments.py`. `reporting.py` converts saved measurements into the charts and report.
The notebooks follow the same order with smaller examples.

References: [UCI data](https://doi.org/10.24432/C5X60B),
[IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html),
[data leakage](https://scikit-learn.org/stable/common_pitfalls.html),
[precision and recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).
