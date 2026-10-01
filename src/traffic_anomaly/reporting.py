"""Generate figures and a readable report directly from saved results."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .data import load_split
from .paths import FIGURES, RESULTS, ROOT, config, prepare_dirs

LABELS = {"typical_hour": "Typical-hour baseline", "isolation_forest": "Isolation Forest"}
COLORS = {"typical_hour": "#226A8E", "isolation_forest": "#C86832"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame, formats=None):
    """Avoid an extra tabulate dependency."""
    formats = formats or {}
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = [formats[c](row[c]) if c in formats else str(row[c]) for c in frame.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def report():
    prepare_dirs()
    cfg = config()
    summary = pd.read_csv(RESULTS / "summary.csv")
    scores = pd.read_csv(RESULTS / "metrics.csv")
    rates = pd.read_csv(RESULTS / "original_alert_rates.csv")
    audit = json.loads((RESULTS / "data_audit.json").read_text())
    manifest = json.loads((RESULTS / "run_manifest.json").read_text())
    train = load_split("train")
    # 1. Only training data informs exploratory pattern charts.
    plot_data = train.assign(
        hour=train.date_time.dt.hour,
        day_type=np.where(train.date_time.dt.dayofweek < 5, "Weekday", "Weekend"),
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    for label, group in plot_data.groupby("day_type"):
        hourly = group.groupby("hour").traffic_volume
        med, lo, hi = hourly.median(), hourly.quantile(0.25), hourly.quantile(0.75)
        ax.plot(med.index, med.values, label=label, linewidth=2)
        ax.fill_between(med.index, lo, hi, alpha=0.12)
    ax.set(
        xlabel="Hour of day (source local time)",
        ylabel="Vehicles per hour",
        title="Training data: median traffic and middle 50% of counts",
        xticks=range(0, 24, 3),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, "01_training_patterns.png")

    # 2. Recall across conditions, with injection-seed ranges.
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    for ax, family in zip(axes.flat, cfg["families"]):
        for j, name in enumerate(LABELS):
            rows = summary[(summary.family == family) & (summary.method == name)].set_index(
                "severity"
            )
            rows = rows.loc[cfg["severities"]]
            x = np.arange(3) + (j - 0.5) * 0.16
            ax.errorbar(
                x,
                rows.recall_median,
                yerr=[rows.recall_median - rows.recall_min, rows.recall_max - rows.recall_median],
                marker="o",
                capsize=3,
                label=LABELS[name],
                color=COLORS[name],
            )
        ax.set(
            title=family.capitalize(),
            xticks=range(3),
            xticklabels=cfg["severities"],
            ylim=(-0.03, 1.03),
        )
        ax.grid(alpha=0.2)
        ax.set_ylabel("Injected-hour recall")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.025), ncol=2, frameon=False
    )
    fig.suptitle("Recall at a 1% validation alert target", y=1.04)
    _save(fig, "02_recall_by_condition.png")

    # 3. A preselected severe spike example, not a cherry-picked best success.
    example = pd.read_csv(RESULTS / "example_spike.csv", parse_dates=["date_time"])
    first = int(np.flatnonzero(example.injected.to_numpy())[0])
    window = example.iloc[max(0, first - 36) : first + 37]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(
        window.date_time, window.original_volume, color="#77838A", label="Original", linewidth=1.3
    )
    ax.plot(window.date_time, window.traffic_volume, color="#202F3A", label="Altered", linewidth=1)
    selected = window[window.injected == 1]
    ax.scatter(
        selected.date_time,
        selected.traffic_volume,
        marker="x",
        s=90,
        color="#B23040",
        label="Injected change",
        zorder=5,
    )
    for name, marker in [("typical_hour", "o"), ("isolation_forest", "s")]:
        flagged = window[window[f"{name}_alert"]]
        ax.scatter(
            flagged.date_time,
            flagged.traffic_volume,
            s=90,
            marker=marker,
            facecolors="none",
            edgecolors=COLORS[name],
            label=LABELS[name],
        )
    ax.set(ylabel="Vehicles per hour", title="First injected severe spike in seed 101")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    _save(fig, "03_example_alerts.png")

    # 4. Threshold-free ranking on the same fixed example.
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in LABELS:
        precision, recall, _ = precision_recall_curve(example.injected, example[f"{name}_score"])
        ax.step(recall, precision, where="post", label=LABELS[name], color=COLORS[name])
    ax.axhline(example.injected.mean(), linestyle=":", color="gray", label="Injected prevalence")
    ax.set(
        xlabel="Recall",
        ylabel="Benchmark precision",
        title="Severe spikes: precision–recall curves",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, "04_precision_recall.png")

    # 5. Alert burden on unchanged test records, by month.
    original = pd.read_csv(RESULTS / "original_test_scores.csv.gz", parse_dates=["date_time"])
    original["month"] = original.date_time.dt.strftime("%Y-%m")
    monthly = (
        original.groupby("month")
        .agg(
            observed_hours=("date_time", "size"),
            typical_hour=("typical_hour_alert", "mean"),
            isolation_forest=("isolation_forest_alert", "mean"),
        )
        .reset_index()
    )
    monthly.to_csv(RESULTS / "monthly_background_alerts.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for name in LABELS:
        ax.plot(
            monthly.month,
            100 * monthly[name],
            marker="o",
            markersize=3,
            label=LABELS[name],
            color=COLORS[name],
        )
    ax.axhline(
        100 * cfg["primary_alert_rate"], color="gray", linestyle=":", label="Validation target"
    )
    ax.set(ylabel="Unchanged hours flagged (%)", title="Background alerts on later traffic")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, "05_monthly_background_alerts.png")

    # 6. A fixed later dataset allows a fair view of threshold trade-offs.
    severe = scores[(scores.suite == "primary") & (scores.severity == "severe")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in LABELS:
        series = severe[severe.method == name].groupby("calibration_target").recall.mean()
        ax.plot(
            series.index * 100, series.values, marker="o", color=COLORS[name], label=LABELS[name]
        )
    ax.set(
        xlabel="Validation background alert target (%)",
        ylabel="Mean injected-hour recall",
        title="Severe scenarios: detection versus alert budget",
        ylim=(0, 1),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, "06_threshold_tradeoff.png")

    primary_rates = rates[rates.calibration_target == cfg["primary_alert_rate"]]
    display = summary[summary.severity == "severe"][
        [
            "family",
            "method",
            "recall_median",
            "recall_min",
            "recall_max",
            "precision_median",
            "ap_median",
        ]
    ].copy()
    display.method = display.method.map(LABELS)
    table = markdown_table(display, {c: lambda x: f"{x:.3f}" for c in display.columns[2:]})
    rate_lines = "\n".join(
        f"- {LABELS[r.method]}: {r.background_alert_rate:.2%} of unchanged test hours flagged."
        for _, r in primary_rates.iterrows()
    )
    split_lines = "\n".join(
        f"- {name}: {v['rows']:,} hours, {v['start']} to {v['end']}."
        for name, v in audit["split_summary"].items()
    )
    paired = pd.read_csv(RESULTS / "paired_recall_differences.csv")
    wins, losses, ties = (
        (paired.forest_minus_baseline > 0).sum(),
        (paired.forest_minus_baseline < 0).sum(),
        (paired.forest_minus_baseline == 0).sum(),
    )
    content = f"""# Results and interpretation

Generated from the saved experiment outputs. Completed at {manifest["completed_at_utc"]}.

## What was run

The study uses a training-only weekday/hour median baseline with a robust MAD spread,
and a 200-tree Isolation Forest using volume and cyclic hour/weekday inputs.
Thresholds are calibrated on unchanged validation data. The test period is used after
the protocol is frozen. Both models receive identical injected records.

- {manifest["n_test_scenarios"]} paired test scenarios; {manifest["n_metric_rows"]} metric rows
  covering two methods and four calibration targets.
- Five injection seeds; {manifest["n_seed_sensitivity_runs"]} separate forest-seed checks.
- 1% target injection rate for the main study, plus 5% severe stress scenarios and a mixed scenario.
- Spikes, drops, shifts and flatlines at three recipe levels.
- Recipe levels change duration as well as magnitude for shifts; they are not a pure magnitude experiment.

## Data audit

The original download contains {audit["raw_rows"]:,} rows.
After resolving repeated timestamps, the model uses {audit["hourly_rows"]:,} unique observed hours.
{audit["same_count_duplicates_collapsed"]:,} duplicate same-count rows were collapsed;
{audit["conflicting_timestamps_excluded"]:,} timestamp groups with conflicting counts were excluded.
There are {audit["missing_hours_in_full_span"]:,} absent hours across the full retained span.
Gaps were retained, not filled with zeros. Source local timestamp labels were used without
guessing a daylight-saving conversion. Ambiguous repeated local times remain a limitation.

{split_lines}

## Severe scenarios at the 1% validation alert target

Entries are across five injection seeds. Recall is per changed hour, not event-adjusted.
These labels identify deliberate changes; unchanged records may still contain real anomalies.

{table}

## Alerts on the unchanged test period

{rate_lines}

These rates are observed background alert burdens, not verified sensor-error false-positive rates.
A 1% calibration target on validation does not guarantee a 1% rate on later traffic.

## How to read the comparison

Across the 60 matched primary family/severity/seed comparisons, Isolation Forest had
higher recall in {wins}, lower recall in {losses}, and equal recall in {ties}.
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
"""
    (ROOT / "reports" / "RESULTS.md").write_text(content)
    print("Six figures and reports/RESULTS.md generated from actual experiment outputs.")
