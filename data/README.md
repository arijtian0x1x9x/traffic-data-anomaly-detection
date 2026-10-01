# Data source and processing

The dataset is **Metro Interstate Traffic Volume**, by John Hogue (2019), from the UCI Machine
Learning Repository. It records hourly westbound I-94 traffic volume near Minneapolis–Saint Paul.

- Dataset and citation: https://doi.org/10.24432/C5X60B
- Source page: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The original archive and compressed CSV are included in `raw/` with their source URLs and SHA-256
hashes in `source.json`. They are unmodified UCI files. If absent, `uv run traffic-study prepare`
downloads them from the recorded source.

The preparation stage creates derived data in `processed/`: one record per retained local timestamp,
chronological splits, excluded conflicting records and a list of missing hours. Repeated records
with the same timestamp and count are collapsed. Conflicting timestamp groups are excluded. Gaps
are retained, and measured zeros remain valid observations.

Weather descriptions can create repeated timestamp rows. Their traffic counts must not be summed.
Weather and holiday fields are available in the source but are not inputs to the core detectors.
Local time labels are kept as supplied; no daylight-saving conversion is inferred. Ambiguous local
clock times remain a limitation.

Synthetic changes are made to copies and recorded in `reports/results/injected_changes.csv.gz`.
They are experimental derivatives, not original observations or verified incident labels.

Please preserve the dataset attribution when sharing these files or derivatives. The repository's
code licence does not replace the dataset's CC BY 4.0 licence.
