# Reference results

These results were produced by the supplied study settings and UCI dataset. Read
[the report](RESULTS.md) first. Four CSV tables retain the numerical summaries;
`figures/` contains the six generated charts, and `provenance.json` records the
configuration and data hashes alongside package versions.

The public version reproduced the earlier study's numerical summary exactly.
Timing columns and local execution metadata are omitted from the published examples.
To produce full measurements and change logs, run `uv run traffic-study all`.
A new run writes to `reports/results/` and leaves this reference directory intact.
