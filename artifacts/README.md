# Generated model files

Training writes fitted models, thresholds and a training manifest here. Validation adds the frozen
protocol record. These files are generated locally and excluded from Git. Recreate them with:

```bash
uv run traffic-study prepare
uv run traffic-study train
uv run traffic-study validate
```
