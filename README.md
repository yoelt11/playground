# playground

Exploratory research experiments. Quick tests spawned from research ideas —
most won't pan out; anything that proves out and becomes real sustained work
graduates into its own dedicated repo.

## Layout

```
playground/
├── src/                    # shared code across experiments
└── experiments/
    ├── <name>/              # one x-gen-scaffolded folder per experiment
    │   ├── README.md
    │   ├── src/main.py
    │   ├── data/            # gitignored — inputs, not versioned
    │   ├── results/         # gitignored — synced out, not versioned
    │   └── notebooks/
    └── ...
```

New experiments are scaffolded with
[x-gen](https://crates.io/crates/x-gen) (`x-gen <name> -d experiments/`).

## Running experiments

Dispatched via [launch-experiment](https://github.com/etservicehub23-code/launch-experiment)
from any of home/work/laptop/eserver:

```bash
launch-experiment run playground "cd experiments/<name> && python src/main.py" --experiment <name>
```

The `--experiment <name>` flag matters — it's what lets
`sync-experiment-results` find and pull back `experiments/<name>/results/`
into `research/experiments/<job_id>/` once the run finishes. See that repo's
README for the full workflow.
