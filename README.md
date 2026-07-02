# notes-pde-opt

Experimental code for Neural Operator-enabled Topology-informed Evolutionary
Strategy for PDE-constrained optimization.

## Repository layout

```text
docs/                 Notes, dataset details, and exported reports
scripts/              Runnable experiment entry points
scripts/legacy/       Older standalone experiment scripts
scripts/notebooks/    Python exports generated from archived notebooks
src/                  Reusable project code
tools/                Maintenance utilities
```

Large local artifacts such as datasets, trained models, CMA-ES logs, images, and
notebook archives are ignored by git. The original notebooks were archived under
`notebooks/archive/`; regenerate their script exports with:

```bash
python tools/convert_notebooks.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Common commands

Train a NOTES DeepONet model:

```bash
python scripts/train_notes.py --dataid 0
```

Run CMA-ES optimization with a trained model:

```bash
python scripts/run_notes_cmaes.py --angle 65 --wavelength 850 --dataid 0
```

## TODO:
Create a outline for each computational experiment.
